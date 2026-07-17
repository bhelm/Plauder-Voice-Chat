"""Voice-chat (plauder) platform adapter for the Hermes gateway.

Makes the browser voice chat a first-class gateway channel: inbound
transcripts become ``MessageEvent``s (full agent pipeline, tools, session
history in the gateway's SessionDB), outbound replies AND asynchronous
deliveries (delegate_task background results, cron, notify_on_complete) go
through ``send()`` over a persistent localhost WebSocket to plauder, which
does its own TTS. The adapter never touches audio — text only.

Env vars (gateway side, ~/.hermes/.env):
  - VOICE_CHAT_BRIDGE_TOKEN   shared secret for the bridge WS (required)
  - VOICE_CHAT_WS_HOST        bridge bind host (default 127.0.0.1)
  - VOICE_CHAT_WS_PORT        bridge bind port (default 8321)
  - VOICE_CHAT_USER_ID        user id for the voice session (default "bernd")
  - VOICE_CHAT_USER_NAME      display name (default = user id, capitalized)
  - VOICE_CHAT_CHANNEL_PROMPT override the voice-mode channel prompt
                              ("-" disables it entirely)
  - VOICE_CHAT_HOME_CHANNEL   chat_id for bare `deliver=voice_chat` cron
                              targets (default "default")
  - VOICE_CHAT_ALLOWED_USERS / VOICE_CHAT_ALLOW_ALL_USERS  gateway auth

Turn correlation: every inbound ``user.message`` carries a ``turn_id``;
``on_processing_start`` marks it active for the chat, ``send()`` tags reply
frames with it, and ``on_processing_complete`` emits ``turn.done``. A
``send()`` with NO active turn is an asynchronous push (``turn_id: null``)
— plauder speaks it. Frames that cannot be delivered are queued in the
bridge and flushed on the next reconnect, so deliveries survive voice-chat
restarts.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_image_from_url,
    resolve_channel_prompt,
)

from .bridge import (
    VoiceBridgeServer,
    format_undelivered_note,
    is_system_text,
)

logger = logging.getLogger(__name__)

PLATFORM_NAME = "voice_chat"
DEFAULT_WS_HOST = "127.0.0.1"
DEFAULT_WS_PORT = 8321

# German because the voice channel is Bernd's spoken interface — this is an
# agent-facing rule line, delivered per turn via MessageEvent.channel_prompt.
DEFAULT_CHANNEL_PROMPT = (
    "VOICE-MODE: Diese Nachricht kommt aus dem Sprach-Chat, deine Antwort "
    "wird vorgelesen. Antworte knapp, in natürlichen gesprochenen Sätzen. "
    "Keine Emojis, kein Markdown, keine Code-Blöcke, keine Tabellen. Keine "
    "URLs vorlesen — beschreibe stattdessen in Worten, worum es geht."
)


def _is_system_message(content: str,
                       metadata: Optional[Dict[str, Any]]) -> bool:
    """Gateway system notice? Clean metadata flag first (the local core
    patch marks lifecycle sends for voice_chat with non_conversational),
    emoji-prefix heuristic (bridge.is_system_text) as fallback for paths
    the core never marks — and after a hermes update reverts the patch."""
    if metadata and metadata.get("non_conversational"):
        return True
    return is_system_text(content)


def _strip_cursor(text: str) -> str:
    """Drop a trailing streaming cursor. The GatewayStreamConsumer appends it
    to mid-stream content — and the FIRST chunk of a stream arrives via
    send(), not edit_message(), so both outbound paths must strip it."""
    text = text.rstrip()
    for cursor in ("▉", "▌", "█"):
        if text.endswith(cursor):
            return text[: -len(cursor)].rstrip()
    return text


def _env(name: str, default: str = "") -> str:
    """Env lookup that prefers the gateway's .env store (mirrors how other
    plugin adapters resolve config) and falls back to the process env."""
    try:
        import hermes_cli.gateway as gateway_mod
        val = gateway_mod.get_env_value(name)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name, default)


def check_requirements() -> bool:
    """aiohttp ships with the gateway; the token is the real gate."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return True


def _is_connected(config) -> bool:
    """Configured = the bridge token is set."""
    return bool(_env("VOICE_CHAT_BRIDGE_TOKEN").strip())


class VoiceChatAdapter(BasePlatformAdapter):
    """Text bridge between the gateway and the plauder voice-chat server."""

    #: Voice replies are spoken — fenced code renders as gibberish.
    supports_code_blocks = False
    #: Enables the GatewayStreamConsumer edit path (progressive
    #: agent.partial frames) when display.platforms.voice_chat.streaming
    #: is on — plauder starts TTS on the first complete sentence.
    SUPPORTS_MESSAGE_EDITING = True
    # supports_async_delivery stays True (base default): the persistent
    # bridge WS is exactly the outbound channel background tasks need.

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform(PLATFORM_NAME))
        self._token = _env("VOICE_CHAT_BRIDGE_TOKEN").strip()
        self._host = _env("VOICE_CHAT_WS_HOST", DEFAULT_WS_HOST)
        self._port = int(_env("VOICE_CHAT_WS_PORT", str(DEFAULT_WS_PORT)))
        self._user_id = _env("VOICE_CHAT_USER_ID", "bernd").strip() or "bernd"
        self._user_name = (_env("VOICE_CHAT_USER_NAME").strip()
                           or self._user_id.capitalize())
        self._bridge: Optional[VoiceBridgeServer] = None
        #: chat_id -> turn_id of the message currently being processed.
        self._active_turns: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Required abstract methods
    # ------------------------------------------------------------------ #
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self._token:
            msg = ("[voice_chat] VOICE_CHAT_BRIDGE_TOKEN not set — refusing "
                   "to start an unauthenticated bridge")
            logger.error(msg)
            self._set_fatal_error("voice_chat_missing_token", msg,
                                  retryable=False)
            return False
        self._bridge = VoiceBridgeServer(
            self._host, self._port, self._token,
            on_user_message=self._on_user_message,
            on_session_reset=self._on_session_reset)
        try:
            await self._bridge.start()
        except OSError as exc:
            msg = f"[voice_chat] bridge bind {self._host}:{self._port} failed: {exc}"
            logger.error(msg)
            self._set_fatal_error("voice_chat_bind_failed", msg,
                                  retryable=True)
            return False
        self._running = True
        return True

    async def disconnect(self) -> None:
        if self._bridge is not None:
            await self._bridge.stop()
            self._bridge = None
        self._running = False
        logger.info("[voice_chat] disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._bridge is None:
            return SendResult(success=False, error="voice_chat bridge not running",
                              retryable=True, error_kind="transient")
        message_id = uuid.uuid4().hex[:12]
        turn_id = self._active_turns.get(chat_id)
        # System notices (restart/online, busy acks, /new confirmation) are
        # shown in the chat but NOT spoken — and never routed into a turn
        # stream, even if one happens to be active.
        system = _is_system_message(content, metadata)
        if system:
            turn_id = None
        # No markdown stripping: plauder renders markdown in the bubble and
        # its TTS sanitizer strips it per sentence anyway.
        frame = {
            "type": "agent.message",
            "text": _strip_cursor(content),
            "turn_id": turn_id,
            "push": turn_id is None,
            "message_id": message_id,
        }
        if system:
            frame["speak"] = False
            frame["system"] = True
        if not await self._bridge.send_frame(chat_id, frame):
            # Voice client currently down (restart, deploy): queue — the
            # frame is flushed on the next reconnect and, lacking a live
            # turn by then, is spoken as a push. The gateway transcript has
            # the message either way, so this counts as delivered.
            self._bridge.queue_frame(chat_id, frame)
            logger.info("[voice_chat] client offline — queued message %s",
                        message_id)
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "Voice-Chat", "type": "dm", "chat_id": chat_id}

    async def edit_message(self, chat_id: str, message_id: str, content: str,
                           *, finalize: bool = False) -> SendResult:
        """Progressive streaming edit from the GatewayStreamConsumer: the
        accumulated text so far, addressed to the message send() created.
        Plauder computes the suffix delta and feeds its sentence-wise TTS."""
        if self._bridge is None:
            return SendResult(success=False, error="voice_chat bridge not running",
                              retryable=True, error_kind="transient")
        # Mid-stream edits carry the streaming cursor — never speak/render it.
        text = _strip_cursor(content)
        if is_system_text(text):
            # Streamed system notice: only the final state is worth showing,
            # silently. Mid-stream growth is dropped.
            if finalize and text:
                await self._bridge.send_frame(chat_id, {
                    "type": "agent.message", "text": text, "turn_id": None,
                    "push": True, "speak": False, "system": True,
                    "message_id": message_id,
                })
            return SendResult(success=True, message_id=message_id)
        frame = {
            "type": "agent.partial",
            "turn_id": self._active_turns.get(chat_id),
            "message_id": message_id,
            "text": text,
            "finalize": finalize,
        }
        if not await self._bridge.send_frame(chat_id, frame):
            if finalize:
                # Client died mid-stream: preserve the final content — it is
                # flushed on reconnect and spoken as a push.
                self._bridge.queue_frame(chat_id, {
                    "type": "agent.message", "text": text,
                    "turn_id": None, "push": True, "message_id": message_id,
                })
            # Non-final partials are worthless after a disconnect: drop.
        return SendResult(success=True, message_id=message_id)

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #

    async def on_processing_start(self, event: MessageEvent) -> None:
        turn_id = (event.metadata or {}).get("vc_turn_id")
        if turn_id and event.source is not None:
            self._active_turns[event.source.chat_id] = turn_id

    async def on_processing_complete(self, event: MessageEvent, outcome) -> None:
        if event.source is None:
            return
        chat_id = event.source.chat_id
        turn_id = self._active_turns.pop(chat_id, None)
        if not turn_id or self._bridge is None:
            return
        status = getattr(outcome, "value", None) or str(outcome)
        # turn.done for a dead connection is useless (a reconnected client
        # has no such pending turn) — send best-effort, never queue.
        await self._bridge.send_frame(chat_id, {
            "type": "turn.done", "turn_id": turn_id, "status": status,
        })

    # ------------------------------------------------------------------ #
    # Inbound from plauder
    # ------------------------------------------------------------------ #
    def _channel_prompt_for(self, chat_id: str) -> Optional[str]:
        override = _env("VOICE_CHAT_CHANNEL_PROMPT").strip()
        if override == "-":
            return None
        if override:
            return override
        try:
            configured = resolve_channel_prompt(self.config.extra or {}, chat_id)
        except Exception:
            configured = None
        return configured or DEFAULT_CHANNEL_PROMPT

    async def _on_user_message(self, chat_id: str, frame: dict) -> None:
        text = str(frame.get("text") or "").strip()
        if not text:
            return
        turn_id = str(frame.get("turn_id") or uuid.uuid4().hex[:8])
        modality = frame.get("modality") or "voice"
        source = self.build_source(
            chat_id=chat_id,
            chat_name="Voice-Chat",
            chat_type="dm",
            user_id=self._user_id,
            user_name=self._user_name,
        )
        # Image uploads arrive as data URLs (plauder resolves /uploads/…
        # before the LLM call); cache them as local files so the agent gets
        # real vision attachments (media_urls = paths, media_types = MIMEs).
        media_urls, media_types = [], []
        for url in (frame.get("image_urls") or []):
            if not isinstance(url, str):
                continue
            cached, mime = await self._cache_image(url)
            if cached:
                media_urls.append(cached)
                media_types.append(mime)
        event = MessageEvent(
            text=text,
            message_type=MessageType.PHOTO if media_urls else MessageType.TEXT,
            source=source,
            message_id=turn_id,
            media_urls=media_urls,
            media_types=media_types,
            metadata={"vc_turn_id": turn_id, "modality": modality},
        )
        channel_prompt = self._channel_prompt_for(chat_id)
        # A background push cancelled by a barge-in before the user heard it
        # (push.undelivered): append an agent-facing note so the agent can weave
        # the content into THIS answer. Ephemeral per-turn context (like the
        # voice-mode rules) — NOT prepended to the user text, which would
        # pollute the gateway transcript. One-shot consumption.
        if self._bridge is not None:
            note = format_undelivered_note(
                self._bridge.consume_undelivered(chat_id))
            if note:
                channel_prompt = (f"{channel_prompt}\n\n{note}"
                                  if channel_prompt else note)
        event.channel_prompt = channel_prompt
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _on_session_reset(self, chat_id: str) -> None:
        """"New Session" in the voice UI: run an internal /new command so
        the gateway rotates its SessionDB session for this chat. The
        confirmation reply arrives without a live turn -> spoken as push."""
        logger.info("[voice_chat] session reset requested (chat_id=%r)",
                    chat_id)
        self._active_turns.pop(chat_id, None)
        source = self.build_source(
            chat_id=chat_id,
            chat_name="Voice-Chat",
            chat_type="dm",
            user_id=self._user_id,
            user_name=self._user_name,
        )
        event = MessageEvent(
            text="/new",
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
            metadata={"vc_control": "session.reset"},
        )
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    _MIME_EXT = {"image/jpeg": ".jpg", "image/png": ".png",
                 "image/webp": ".webp", "image/gif": ".gif"}

    async def _cache_image(self, url: str) -> tuple[Optional[str], str]:
        """data:/http(s) image URL -> (local cached path, mime). (None, "")
        when undecodable — the turn proceeds without that attachment."""
        try:
            if url.startswith("data:"):
                header, _, b64 = url.partition(",")
                if not b64:
                    return None, ""
                mime = header[5:].split(";", 1)[0].strip().lower() or "image/jpeg"
                data = base64.b64decode(b64, validate=False)
                path = cache_image_from_bytes(
                    data, ext=self._MIME_EXT.get(mime, ".jpg"))
                return path, mime
            if url.startswith(("http://", "https://")):
                path = await cache_image_from_url(url)
                return path, "image/jpeg"
        except Exception as exc:
            logger.warning("[voice_chat] image attachment dropped: %s", exc)
        return None, ""


def _build_adapter(config: PlatformConfig) -> VoiceChatAdapter:
    return VoiceChatAdapter(config)


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None,
                           media_files=None, force_document=False):
    """Out-of-process delivery (``hermes send``, cron without a live
    gateway... in THIS process): POST to the bridge's /push endpoint —
    the bridge lives in the gateway process and holds the WS to plauder."""
    import aiohttp

    token = _env("VOICE_CHAT_BRIDGE_TOKEN").strip()
    if not token:
        return {"error": "voice_chat not configured (VOICE_CHAT_BRIDGE_TOKEN)"}
    host = _env("VOICE_CHAT_WS_HOST", DEFAULT_WS_HOST)
    port = _env("VOICE_CHAT_WS_PORT", str(DEFAULT_WS_PORT))
    url = f"http://{host}:{port}/push"
    try:
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.post(
                    url, json={"chat_id": chat_id or "default",
                               "text": message},
                    headers={"X-Bridge-Token": token}) as resp:
                body = await resp.json()
                if resp.status != 200:
                    return {"error": f"bridge /push {resp.status}: {body}"}
                return {"success": True, "platform": "voice_chat",
                        "chat_id": chat_id or "default",
                        "message_id": body.get("message_id", ""),
                        "queued": body.get("queued", False)}
    except Exception as exc:
        return {"error": f"voice_chat bridge unreachable ({url}): {exc} — "
                         "is the gateway running?"}


def _env_enablement() -> dict:
    """Seed the platform's home channel so bare targets (``hermes send -t
    voice_chat``, ``deliver=voice_chat``) resolve without an explicit chat."""
    home = _env("VOICE_CHAT_HOME_CHANNEL", "default").strip() or "default"
    return {"home_channel": {"chat_id": home, "name": "Voice-Chat"}}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name=PLATFORM_NAME,
        label="Voice-Chat (Plauder)",
        adapter_factory=_build_adapter,
        check_fn=check_requirements,
        is_connected=_is_connected,
        required_env=["VOICE_CHAT_BRIDGE_TOKEN"],
        allowed_users_env="VOICE_CHAT_ALLOWED_USERS",
        allow_all_env="VOICE_CHAT_ALLOW_ALL_USERS",
        cron_deliver_env_var="VOICE_CHAT_HOME_CHANNEL",
        env_enablement_fn=_env_enablement,
        standalone_sender_fn=_standalone_send,
        emoji="🎙️",
    )
