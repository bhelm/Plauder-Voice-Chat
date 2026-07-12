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
    resolve_channel_prompt,
)
from gateway.platforms.helpers import strip_markdown

from .bridge import VoiceBridgeServer

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
            on_user_message=self._on_user_message)
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
        frame = {
            "type": "agent.message",
            "text": self.format_message(content),
            "turn_id": turn_id,
            "push": turn_id is None,
            "message_id": message_id,
        }
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

    # ------------------------------------------------------------------ #
    # Formatting / lifecycle hooks
    # ------------------------------------------------------------------ #
    def format_message(self, content: str) -> str:
        """Markdown renders as literal characters in a spoken/plain channel."""
        return strip_markdown(content)

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
        # Image uploads arrive as URLs served by plauder; the agent can fetch
        # them with its tools. (media_urls expects local file paths, so URLs
        # travel inside the text instead.)
        image_urls = [u for u in (frame.get("image_urls") or [])
                      if isinstance(u, str)]
        if image_urls:
            text += "\n\n[Bilder aus dem Voice-Chat: " + " ".join(image_urls) + "]"
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=turn_id,
            metadata={"vc_turn_id": turn_id, "modality": modality},
        )
        event.channel_prompt = self._channel_prompt_for(chat_id)
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


def _build_adapter(config: PlatformConfig) -> VoiceChatAdapter:
    return VoiceChatAdapter(config)


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
        emoji="🎙️",
    )
