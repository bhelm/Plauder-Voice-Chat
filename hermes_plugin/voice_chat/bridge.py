"""Bridge WebSocket server between the Hermes gateway and the voice chat.

Runs INSIDE the gateway process (started by ``VoiceChatAdapter.connect()``).
The plauder server connects as a WebSocket *client* (its ``hermes_gateway``
LLM backend) and keeps the connection open — that persistent link is what
enables asynchronous push delivery (delegate_task results, cron, terminal
notify_on_complete) into the voice chat, which the stateless ``/v1`` HTTP
path can never do.

Wire protocol (JSON text frames, both sides ignore unknown types):

  plauder -> gateway:
    {"type": "hello", "token": "...", "chat_id": "default",
     "client": "plauder", "proto": 1}
    {"type": "user.message", "turn_id": "ab12cd", "text": "...",
     "modality": "voice"|"text", "image_urls": ["data:image/...;base64,…"]}
    {"type": "session.reset"}
        "New Session" in the voice UI -> gateway rotates its session (/new)
    {"type": "push.undelivered", "text": "...", "played_s": 0.0, "ts": 123.4}
        a background push was cancelled by a barge-in before the user could
        hear it -> stashed per chat; the NEXT turn's channel_prompt tells the
        agent so it can weave the content into its answer (see adapter.py)

  gateway -> plauder:
    {"type": "hello.ok", "proto": 1, "platform": "voice_chat"}
    {"type": "agent.message", "text": "...", "turn_id": "ab12cd"|null,
     "push": bool, "message_id": "...",
     "speak": bool (optional, default true), "system": bool (optional)}
        turn_id set   -> a NEW message belonging to that turn (initial
                         streaming chunk or a complete reply)
        turn_id null  -> unsolicited push (speak it)
        speak=false   -> gateway system notice: show a bubble, no TTS
    {"type": "agent.partial", "turn_id": "ab12cd", "message_id": "...",
     "text": "<full accumulated text>", "finalize": bool}
        progressive token streaming: text REPLACES the message's content
        (grow-only while streaming; a finalize may reformat)
    {"type": "turn.done", "turn_id": "ab12cd", "status": "ok"|...}

This module is deliberately free of ``gateway.*`` imports so the voice-chat
repo can unit-test the protocol without the Hermes codebase
(``tests/test_hermes_bridge.py`` there). Only ``adapter.py`` touches Hermes.
"""

from __future__ import annotations

import asyncio
import collections
import hmac
import json
import logging
import uuid
from typing import Awaitable, Callable, Deque, Dict, Optional

from aiohttp import WSMsgType, web

LOG = logging.getLogger("voice_chat.bridge")

PROTO_VERSION = 1
HELLO_TIMEOUT_S = 10.0
#: WS close code for a failed handshake (bad/missing token).
CLOSE_UNAUTHORIZED = 4401
#: Per-chat cap of frames queued while the voice client is disconnected.
QUEUE_MAX = 50
#: Per-chat cap of pending undelivered-push notes (FIFO, oldest dropped).
UNDELIVERED_MAX = 5
#: Frame size cap — user.message frames may carry base64 data-URL images.
MAX_MSG_BYTES = 32 * 1024 * 1024
#: Leading emoji of the gateway's hardcoded system texts ("♻️ Gateway
#: online…", "⚠️ Gateway restarting…", "⚡ Interrupting current task…",
#: "✨ Session reset!…"). Texts starting with one are shown as silent
#: bubbles instead of being spoken — safe because the voice channel prompt
#: forbids Antonia emojis, so real replies never start with one.
SYSTEM_EMOJI_PREFIXES = ("♻️", "⚠️", "⚡", "✨", "⏳", "🆕")


def is_system_text(text: str) -> bool:
    """Heuristic system-notice classifier (see SYSTEM_EMOJI_PREFIXES)."""
    return text.lstrip().startswith(SYSTEM_EMOJI_PREFIXES)


def format_undelivered_note(pending: list) -> str:
    """Render the agent-facing note for background pushes the user never (fully)
    heard, from a list of ``{"text", "played_s", ...}`` dicts. German because
    it is injected into the voice channel_prompt (like DEFAULT_CHANNEL_PROMPT),
    which is Bernd's spoken interface. Pure/side-effect-free (unit-tested).
    The push text is quoted in FULL — the facts are the whole point, so nothing
    is truncated. Returns "" when there is nothing worth noting."""
    lines = []
    for item in pending or []:
        text = str((item or {}).get("text") or "").strip()
        if not text:
            continue
        try:
            played = max(0.0, float((item or {}).get("played_s") or 0.0))
        except (TypeError, ValueError):
            played = 0.0
        prefix = (f"(nur {played:.0f}s vorgelesen) " if played >= 0.5
                  else "(nicht vorgelesen) ")
        lines.append(f"{prefix}»{text}«")
    if not lines:
        return ""
    return (
        "HINWEIS: Diese Hintergrund-Nachricht(en) von dir wurden dem Nutzer "
        "NICHT (bzw. nur kurz) vorgelesen — er kennt sie vermutlich nicht. "
        "Falls noch relevant, flechte die Kernpunkte in deine Antwort ein:\n"
        + "\n".join(lines))


class VoiceBridgeServer:
    """Localhost WS server the plauder client connects to.

    ``on_user_message(chat_id, frame)`` is awaited for every valid
    ``user.message`` frame; it must return fast (the adapter just spawns
    ``handle_message`` as a task).
    """

    def __init__(self, host: str, port: int, token: str, *,
                 on_user_message: Callable[[str, dict], Awaitable[None]],
                 on_session_reset: Optional[
                     Callable[[str], Awaitable[None]]] = None):
        self.host = host
        self.port = port
        self.token = token
        self._on_user_message = on_user_message
        self._on_session_reset = on_session_reset
        self._runner: Optional[web.AppRunner] = None
        #: chat_id -> live WebSocket (latest hello wins).
        self._conns: Dict[str, web.WebSocketResponse] = {}
        #: chat_id -> frames waiting for the next (re)connect.
        self._queues: Dict[str, Deque[dict]] = {}
        #: chat_id -> pending undelivered-push notes (barge-in cancelled a push
        #: before the user heard it); consumed one-shot into the next turn's
        #: channel_prompt. Bounded FIFO (UNDELIVERED_MAX).
        self._undelivered: Dict[str, Deque[dict]] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        async def _health(_request):
            return web.Response(text="ok")

        app = web.Application(client_max_size=1024 * 1024)
        app.router.add_get("/ws", self._ws_handler)
        app.router.add_get("/health", _health)
        app.router.add_post("/push", self._push_handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port,
                           reuse_address=True)
        await site.start()
        # port=0 (tests): record the actually bound port.
        if not self.port:
            server = site._server  # noqa: SLF001 - aiohttp exposes no API
            if server and server.sockets:
                self.port = server.sockets[0].getsockname()[1]
        LOG.info("voice-chat bridge listening on ws://%s:%s/ws",
                 self.host, self.port)

    async def stop(self) -> None:
        for ws in list(self._conns.values()):
            try:
                await ws.close()
            except Exception:
                pass
        self._conns.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def connected(self, chat_id: str) -> bool:
        ws = self._conns.get(chat_id)
        return ws is not None and not ws.closed

    # ------------------------------------------------------------------ #
    # Outbound
    # ------------------------------------------------------------------ #
    async def send_frame(self, chat_id: str, frame: dict) -> bool:
        """Deliver a frame to the live connection. False when not connected
        or the send fails (caller decides whether to queue)."""
        ws = self._conns.get(chat_id)
        if ws is None or ws.closed:
            return False
        try:
            await ws.send_json(frame)
            return True
        except Exception as exc:
            LOG.warning("bridge send to %r failed: %s", chat_id, exc)
            return False

    def queue_frame(self, chat_id: str, frame: dict) -> None:
        """Hold a frame for the next (re)connect. Bounded: oldest dropped."""
        q = self._queues.setdefault(chat_id,
                                    collections.deque(maxlen=QUEUE_MAX))
        if len(q) == q.maxlen:
            LOG.warning("bridge queue for %r full — dropping oldest frame",
                        chat_id)
        q.append(frame)

    async def _flush_queue(self, chat_id: str) -> None:
        q = self._queues.get(chat_id)
        while q:
            frame = q.popleft()
            if not await self.send_frame(chat_id, frame):
                q.appendleft(frame)   # connection died again — keep it
                return

    # ------------------------------------------------------------------ #
    # Undelivered-push notes (barge-in cancelled a push before it was heard)
    # ------------------------------------------------------------------ #
    def _record_undelivered(self, chat_id: str, frame: dict) -> None:
        """Stash a ``push.undelivered`` frame for the chat. Bounded FIFO —
        the oldest note is dropped once the cap is reached."""
        text = str(frame.get("text") or "").strip()
        if not text:
            return
        try:
            played = max(0.0, float(frame.get("played_s") or 0.0))
        except (TypeError, ValueError):
            played = 0.0
        q = self._undelivered.setdefault(
            chat_id, collections.deque(maxlen=UNDELIVERED_MAX))
        q.append({"text": text, "played_s": played,
                  "ts": frame.get("ts")})
        LOG.info("voice_chat: undelivered push stashed for %r (%.1fs): %r",
                 chat_id, played, text[:80])

    def consume_undelivered(self, chat_id: str) -> list:
        """Return and clear the chat's pending undelivered-push notes (one-shot,
        oldest first). Empty list when there is nothing pending."""
        q = self._undelivered.pop(chat_id, None)
        return list(q) if q else []

    async def _push_handler(self, request: web.Request) -> web.Response:
        """Out-of-process push entry (``hermes send``, standalone cron):
        token-authenticated HTTP POST -> agent.message push frame to the
        voice client (queued when it is offline)."""
        token = request.headers.get("X-Bridge-Token", "")
        if not self.token or not hmac.compare_digest(token, self.token):
            return web.json_response({"error": "bad token"}, status=401)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        text = str((body or {}).get("text") or "").strip()
        if not text:
            return web.json_response({"error": "text required"}, status=400)
        chat_id = str((body or {}).get("chat_id") or "default")
        frame = {"type": "agent.message", "text": text, "turn_id": None,
                 "push": True, "message_id": uuid.uuid4().hex[:12]}
        if is_system_text(text):
            frame["speak"] = False
            frame["system"] = True
        delivered = await self.send_frame(chat_id, frame)
        if not delivered:
            self.queue_frame(chat_id, frame)
        return web.json_response({"success": True, "queued": not delivered,
                                  "message_id": frame["message_id"]})

    # ------------------------------------------------------------------ #
    # Inbound
    # ------------------------------------------------------------------ #
    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=MAX_MSG_BYTES)
        await ws.prepare(request)

        chat_id = await self._handshake(ws)
        if chat_id is None:
            return ws

        prev = self._conns.get(chat_id)
        if prev is not None and not prev.closed:
            LOG.info("bridge: new connection for %r replaces the old one",
                     chat_id)
            try:
                await prev.close()
            except Exception:
                pass
        self._conns[chat_id] = ws
        await ws.send_json({"type": "hello.ok", "proto": PROTO_VERSION,
                            "platform": "voice_chat"})
        LOG.info("bridge: voice client connected (chat_id=%r)", chat_id)
        await self._flush_queue(chat_id)

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    if msg.type == WSMsgType.ERROR:
                        break
                    continue
                try:
                    frame = json.loads(msg.data)
                except json.JSONDecodeError:
                    LOG.warning("bridge: non-JSON frame from %r ignored",
                                chat_id)
                    continue
                if not isinstance(frame, dict):
                    continue
                ftype = frame.get("type")
                if ftype == "user.message":
                    try:
                        await self._on_user_message(chat_id, frame)
                    except Exception:
                        LOG.exception("bridge: user.message handler failed")
                elif ftype == "session.reset" and self._on_session_reset:
                    try:
                        await self._on_session_reset(chat_id)
                    except Exception:
                        LOG.exception("bridge: session.reset handler failed")
                elif ftype == "push.undelivered":
                    self._record_undelivered(chat_id, frame)
                # Unknown frame types: ignored (forward compatibility).
        finally:
            if self._conns.get(chat_id) is ws:
                del self._conns[chat_id]
            LOG.info("bridge: voice client disconnected (chat_id=%r)", chat_id)
        return ws

    async def _handshake(self, ws: web.WebSocketResponse) -> Optional[str]:
        """Expect a valid hello as the FIRST frame; returns the chat_id or
        None (connection closed) on any violation."""
        try:
            msg = await ws.receive(timeout=HELLO_TIMEOUT_S)
        except asyncio.TimeoutError:
            await ws.close(code=CLOSE_UNAUTHORIZED, message=b"hello timeout")
            return None
        if msg.type != WSMsgType.TEXT:
            await ws.close(code=CLOSE_UNAUTHORIZED, message=b"hello expected")
            return None
        try:
            hello = json.loads(msg.data)
        except json.JSONDecodeError:
            hello = None
        if (not isinstance(hello, dict) or hello.get("type") != "hello"
                or not isinstance(hello.get("token"), str)):
            await ws.close(code=CLOSE_UNAUTHORIZED, message=b"hello expected")
            return None
        if not self.token or not hmac.compare_digest(hello["token"],
                                                     self.token):
            LOG.warning("bridge: rejected connection (bad token)")
            await ws.close(code=CLOSE_UNAUTHORIZED, message=b"bad token")
            return None
        chat_id = str(hello.get("chat_id") or "default")
        return chat_id
