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

  gateway -> plauder:
    {"type": "hello.ok", "proto": 1, "platform": "voice_chat"}
    {"type": "agent.message", "text": "...", "turn_id": "ab12cd"|null,
     "push": bool, "message_id": "..."}
        turn_id set   -> a NEW message belonging to that turn (initial
                         streaming chunk or a complete reply)
        turn_id null  -> unsolicited push (speak it)
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
#: Frame size cap — user.message frames may carry base64 data-URL images.
MAX_MSG_BYTES = 32 * 1024 * 1024


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
