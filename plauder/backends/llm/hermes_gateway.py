"""LLM backend: persistent WebSocket to the Hermes gateway's voice_chat
platform adapter (see hermes_plugin/ in this repo).

Instead of a stateless POST to /v1/chat/completions, this backend keeps ONE
WebSocket open to the gateway's bridge server. A turn sends the LAST user
message as a ``user.message`` frame (the gateway keeps the conversation
history itself — the rest of the local message list is display-only) and
yields ``agent.message`` frames tagged with the turn id until ``turn.done``.

Frames WITHOUT a matching turn id are asynchronous pushes (background task
results, cron deliveries): they are handed to ``set_push_handler``'s
callback, which the server wires to its speak-a-push machinery. That push
path is the whole reason this backend exists — the HTTP path can never
deliver anything after its response closed.

The connection manager reconnects forever with capped backoff; a turn
started while disconnected fails fast (the server's retry/error path
handles it like any upstream failure).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from aiohttp import ClientSession, WSMsgType

from ..base import BackendError, UpstreamTimeoutError
from .base import LLMBackend

LOG = logging.getLogger("voice-chat.hermes-gateway")

PROTO_VERSION = 1
CONNECT_WAIT_S = 10.0      # how long a turn waits for the link before failing
HANDSHAKE_TIMEOUT_S = 10.0
BACKOFF_MAX_S = 15.0
PENDING_PUSH_MAX = 20      # pushes buffered before the handler is wired
# Debounce window per pushed message_id: a streamed orphan reply arrives as
# an initial agent.message chunk plus growing agent.partial frames — without
# coalescing, the chunk AND the final text would both be spoken ("Session"
# … "Session wiederhergestellt …", the 12.07. double-speak bug).
PUSH_DEBOUNCE_S = 1.2


class HermesGatewayLLMBackend(LLMBackend):
    def __init__(self, *, url: str, token: str, chat_id: str = "default",
                 timeout: int = 300):
        self.url = url
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        # Written by server._apply_hermes_headers (LLM singleton contract);
        # a non-empty session_id also tells _hermes_keeps_history() that the
        # gateway owns the transcript. Not sent anywhere by this backend.
        self.session_key = ""
        self.session_id = ""
        self.last_meta: dict = {}
        self._session: ClientSession | None = None
        self._ws = None
        self._conn_task: asyncio.Task | None = None
        self._connected = asyncio.Event()
        self._closed = False
        #: turn_id -> queue of ("text"|"done"|"error", payload)
        self._turns: dict[str, asyncio.Queue] = {}
        self._push_handler = None
        self._pending_pushes: list[tuple[str, bool]] = []
        self._push_tasks: set = set()
        #: message_id -> {"text", "speak", "task"} — debounce buffer that
        #: coalesces streamed push updates into ONE dispatched message.
        self._push_buf: dict[str, dict] = {}

    @classmethod
    def from_config(cls, cfg) -> "HermesGatewayLLMBackend":
        return cls(
            url=cfg.hermes_gateway_ws_url,
            token=cfg.hermes_gateway_token,
            chat_id=cfg.hermes_gateway_chat_id,
            timeout=cfg.llm_timeout,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def load(self) -> None:
        if not self.token:
            raise BackendError(
                "hermes_gateway needs HERMES_GATEWAY_TOKEN (the shared "
                "bridge secret, same value as the gateway's "
                "VOICE_CHAT_BRIDGE_TOKEN).")
        if not self.url:
            raise BackendError("hermes_gateway needs HERMES_GATEWAY_WS_URL.")
        self._session = ClientSession()
        self._conn_task = asyncio.create_task(self._run_connection())

    async def close(self) -> None:
        self._closed = True
        if self._conn_task is not None:
            self._conn_task.cancel()
            await asyncio.gather(self._conn_task, return_exceptions=True)
            self._conn_task = None
        for task in list(self._push_tasks):
            task.cancel()
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def describe(self) -> dict:
        return {
            "engine": "hermes_gateway",
            "url": self.url,
            "chat_id": self.chat_id,
            "connected": self.connected,
        }

    # ------------------------------------------------------------------ #
    # Push plumbing (server wires its speak function here)
    # ------------------------------------------------------------------ #
    def set_push_handler(self, handler) -> None:
        """``handler(text, speak)`` is awaited for every unsolicited agent
        message (speak=False -> display only, no TTS — gateway system
        notices). Pushes that arrived earlier are flushed immediately."""
        self._push_handler = handler
        pending, self._pending_pushes = self._pending_pushes, []
        for text, speak in pending:
            self._dispatch_push(text, speak)

    def _dispatch_push(self, text: str, speak: bool = True) -> None:
        if self._push_handler is None:
            self._pending_pushes.append((text, speak))
            del self._pending_pushes[:-PENDING_PUSH_MAX]
            return

        async def _run():
            try:
                await self._push_handler(text, speak)
            except Exception:
                LOG.exception("push handler failed")

        task = asyncio.create_task(_run())
        self._push_tasks.add(task)
        task.add_done_callback(self._push_tasks.discard)

    # --- push coalescing (see PUSH_DEBOUNCE_S) ------------------------- #
    def _buffer_push(self, mid: str, text: str, speak: bool,
                     *, flush: bool = False) -> None:
        entry = self._push_buf.get(mid)
        if entry is None:
            entry = self._push_buf[mid] = {"text": text, "speak": speak,
                                           "task": None}
        else:
            entry["text"] = text
            entry["speak"] = entry["speak"] and speak
        if entry["task"] is not None:
            entry["task"].cancel()
            entry["task"] = None
        if flush:
            self._flush_push(mid)
            return

        async def _timer():
            try:
                await asyncio.sleep(PUSH_DEBOUNCE_S)
            except asyncio.CancelledError:
                return
            self._flush_push(mid)

        entry["task"] = asyncio.create_task(_timer())
        self._push_tasks.add(entry["task"])
        entry["task"].add_done_callback(self._push_tasks.discard)

    def _flush_push(self, mid: str) -> None:
        entry = self._push_buf.pop(mid, None)
        if entry is None:
            return
        if entry["task"] is not None:
            entry["task"].cancel()
        text = entry["text"].strip()
        if text:
            self._dispatch_push(text, entry["speak"])

    async def notify_push_undelivered(self, text: str,
                                      played_s: float) -> None:
        """Tell the gateway a background push was NOT delivered to the user
        (barge-in cancelled it before he could hear it). The adapter stashes it
        per chat and appends a note to the NEXT turn's channel_prompt, so the
        agent can weave the content into its answer. Best-effort: with the
        bridge down we drop it (the next user.message wouldn't get through
        either). Sent over the same WS as user.message, and — because the
        interrupting turn only sends its user.message after the debounce
        window — this frame reliably precedes it."""
        ws = self._ws
        if ws is None:
            LOG.warning("notify_push_undelivered: bridge not connected — dropped")
            return
        try:
            await ws.send_json({"type": "push.undelivered", "text": text,
                                "played_s": played_s, "ts": time.time()})
        except Exception as exc:
            LOG.warning("notify_push_undelivered send failed: %s", exc)

    async def reset_session(self) -> None:
        """Ask the gateway for a fresh session ("New Session" button).
        The adapter turns the frame into an internal /new command that
        rotates the gateway's SessionDB session. Best-effort: with the
        bridge down, only the local reset happens (logged)."""
        ws = self._ws
        if ws is None:
            LOG.warning("reset_session: bridge not connected — "
                        "gateway session unchanged")
            return
        try:
            await ws.send_json({"type": "session.reset"})
        except Exception as exc:
            LOG.warning("reset_session send failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Connection manager
    # ------------------------------------------------------------------ #
    async def _run_connection(self) -> None:
        backoff = 1.0
        while not self._closed:
            ws = None
            try:
                ws = await self._session.ws_connect(
                    self.url, heartbeat=20,
                    max_msg_size=32 * 1024 * 1024)
                await ws.send_json({
                    "type": "hello", "token": self.token,
                    "chat_id": self.chat_id, "client": "plauder",
                    "proto": PROTO_VERSION,
                })
                msg = await ws.receive(timeout=HANDSHAKE_TIMEOUT_S)
                if msg.type != WSMsgType.TEXT:
                    raise RuntimeError(
                        f"bridge handshake rejected ({msg.type.name}"
                        f"{'' if msg.extra is None else ': ' + str(msg.extra)})")
                hello = json.loads(msg.data)
                if hello.get("type") != "hello.ok":
                    raise RuntimeError(f"unexpected handshake frame: {hello!r}")
                self._ws = ws
                self._connected.set()
                backoff = 1.0
                LOG.info("bridge connected: %s", self.url)
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        try:
                            frame = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(frame, dict):
                            self._handle_frame(frame)
                    elif msg.type == WSMsgType.ERROR:
                        break
                LOG.warning("bridge connection closed")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("bridge connect/receive failed: %s", exc)
            finally:
                self._connected.clear()
                self._ws = None
                if ws is not None and not ws.closed:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                self._fail_pending("bridge connection lost")
            if not self._closed:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)

    def _handle_frame(self, frame: dict) -> None:
        ftype = frame.get("type")
        if ftype in ("agent.message", "agent.partial"):
            turn_id = frame.get("turn_id")
            text = str(frame.get("text") or "")
            queue = self._turns.get(turn_id) if turn_id else None
            if queue is not None:
                kind = "msg" if ftype == "agent.message" else "partial"
                queue.put_nowait((kind, frame.get("message_id"), text))
            elif text.strip():
                # No (live) turn -> asynchronous push. Coalesced per
                # message_id (see PUSH_DEBOUNCE_S): a streamed orphan reply
                # (initial chunk + partial growth + finalize) is dispatched
                # ONCE with its final text instead of chunk-by-chunk.
                mid = str(frame.get("message_id") or uuid.uuid4().hex[:12])
                speak = frame.get("speak") is not False
                if ftype == "agent.message":
                    # System notices (speak=False) are single-shot: no
                    # stream will follow, skip the debounce delay.
                    self._buffer_push(mid, text, speak, flush=not speak)
                elif frame.get("finalize"):
                    self._buffer_push(mid, text, speak, flush=True)
                elif mid in self._push_buf:
                    self._buffer_push(mid, text, speak)
                # Mid-stream orphan partial with no buffered start: skip —
                # the finalize will carry the full text anyway.
        elif ftype == "turn.done":
            queue = self._turns.get(frame.get("turn_id"))
            if queue is not None:
                queue.put_nowait(("done", None,
                                  str(frame.get("status") or "ok")))
        # Unknown frame types: ignored (forward compatibility).

    def _fail_pending(self, reason: str) -> None:
        for queue in self._turns.values():
            queue.put_nowait(("error", None, reason))

    # ------------------------------------------------------------------ #
    # Chat interface
    # ------------------------------------------------------------------ #
    @staticmethod
    def _last_user_input(messages: list[dict]) -> tuple[str, list[str]]:
        """Text + image URLs of the LAST user message. The gateway keeps the
        session history itself — earlier local messages are not resent."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content, []
            if isinstance(content, list):
                texts, images = [], []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        texts.append(str(part.get("text") or ""))
                    elif part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url")
                        if url:
                            images.append(str(url))
                return " ".join(t for t in texts if t), images
        return "", []

    async def chat(self, messages: list[dict],
                   system_hint: str | None = None) -> str:
        parts = [d async for d in self.chat_stream(messages,
                                                   system_hint=system_hint)]
        return "".join(parts).strip()

    async def chat_stream(self, messages: list[dict],
                          system_hint: str | None = None):
        if self._session is None:
            raise RuntimeError("LLM not initialized (load() did not run)")
        try:
            await asyncio.wait_for(self._connected.wait(),
                                   timeout=CONNECT_WAIT_S)
        except asyncio.TimeoutError:
            raise RuntimeError(
                "hermes_gateway: bridge not connected (is the gateway "
                "running with the voice_chat platform?)") from None

        text, image_urls = self._last_user_input(messages)
        if not text and not image_urls:
            return

        turn_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue = asyncio.Queue()
        self._turns[turn_id] = queue
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        try:
            frame = {"type": "user.message", "turn_id": turn_id,
                     "text": text, "modality": "voice"}
            if image_urls:
                frame["image_urls"] = image_urls
            ws = self._ws
            if ws is None:
                raise RuntimeError("hermes_gateway: bridge not connected")
            await ws.send_json(frame)

            # Per-message accumulation: agent.message opens a message,
            # agent.partial replaces its content (grow-only while streaming).
            # Only the SUFFIX beyond what was already yielded is emitted, so
            # progressive edits stream into the TTS without double-speaking.
            seen: dict[str, str] = {}   # message_id -> text accounted for
            yielded_any = False
            anon = 0
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise UpstreamTimeoutError(
                        f"hermes_gateway: no reply within {self.timeout}s")
                try:
                    kind, mid, payload = await asyncio.wait_for(
                        queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise UpstreamTimeoutError(
                        f"hermes_gateway: no reply within {self.timeout}s"
                    ) from None
                if kind in ("msg", "partial"):
                    if mid is None:
                        anon += 1
                        mid = f"anon-{anon}"
                    prev = seen.get(mid)
                    if prev is None:                      # new message
                        seen[mid] = payload
                        if payload:
                            # Later messages of the same turn (split
                            # delivery, segment breaks) are read as
                            # separate paragraphs.
                            yield ("\n\n" + payload) if yielded_any else payload
                            yielded_any = True
                    elif payload == prev:
                        continue
                    elif payload.startswith(prev):        # grew — yield suffix
                        seen[mid] = payload
                        delta = payload[len(prev):]
                        if delta:
                            yield delta
                            yielded_any = True
                    else:
                        # Reformatted finalize (e.g. markdown cleanup): the
                        # content was already spoken — record, don't repeat.
                        seen[mid] = payload
                elif kind == "done":
                    break
                else:
                    raise RuntimeError(f"hermes_gateway: {payload}")
            self.last_meta = {"id": turn_id, "finish_reason": "stop",
                              "usage": {}}
        finally:
            self._turns.pop(turn_id, None)
