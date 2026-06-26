"""Telegram-Bridge: bidirektionale Text-Spiegelung Voice <-> Telegram.

Legacy-Feature (OpenClaw-CLI-basiert), per ENV (TELEGRAM_MIRROR) standardmäßig
AUS. Aus dem monolithischen server.py extrahiert; Logik unverändert. Wird nur
aktiv, wenn ein Telegram-Account/Target aufgelöst werden kann.

  Outbound: User-Input + Agent-Antwort werden als Bot-Nachricht in den Telegram-
            Chat gepostet (`openclaw message send`).
  Inbound:  Die Agent-Session-JSONL wird getailt; neue Messages werden als
            `external.message`-Frame an alle Browser-WebSockets gebroadcastet.
  Echo-Filter + Suppression-Window verhindern Doppelungen während Voice-Calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from pathlib import Path

LOG = logging.getLogger("voice-chat.telegram")


class TelegramBridge:
    def __init__(self, *, agent_id: str, account_id: str | None,
                 target_chat_id: str | None, openclaw_cli: str,
                 sessions_dir: Path):
        self.agent_id = agent_id
        self.account_id = account_id
        self.target_chat_id = target_chat_id
        self.openclaw_cli = openclaw_cli
        self.sessions_dir = Path(sessions_dir)
        self._recent_self_sent: deque[str] = deque(maxlen=128)
        self._local_call_depth: int = 0
        self._recent_seen_in_jsonl: set[str] = set()
        self._recent_broadcast_keys: deque[str] = deque(maxlen=64)
        self._recent_broadcast_set: set[str] = set()
        self._broadcast_channels: list = []
        self._watcher_task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def begin_local_call(self) -> None:
        self._local_call_depth += 1

    def end_local_call(self) -> None:
        async def _delayed():
            await asyncio.sleep(1.5)
            if self._local_call_depth > 0:
                self._local_call_depth -= 1
        asyncio.create_task(_delayed())

    @property
    def in_local_call(self) -> bool:
        return self._local_call_depth > 0

    @property
    def enabled(self) -> bool:
        return bool(self.account_id and self.target_chat_id)

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def remember_self_sent(self, text: str) -> None:
        norm = self._normalize(text)
        if norm:
            self._recent_self_sent.append(norm)

    def is_echo(self, text: str) -> bool:
        return self._normalize(text) in self._recent_self_sent

    def _dedup_broadcast(self, role: str, text: str) -> bool:
        key = f"{role}:{self._normalize(text)}"
        if key in self._recent_broadcast_set:
            return True
        self._recent_broadcast_keys.append(key)
        self._recent_broadcast_set.add(key)
        if len(self._recent_broadcast_set) > len(self._recent_broadcast_keys) + 8:
            self._recent_broadcast_set = set(self._recent_broadcast_keys)
        return False

    async def send(self, text: str, *, silent: bool = True,
                   echo_text: str | None = None) -> bool:
        if not self.enabled:
            return False
        text = (text or "").strip()
        if not text:
            return False
        self.remember_self_sent(echo_text if echo_text is not None else text)
        cmd = [
            str(self.openclaw_cli), "message", "send",
            "--channel", "telegram",
            "--account", str(self.account_id),
            "--target", str(self.target_chat_id),
            "--message", text,
            "--json",
        ]
        if silent:
            cmd.append("--silent")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode != 0:
                LOG.warning("telegram send failed rc=%s err=%r", proc.returncode, stderr[:200])
                return False
            return True
        except asyncio.TimeoutError:
            LOG.warning("telegram send timeout")
            return False
        except Exception as exc:
            LOG.exception("telegram send error: %s", exc)
            return False

    def register_broadcast(self, ws) -> None:
        if ws not in self._broadcast_channels:
            self._broadcast_channels.append(ws)

    def unregister_broadcast(self, ws) -> None:
        if ws in self._broadcast_channels:
            self._broadcast_channels.remove(ws)

    async def _broadcast(self, payload: dict) -> None:
        dead = []
        for ws in list(self._broadcast_channels):
            try:
                if ws.closed:
                    dead.append(ws)
                    continue
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister_broadcast(ws)

    def _find_active_session_file(self) -> Path | None:
        if not self.sessions_dir.is_dir():
            return None
        cands = [
            p for p in self.sessions_dir.glob("*.jsonl")
            if not p.name.endswith(".trajectory.jsonl")
            and not p.name.startswith("sessions.")
        ]
        if not cands:
            return None
        return max(cands, key=lambda p: p.stat().st_mtime)

    def _extract_message_text(self, msg: dict) -> tuple[str, str]:
        m = msg.get("message") or {}
        role = (m.get("role") or "").strip()
        content = m.get("content")
        if isinstance(content, str):
            return role, content
        if not isinstance(content, list):
            return role, ""
        text_parts = []
        has_image = False
        has_audio = False
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t == "text":
                text_parts.append(part.get("text") or "")
            elif t in ("image", "image_url"):
                has_image = True
            elif t in ("audio", "audio_url"):
                has_audio = True
        text = " ".join(p for p in text_parts if p).strip()
        if not text:
            if has_image:
                text = "📷 Bild gesendet"
            elif has_audio:
                text = "🎵 Audio gesendet"
        return role, text

    async def _tail_session_file(self) -> None:
        current_file: Path | None = None
        offset = 0
        init_file = self._find_active_session_file()
        if init_file:
            current_file = init_file
            offset = init_file.stat().st_size
            LOG.info("telegram-bridge tailing %s from offset %d", init_file, offset)
        else:
            LOG.info("telegram-bridge: no active session file yet, will retry")

        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=0.5)
                break
            except asyncio.TimeoutError:
                pass
            try:
                latest = self._find_active_session_file()
                if latest is None:
                    continue
                if current_file is None or latest != current_file:
                    LOG.info("telegram-bridge: switched session file -> %s", latest)
                    current_file = latest
                    offset = 0
                size = current_file.stat().st_size
                if size < offset:
                    offset = 0
                if size == offset:
                    continue
                with current_file.open("rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                    offset = f.tell()
                for raw in chunk.splitlines():
                    if not raw.strip():
                        continue
                    try:
                        d = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if d.get("type") != "message":
                        continue
                    msg_id = d.get("id")
                    if msg_id and msg_id in self._recent_seen_in_jsonl:
                        continue
                    if msg_id:
                        self._recent_seen_in_jsonl.add(msg_id)
                        if len(self._recent_seen_in_jsonl) > 512:
                            keep = list(self._recent_seen_in_jsonl)[-256:]
                            self._recent_seen_in_jsonl = set(keep)
                    role, text = self._extract_message_text(d)
                    if not role or not text:
                        continue
                    if self.in_local_call:
                        continue
                    if self.is_echo(text):
                        continue
                    if self._dedup_broadcast(role, text):
                        continue
                    LOG.info("telegram-bridge: external %s message: %r", role, text[:140])
                    await self._broadcast({
                        "type": "external.message",
                        "role": role,
                        "text": text,
                        "source": "telegram",
                        "ts": time.time(),
                    })
            except Exception as exc:
                LOG.exception("telegram-bridge tail error: %s", exc)
                await asyncio.sleep(2.0)

    def start(self) -> None:
        if self._watcher_task is None:
            self._stop.clear()
            self._watcher_task = asyncio.create_task(self._tail_session_file())
            LOG.info("telegram-bridge: watcher started (account=%s target=%s)",
                     self.account_id, self.target_chat_id)

    async def stop(self) -> None:
        self._stop.set()
        if self._watcher_task:
            try:
                await asyncio.wait_for(self._watcher_task, timeout=3)
            except Exception:
                pass
            self._watcher_task = None
