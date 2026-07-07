"""Voice library client + active-voice persistence.

Mediates between the browser and the OmniVoice wrapper's ``/v1/audio/voices``
CRUD API (the browser never talks to the GPU box directly). The library itself —
reference WAVs + metadata — lives on the wrapper; here we only proxy the calls
and remember which voice is currently *active* for the whole session.

The active voice is a single global selection (all connected devices share one
session). Its id is persisted to a small state file (same idea as the rotated
Hermes session id in ``turn_state.py``) so it survives app restarts and applies
to new sessions. An unknown/stale id degrades gracefully: the wrapper falls back
to its built-in ``default`` voice, so audio never breaks.

Only wired up when ``TTS_CLONE_ENABLED`` is set AND TTS points at the wrapper
(plain OpenAI TTS can't clone). ``VoiceLibrary`` is otherwise never constructed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiohttp

LOG = logging.getLogger("voice-chat")

DEFAULT_VOICE_ID = "default"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class VoiceLibrary:
    def __init__(self, base_url: str, api_key: str = "", state_path: str = ""):
        # base_url is the OpenAI-compatible root incl. /v1 (CFG.tts_openai_base_url),
        # e.g. http://gpu-box:8880/v1 → voices CRUD lives at {base}/audio/voices.
        self._base = (base_url or "").rstrip("/")
        self._api_key = api_key or ""
        self._state_path = Path(state_path) if state_path else _PROJECT_ROOT / ".active_voice"
        self._active: str | None = None  # in-memory cache of the persisted id

    # ---- HTTP helpers -----------------------------------------------------
    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _url(self, path: str) -> str:
        return f"{self._base}/audio/voices{path}"

    async def list(self) -> list[dict]:
        """All voices known to the wrapper (default first). Raises on transport error."""
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(self._url(""), headers=self._headers) as r:
                r.raise_for_status()
                data = await r.json()
        return list(data.get("voices", []))

    async def register(self, data: bytes, *, filename: str, content_type: str,
                       name: str, ref_text: str) -> dict:
        """Register a recorded/uploaded sample as a new cloned voice. Returns the
        voice dict ({id,name,created,isDefault}). Cloning is a one-time GPU cost,
        so the timeout is generous."""
        form = aiohttp.FormData()
        form.add_field("file", data, filename=filename or "sample.wav",
                       content_type=content_type or "application/octet-stream")
        form.add_field("name", name or "")
        form.add_field("ref_text", ref_text or "")
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(self._url(""), data=form, headers=self._headers) as r:
                body = await r.json()
                if r.status >= 400:
                    raise VoiceApiError(body.get("error", {}).get("message", f"HTTP {r.status}"))
                return body

    async def rename(self, voice_id: str, name: str) -> dict:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.patch(self._url(f"/{voice_id}"), json={"name": name},
                               headers=self._headers) as r:
                body = await r.json()
                if r.status >= 400:
                    raise VoiceApiError(body.get("error", {}).get("message", f"HTTP {r.status}"))
                return body

    async def delete(self, voice_id: str) -> None:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.delete(self._url(f"/{voice_id}"), headers=self._headers) as r:
                if r.status >= 400:
                    body = await r.json()
                    raise VoiceApiError(body.get("error", {}).get("message", f"HTTP {r.status}"))

    # ---- active-voice persistence ----------------------------------------
    def get_active(self) -> str:
        """The globally selected voice id (cached; falls back to DEFAULT_VOICE_ID)."""
        if self._active is not None:
            return self._active
        try:
            sid = self._state_path.read_text(encoding="utf-8").strip()
        except OSError:
            sid = ""
        self._active = sid or DEFAULT_VOICE_ID
        return self._active

    def set_active(self, voice_id: str) -> str:
        """Persist the active voice id (atomically). A write failure is non-fatal
        — the choice still applies to the running process, it just won't survive
        a restart."""
        vid = (voice_id or "").strip() or DEFAULT_VOICE_ID
        self._active = vid
        try:
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(vid, encoding="utf-8")
            tmp.replace(self._state_path)
        except OSError as exc:
            LOG.warning("could not persist active voice id: %s", exc)
        return vid

    async def active_meta(self) -> dict:
        """{id,name} of the active voice for the hello frame. Best-effort: on any
        wrapper error it reports the raw id so hello still succeeds."""
        vid = self.get_active()
        try:
            for v in await self.list():
                if v.get("id") == vid:
                    return {"id": vid, "name": v.get("name", vid)}
        except Exception as exc:  # noqa: BLE001
            LOG.debug("active_meta: could not list voices: %s", exc)
        return {"id": vid, "name": vid}


class VoiceApiError(Exception):
    """Raised for 4xx/5xx responses from the wrapper voice API (message surfaced to the client)."""
