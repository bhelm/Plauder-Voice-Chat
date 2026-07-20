"""AI-Voice — the single owner of the voice the assistant speaks with.

One feature, two interchangeable SOURCES, selected by ``AI_VOICE_SOURCE``:

* ``local``   — OmniVoice runs in-process (``TTS_BACKEND=omnivoice_local``).
  The voice is either cloned from a local reference recording or *designed*
  from a free-text description (OmniVoice's ``instruct`` parameter). Switching
  happens on the loaded backend object.
* ``wrapper`` — OmniVoice runs as a separate HTTP service behind an
  OpenAI-compatible API (``TTS_BACKEND=openai`` + ``TTS_OPENAI_BASE_URL``).
  Voices are CRUD resources on that service; the plain HTTP client lives in
  ``plauder.voices`` (``VoiceLibrary``) and is transport only.
* ``auto`` (default) — wrapper when a library is wired, else local.

This module holds everything ABOVE the transport: which source is active, the
cloned/designed choice and its persistence, the clone/upload handlers, the
hello capability blocks and the state fan-out to every connected browser.

Runtime state (CFG/STT/TTS/VOICES, set by ``configure``) lives in
``plauder.server``; read at call time via ``server.<name>`` to avoid an import
cycle — same pattern as ``plauder.app``.

Sample management for locally cloned voices lands here too. Those recordings
are deliberately separate from the House-Mode speaker-ID samples: those
identify who is speaking, these reproduce a voice.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from aiohttp import web

from . import audio as audio_utils
from . import server
from .config import SAMPLE_RATE
from .voice_store import DEFAULT_VOICE_ID, LocalVoiceStore

LOG = logging.getLogger("voice-chat")

MAX_VOICE_UPLOAD_BYTES = 25 * 1024 * 1024

# Voice modes of the LOCAL source (they mirror OmniVoice's own modes).
MODES = ("clone", "design")
DEFAULT_STATE = {"mode": "clone", "instruct": ""}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATE_PATH = _PROJECT_ROOT / ".ai_voice.json"

_state: dict | None = None
_store: LocalVoiceStore | None = None


def store() -> LocalVoiceStore:
    """The local voice library (lazily built, rooted at VOICES_DIR)."""
    global _store
    if _store is None:
        root = getattr(server.CFG, "voices_dir", "") if server.CFG else ""
        _store = LocalVoiceStore(root or (_PROJECT_ROOT / "voices"))
    return _store

# One short, neutral sentence per UI language for the voice preview button.
_PREVIEW_SENTENCE = {
    "de": "Hallo, so klingt diese Stimme.",
    "en": "Hi, this is how this voice sounds.",
}


# --------------------------------------------------------------------------- #
# Source selection
# --------------------------------------------------------------------------- #
def _local_backend():
    """The TTS backend iff it can switch voices in-process.

    Duck-typed on purpose: any backend growing ``set_voice_mode`` participates,
    and the HTTP/wrapper backend simply never does.
    """
    tts = server.TTS
    return tts if tts is not None and hasattr(tts, "set_voice_mode") else None


def _library_wired() -> bool:
    return bool(server.CFG and server.CFG.tts_clone_enabled and server.VOICES is not None)


def source() -> str | None:
    """Active source: ``"wrapper"``, ``"local"`` or None when neither is usable.

    ``AI_VOICE_SOURCE`` pins a choice; an explicitly pinned source that is not
    actually wired resolves to None rather than silently falling back — a
    misconfiguration should be visible, not papered over.
    """
    # getattr: server.CFG is duck-typed in tests (only the fields under test)
    want = (getattr(server.CFG, "ai_voice_source", "auto") if server.CFG else "auto") or "auto"
    if want == "wrapper":
        return "wrapper" if _library_wired() else None
    if want == "local":
        return "local" if _local_backend() is not None else None
    if _library_wired():
        return "wrapper"
    return "local" if _local_backend() is not None else None


def library_active() -> bool:
    """The wrapper's voice library is usable (CRUD over HTTP)."""
    return source() == "wrapper"


def active_voice_id() -> str | None:
    """Voice id to pass into the TTS call, or None when the backend should use
    its own configured default (local source: the voice is backend state)."""
    return server.VOICES.get_active() if library_active() else None


# --------------------------------------------------------------------------- #
# Cloned vs. designed voice (local source) — persisted across restarts
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    """Persisted choice, falling back to the .env start defaults."""
    global _state
    if _state is not None:
        return _state
    cfg = server.CFG
    state = dict(DEFAULT_STATE)
    if cfg is not None:
        if cfg.omnivoice_mode in MODES:
            state["mode"] = cfg.omnivoice_mode
        state["instruct"] = cfg.omnivoice_instruct or ""
    try:
        with open(_STATE_PATH, encoding="utf-8") as fh:
            stored = json.load(fh)
        if stored.get("mode") in MODES:
            state["mode"] = stored["mode"]
        if isinstance(stored.get("instruct"), str):
            state["instruct"] = stored["instruct"]
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        LOG.warning("ai-voice state unreadable (%s) — using defaults", exc)
    _state = state
    return _state


def save_state(state: dict) -> None:
    """Atomic write (tmp + replace) so a crash mid-write cannot truncate it."""
    global _state
    _state = state
    tmp = _STATE_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, _STATE_PATH)
    except Exception:
        LOG.exception("could not persist ai-voice state")


async def _ensure_ref_text(voice_id: str) -> None:
    """Adopted files arrive without a transcript. Fill it in on first use via
    the local STT and persist it, so the cost is paid once — OmniVoice clones
    noticeably better with a matching ref_text than without one.

    A configured OMNIVOICE_REF_TEXT wins for the file it belongs to: the user
    typed it for exactly this recording, and it beats a fresh transcription.
    """
    v = store()._find(voice_id)
    if v is None:
        return
    sid = v.get("activeSample")
    smp = next((s for s in v.get("samples", []) if s["id"] == sid), None)
    if smp is None or smp.get("refText"):
        return
    path = store().sample_path(voice_id, smp)
    cfg_audio = getattr(server.CFG, "omnivoice_ref_audio", None) if server.CFG else None
    cfg_text = getattr(server.CFG, "omnivoice_ref_text", None) if server.CFG else None
    if cfg_audio and cfg_text:
        try:
            if Path(cfg_audio).resolve() == path.resolve():
                store().set_ref_text(voice_id, sid, cfg_text)
                return
        except Exception:  # noqa: BLE001
            pass
    if server.STT is None:
        return
    try:
        pcm = await _ffmpeg_decode_f32le(path.read_bytes(), SAMPLE_RATE)
        text = (await server.STT.transcribe(pcm, SAMPLE_RATE) or "").strip()
        if text:
            store().set_ref_text(voice_id, sid, text)
            LOG.info("ai-voice: transcribed adopted sample %s (%r)", path.name, text[:60])
    except Exception as exc:  # noqa: BLE001
        LOG.warning("ai-voice: could not transcribe %s: %s", path, exc)


async def bootstrap() -> None:
    """Boot-time setup of the local library: adopt loose reference WAVs and,
    on a fresh install, make the configured OMNIVOICE_REF_AUDIO the active
    voice — so the voice the server has always spoken with shows up selected
    instead of the user finding an empty library."""
    if source() != "local":
        return
    try:
        added = store().discover()
        if added:
            LOG.info("ai-voice: adopted %d existing reference recording(s): %s",
                     len(added), ", ".join(v["name"] for v in added))
    except Exception:
        LOG.exception("ai-voice: could not scan the voice directory")
        return
    cfg_audio = getattr(server.CFG, "omnivoice_ref_audio", None) if server.CFG else None
    if store().get_active() == DEFAULT_VOICE_ID and cfg_audio:
        match = store().find_by_path(cfg_audio)
        if match is not None:
            store().set_active(match["id"])
            LOG.info("ai-voice: active voice = %r (from OMNIVOICE_REF_AUDIO)",
                     match["name"])
    await apply_state()


async def apply_state(state: dict | None = None) -> None:
    """Pushes the current choice into the loaded TTS backend (local source).

    In clone mode the ACTIVE voice's reference sample goes along, so selecting
    a voice and switching modes are the same operation as far as the backend is
    concerned. No reference (built-in default or a voice without samples) means
    the backend keeps its configured one.
    """
    be = _local_backend()
    if be is None:
        return
    st = state or load_state()
    kw: dict = {"instruct": st.get("instruct") or None}
    if st["mode"] == "clone":
        vid = store().get_active()
        if vid != DEFAULT_VOICE_ID:
            await _ensure_ref_text(vid)
        ref = store().reference()
        if ref is not None:
            kw["ref_audio"], kw["ref_text"] = ref
    try:
        await be.set_voice_mode(st["mode"], **kw)
    except Exception:
        LOG.exception("could not apply ai-voice mode %s", st.get("mode"))


# --------------------------------------------------------------------------- #
# Hello / state fan-out
# --------------------------------------------------------------------------- #
async def voices_list() -> list[dict]:
    """The voice library of whichever source is active. Degrades gracefully —
    a slow/down wrapper yields an empty list instead of failing the caller."""
    src = source()
    if src == "local":
        return store().list()
    if src == "wrapper":
        try:
            return await asyncio.wait_for(server.VOICES.list(), timeout=5)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("voice library unreachable: %s", exc)
    return []


async def state_block() -> dict:
    """THE AI-Voice block — one payload for the hello frame and for every
    ``aivoice.state`` push. The section is a permanent part of the UI, so this
    always describes a usable state; the capability flags say what the active
    source can actually do rather than hiding the section.
    """
    src = source()
    st = load_state()
    return {
        "available": src is not None,
        "source": src or "none",
        # Voice design is an OmniVoice in-process feature; the wrapper only
        # serves pre-cloned voices.
        "canDesign": src == "local",
        # Recording/uploading new samples needs somewhere to put them.
        "canClone": src is not None,
        "canManageSamples": src == "local",
        "mode": st["mode"] if src == "local" else "clone",
        "instruct": st["instruct"],
        "active": (store().get_active() if src == "local"
                   else (server.VOICES.get_active() if src == "wrapper"
                         else DEFAULT_VOICE_ID)),
        "voices": await voices_list(),
    }


async def emit_state(origin_ws=None) -> None:
    """Push the fresh AI-Voice state to EVERY connected browser after a
    mutation (select/rename/delete/new sample/mode switch), so the shared
    session stays in sync — same idea as chat.remote / session.reset.remote."""
    payload = {"type": "aivoice.state", **(await state_block()), "ts": time.time()}
    if origin_ws is not None:
        try:
            await origin_ws.send_json(payload)
        except Exception:
            pass
    await server._broadcast_peers(origin_ws, payload)


def preview_sentence() -> str:
    lang = (server.CFG.app_language if server.CFG else "en")
    return _PREVIEW_SENTENCE.get(lang, _PREVIEW_SENTENCE["en"])


# --------------------------------------------------------------------------- #
# Library mutations — routed to the active source
# --------------------------------------------------------------------------- #
async def select_voice(voice_id: str) -> None:
    """Make a voice the active one. Local: also re-points the loaded backend at
    that voice's reference sample (selecting IS re-cloning, in-process)."""
    src = source()
    if src == "local":
        store().set_active(voice_id or DEFAULT_VOICE_ID)
        st = dict(load_state())
        st["mode"] = "clone"          # picking a voice means: use a cloned one
        save_state(st)
        await apply_state(st)
    elif src == "wrapper":
        server.VOICES.set_active(voice_id or DEFAULT_VOICE_ID)


async def rename_voice(voice_id: str, name: str) -> None:
    src = source()
    if src == "local":
        store().rename_voice(voice_id, name)
    elif src == "wrapper":
        await server.VOICES.rename(voice_id, name)


async def delete_voice(voice_id: str) -> None:
    src = source()
    if src == "local":
        was_active = store().get_active() == voice_id
        store().delete_voice(voice_id)
        if was_active:
            await apply_state()       # fell back to default — re-point the backend
    elif src == "wrapper":
        await server.VOICES.delete(voice_id)
        if server.VOICES.get_active() == voice_id:
            server.VOICES.set_active(DEFAULT_VOICE_ID)


async def create_voice(name: str) -> dict | None:
    """Add an empty voice and make it the active one, so the next recording or
    upload lands in it. Local source only — wrapper voices come into existence
    by cloning, there is nothing to create up front.

    An empty voice has no reference yet, so ``apply_state`` leaves the backend
    on its current one: the assistant keeps its voice until the first sample
    actually exists.
    """
    if source() != "local":
        return None
    voice = store().create_voice(name)
    await select_voice(voice["id"])
    return voice


async def set_active_sample(voice_id: str, sample_id: str) -> None:
    """Pick which take a voice clones from — re-applies at once if it is the
    voice currently speaking."""
    if source() != "local":
        return
    if store().set_active_sample(voice_id, sample_id) and \
            store().get_active() == voice_id:
        await apply_state()


async def delete_sample(voice_id: str, sample_id: str) -> None:
    if source() != "local":
        return
    if store().delete_sample(voice_id, sample_id) and \
            store().get_active() == voice_id:
        await apply_state()           # the active reference may have moved


def rename_sample(voice_id: str, sample_id: str, name: str) -> None:
    if source() == "local":
        store().rename_sample(voice_id, sample_id, name)


async def _pcm_to_sample(buf: bytes, *, voice_id: str, name: str) -> dict:
    """Shared tail of the recorded/uploaded sample paths (local source):
    clean the edges, transcribe for ref_text, store as WAV."""
    if server.CFG is not None and server.CFG.tts_clone_trim:
        cleaned, tinfo = audio_utils.trim_clone_reference(buf, SAMPLE_RATE)
        if cleaned is None:
            return {"ok": False, "error": tinfo["reason"] or "no_speech"}
        if tinfo["dropped_head"] or tinfo["dropped_tail"]:
            LOG.info("voice sample: dropped cut-off edge speech (head=%s tail=%s)",
                     tinfo["dropped_head"], tinfo["dropped_tail"])
        buf = cleaned
    # The transcript is taken from the CLEANED buffer so it matches the stored
    # reference exactly — OmniVoice subtitles every audible word.
    ref_text = ""
    if server.STT is not None:
        try:
            ref_text = (await server.STT.transcribe(buf, SAMPLE_RATE) or "").strip()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("voice sample: auto-transcribe failed: %s", exc)
    if not ref_text:
        return {"ok": False, "error": "no_speech"}
    wav = await asyncio.to_thread(audio_utils.f32le_bytes_to_wav_bytes, buf, SAMPLE_RATE)
    seconds = len(buf) / 4.0 / SAMPLE_RATE
    sample = store().add_sample(voice_id, wav, ref_text=ref_text,
                                name=name, seconds=seconds)
    if sample is None:
        return {"ok": False, "error": "unknown_voice"}
    if store().get_active() == voice_id:
        await apply_state()           # first sample of the active voice
    return {"ok": True, "voiceId": voice_id, "sample": sample, "refText": ref_text}


async def add_recorded_sample(buf: bytes, *, voice_id: str, name: str) -> dict:
    """Commit a recorded clone sample. Creates the voice when no id is given,
    so 'record a new voice' is one step for the user."""
    if source() != "local":
        return {"ok": False, "error": "unavailable"}
    if len(buf) < int(SAMPLE_RATE * 4 * 1.0):  # < ~1 s of f32 PCM
        return {"ok": False, "error": "too_short"}
    vid = voice_id
    created = None
    if not vid or vid == DEFAULT_VOICE_ID or store()._find(vid) is None:
        created = store().create_voice(name)
        vid = created["id"]
    res = await _pcm_to_sample(buf, voice_id=vid, name=name)
    if not res.get("ok") and created is not None:
        store().delete_voice(vid)     # don't leave an empty voice behind
    return res


# --------------------------------------------------------------------------- #
# Registering new cloned voices (recorded or uploaded)
# --------------------------------------------------------------------------- #
async def _ffmpeg_decode_f32le(data: bytes, sample_rate: int) -> bytes:
    """Decode arbitrary uploaded audio (mp3/m4a/ogg/webm/wav/…) to raw mono
    float32 LE PCM at ``sample_rate`` via ffmpeg — the format STT.transcribe
    expects. Raises on a non-zero ffmpeg exit (unsupported/corrupt input)."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate(data)
    if proc.returncode != 0:
        raise RuntimeError((err or b"").decode("utf-8", "replace")[:200] or "ffmpeg failed")
    return out


async def upload_voice_sample(request):
    """Register an UPLOADED audio file as a new cloned voice (mirrors the image
    /upload multipart pattern). ref_text: the client's manual transcript if given,
    otherwise a best-effort ffmpeg-decode + STT of the file. The sample bytes are
    forwarded to the wrapper, which decodes to the final reference and clones."""
    if source() is None:
        return web.json_response({"ok": False, "error": "voice cloning disabled"}, status=400)
    try:
        reader = await request.multipart()
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"multipart parse: {exc}"}, status=400)

    data = b""
    name = ""
    ref_text = ""
    content_type = ""
    filename = "upload"
    voice_id = ""
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            content_type = (part.headers.get("Content-Type") or "").lower().split(";")[0].strip()
            filename = part.filename or "upload"
            chunks: list = []
            total = 0
            while True:
                chunk = await part.read_chunk(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_VOICE_UPLOAD_BYTES:
                    return web.json_response(
                        {"ok": False, "error": f"file too large (>{MAX_VOICE_UPLOAD_BYTES} bytes)"},
                        status=413)
                chunks.append(chunk)
            data = b"".join(chunks)
        elif part.name == "name":
            name = (await part.text()).strip()
        elif part.name in ("refText", "ref_text"):
            ref_text = (await part.text()).strip()
        elif part.name in ("voiceId", "voice_id"):
            voice_id = (await part.text()).strip()

    if not data:
        return web.json_response({"ok": False, "error": "missing field 'file'"}, status=400)
    if content_type and not (content_type.startswith("audio/")
                             or content_type == "application/octet-stream"):
        return web.json_response(
            {"ok": False, "error": f"unsupported content type: {content_type}"}, status=400)

    # Decode for cleanup + auto-transcription. Best-effort: if ffmpeg can't
    # read the file, the original bytes are forwarded untouched (the wrapper
    # decodes on its own) and a manual transcript is required.
    pcm = b""
    try:
        pcm = await _ffmpeg_decode_f32le(data, SAMPLE_RATE)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("voice upload: decode failed (forwarding as-is): %s", exc)
    if pcm and server.CFG is not None and server.CFG.tts_clone_trim:
        # Same edge cleanup as recorded samples. Lenient here (uploads are
        # usually pre-cut): when nothing usable remains, forward the original
        # instead of rejecting.
        cleaned, tinfo = audio_utils.trim_clone_reference(pcm, SAMPLE_RATE)
        if cleaned is not None:
            if tinfo["dropped_head"] or tinfo["dropped_tail"]:
                LOG.info("voice upload: dropped cut-off edge speech (head=%s tail=%s)",
                         tinfo["dropped_head"], tinfo["dropped_tail"])
            pcm = cleaned
            data = await asyncio.to_thread(
                audio_utils.f32le_bytes_to_wav_bytes, pcm, SAMPLE_RATE)
            filename = "upload.wav"
            content_type = "audio/wav"
    if not ref_text and pcm and server.STT is not None:
        # Best-effort auto-transcript so the user need not type it for uploads.
        try:
            ref_text = (await server.STT.transcribe(pcm, SAMPLE_RATE) or "").strip()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("voice upload: auto-transcribe failed: %s", exc)
    if not ref_text:
        # Signal the client to ask the user for the transcript and retry.
        return web.json_response({"ok": False, "error": "need_transcript"}, status=400)

    # Local source: the decoded PCM becomes a sample of an existing or new voice.
    if source() == "local":
        if not pcm:
            return web.json_response(
                {"ok": False, "error": "decode_failed"}, status=400)
        vid = voice_id
        if not vid or vid == DEFAULT_VOICE_ID or store()._find(vid) is None:
            vid = store().create_voice(name)["id"]
        wav = await asyncio.to_thread(
            audio_utils.f32le_bytes_to_wav_bytes, pcm, SAMPLE_RATE)
        sample = store().add_sample(vid, wav, ref_text=ref_text, name=name,
                                    seconds=len(pcm) / 4.0 / SAMPLE_RATE)
        if sample is None:
            return web.json_response({"ok": False, "error": "store_failed"}, status=500)
        if store().get_active() == vid:
            await apply_state()
        try:
            await emit_state(None)
        except Exception:  # noqa: BLE001
            pass
        return web.json_response({"ok": True, "voiceId": vid, "sample": sample,
                                  "refText": ref_text})

    try:
        voice = await server.VOICES.register(
            data, filename=filename, content_type=content_type or "application/octet-stream",
            name=name, ref_text=ref_text)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("voice upload: register failed")
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    # Fan the fresh library state out to every connected browser.
    try:
        await emit_state(None)
    except Exception:  # noqa: BLE001
        pass
    return web.json_response({"ok": True, "voice": voice, "refText": ref_text})


async def clone_commit(buf: bytes, name: str) -> dict:
    """Validate, clean, transcribe and register a recorded clone sample.
    Returns the ``aivoice.sample.ack`` payload (without ``ts``). Strict about the
    edge cleanup: half words cut off by the recording window are dropped, and
    when nothing usable remains the user is asked to re-record (unlike uploads,
    where the original is forwarded as-is)."""
    if len(buf) < int(SAMPLE_RATE * 4 * 1.0):  # < ~1 s of f32 PCM
        return {"ok": False, "error": "too_short"}
    if server.CFG is not None and server.CFG.tts_clone_trim:
        cleaned, tinfo = audio_utils.trim_clone_reference(buf, SAMPLE_RATE)
        if cleaned is None:
            return {"ok": False, "error": tinfo["reason"] or "no_speech"}
        if tinfo["dropped_head"] or tinfo["dropped_tail"]:
            LOG.info("voice clone: dropped cut-off edge speech (head=%s tail=%s, kept %.1fs)",
                     tinfo["dropped_head"], tinfo["dropped_tail"], tinfo["kept_s"])
        buf = cleaned
    try:
        # ref_text is transcribed from the CLEANED buffer, so it matches the
        # reference audio exactly (the model subtitles every audible word).
        ref_text = (await server.STT.transcribe(buf, SAMPLE_RATE) or "").strip()
        if not ref_text:
            return {"ok": False, "error": "no_speech"}
        wav = await asyncio.to_thread(
            audio_utils.f32le_bytes_to_wav_bytes, buf, SAMPLE_RATE)
        voice = await server.VOICES.register(
            wav, filename="clone.wav", content_type="audio/wav",
            name=name, ref_text=ref_text)
        LOG.info("voice clone registered: id=%s name=%r ref=%r",
                 voice.get("id"), voice.get("name"), ref_text[:60])
        return {"ok": True, "voice": voice, "refText": ref_text}
    except Exception as exc:  # noqa: BLE001
        LOG.exception("voice clone failed")
        return {"ok": False, "error": str(exc)}
