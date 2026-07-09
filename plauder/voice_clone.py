"""Voice library / cloning — server-side handlers.

Registering new cloned voices from the browser: the ``/voice-upload`` HTTP
endpoint (uploaded files) and the commit step of the recorded-sample flow
(``voice.clone.*`` WS messages, buffered in ``ws_handler``), plus the small
helpers the hello frame and the TTS path need (active voice id, capability
block, preview sentence). The voice CRUD client itself lives in
``plauder.voices`` (``VoiceLibrary``).

Runtime state (CFG/STT/VOICES, set by ``configure``) lives in
``plauder.server``; this module reads it at call time via ``server.<name>``
to avoid an import cycle — same pattern as ``plauder.app``.
"""

from __future__ import annotations

import asyncio
import logging
import time

from aiohttp import web

from . import audio as audio_utils
from . import server
from .config import SAMPLE_RATE

LOG = logging.getLogger("voice-chat")

MAX_VOICE_UPLOAD_BYTES = 25 * 1024 * 1024

# One short, neutral sentence per UI language for the voice preview button.
_PREVIEW_SENTENCE = {
    "de": "Hallo, so klingt diese Stimme.",
    "en": "Hi, this is how this voice sounds.",
}


def clone_active() -> bool:
    """Voice library usable: TTS_CLONE_ENABLED and a VoiceLibrary is wired
    (TTS points at the OmniVoice wrapper). Drives the hello advertisement."""
    return bool(server.CFG and server.CFG.tts_clone_enabled and server.VOICES is not None)


def active_voice_id() -> str | None:
    """The globally selected cloned voice id to pass into the TTS call, or None
    when the voice library is disabled (backend uses its configured default)."""
    return server.VOICES.get_active() if clone_active() else None


async def voice_clone_hello() -> dict:
    """The ``voiceClone`` capability block for the hello frame: availability plus
    the current voice list + active id (so the client renders the library at
    once). Degrades gracefully — a slow/down wrapper never blocks or fails hello."""
    if not clone_active():
        return {"available": False}
    out = {"available": True, "active": server.VOICES.get_active(), "voices": []}
    try:
        out["voices"] = await asyncio.wait_for(server.VOICES.list(), timeout=5)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("voice library unreachable for hello: %s", exc)
    return out


async def emit_voice_state(origin_ws) -> None:
    """Push the fresh voice library state (list + active id) to EVERY connected
    browser after a mutation (select/rename/delete/new), so the shared session
    stays in sync — same idea as chat.remote / session.reset.remote."""
    payload = {"type": "voice.state", **(await voice_clone_hello()), "ts": time.time()}
    try:
        await origin_ws.send_json(payload)
    except Exception:
        pass
    await server._broadcast_peers(origin_ws, payload)


def preview_sentence() -> str:
    lang = (server.CFG.app_language if server.CFG else "en")
    return _PREVIEW_SENTENCE.get(lang, _PREVIEW_SENTENCE["en"])


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
    if not clone_active():
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

    try:
        voice = await server.VOICES.register(
            data, filename=filename, content_type=content_type or "application/octet-stream",
            name=name, ref_text=ref_text)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("voice upload: register failed")
        return web.json_response({"ok": False, "error": str(exc)}, status=502)

    # Fan the fresh library state out to every connected browser.
    try:
        payload = {"type": "voice.state", **(await voice_clone_hello()), "ts": time.time()}
        await server._broadcast_peers(None, payload)
    except Exception:  # noqa: BLE001
        pass
    return web.json_response({"ok": True, "voice": voice, "refText": ref_text})


async def clone_commit(buf: bytes, name: str) -> dict:
    """Validate, clean, transcribe and register a recorded clone sample.
    Returns the ``voice.clone.ack`` payload (without ``ts``). Strict about the
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
