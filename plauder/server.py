#!/usr/bin/env python3
"""Voice-Chat Server — HTTP/WebSocket-Layer.

Nur noch Transport + Turn-Orchestrierung. STT/TTS/LLM stecken hinter pluggable
Backends (siehe plauder.backends), gewählt per .env. Text-Verarbeitung
(Sanitizer, Halluzinations-Filter, Merging) und Turn-State sind eigene Module.

Pipeline:
  Browser (16 kHz float32-PCM via VAD/Push-to-Talk)
    └─ WebSocket → TurnState (Debounce + Coalescing)
                 → STT.transcribe → Halluzinations-Filter
                 → ConversationManager.chat (LLM + Verlauf)
                 → Sanitizer → TTS.synth → WAV → Browser
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import signal
import sys
import time
import uuid
from collections import deque
from pathlib import Path

from aiohttp import WSMsgType, web

from . import audio as audio_utils
from . import sanitizer
from . import wake
from .backends import LLMBackend, STTBackend, TTSBackend, UpstreamTimeoutError
from .config import SAMPLE_RATE, Config, load_config
from .session import ConversationManager
from .telegram_bridge import TelegramBridge
from .turn_state import TurnState, vad_params_for_debounce

LOG = logging.getLogger("voice-chat")

HERE = Path(__file__).resolve().parent.parent
STATIC_DIR = HERE / "static"
INDEX_HTML = STATIC_DIR / "index.html"
UPLOAD_DIR = HERE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_MIME = {
    "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", "image/bmp",
}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

STAGE = 6  # Protokoll-Version (Client-kompatibel)

# --------------------------------------------------------------------------- #
# Laufzeit-State (per configure() / main() gefüllt). Modul-Globals, damit die
# Turn-Handler ohne Request-Kontext darauf zugreifen können.
# --------------------------------------------------------------------------- #
CFG: Config | None = None
STT: STTBackend | None = None
TTS: TTSBackend | None = None
CONV: ConversationManager | None = None
BRIDGE: TelegramBridge | None = None
GHOST: sanitizer.HallucinationFilter | None = None

# House-Mode Speaker-ID (lazy, optional — speaker_id-Modul ist nicht Teil dieses
# Repos; bleibt deaktiviert, wenn es fehlt).
_SPEAKER_IDENTIFIER = None
_SPEAKER_INIT_FAILED = False


def configure(cfg: Config, *, stt=None, tts=None, conv=None, bridge=None, ghost=None):
    """Setzt den Laufzeit-State. Tests können hier Mock-Backends injizieren."""
    global CFG, STT, TTS, CONV, BRIDGE, GHOST
    CFG = cfg
    STT = stt
    TTS = tts
    CONV = conv
    BRIDGE = bridge
    GHOST = ghost if ghost is not None else sanitizer.HallucinationFilter.from_config(cfg)


def get_speaker_identifier():
    """Lazy-Load des optionalen House-Mode Speaker-Identifiers."""
    global _SPEAKER_IDENTIFIER, _SPEAKER_INIT_FAILED
    if CFG is None or not CFG.house_speaker_id or _SPEAKER_INIT_FAILED:
        return None
    if _SPEAKER_IDENTIFIER is None:
        try:
            import speaker_id as _spk  # optional, nicht im Repo
            data_dir = Path(CFG.house_data_dir)
            embedder = _spk.SpeakerEmbedder(str(data_dir / "models" / "campplus_multilingual.onnx"))
            store = _spk.SpeakerStore(str(data_dir / "speakers.json"))
            _SPEAKER_IDENTIFIER = _spk.SpeakerIdentifier(embedder, store)
            LOG.info("🔊 Speaker-ID aktiv (%d Sprecher)", len(store.all()))
        except Exception as exc:
            _SPEAKER_INIT_FAILED = True
            LOG.warning("Speaker-ID-Init fehlgeschlagen, deaktiviert: %s", exc)
            return None
    return _SPEAKER_IDENTIFIER


# --------------------------------------------------------------------------- #
# Identitäts-Helfer (für hello/healthz-Frames)
# --------------------------------------------------------------------------- #
def _agent_id() -> str:
    return CFG.openclaw_agent_id if CFG else "antonia"


def _default_user() -> str:
    return CFG.openclaw_user_id if CFG else "voice-user"


def _session_key_for_user(user_id: str) -> str:
    return f"agent:{_agent_id()}:openai-user:{user_id}"


# ============================================================================ #
# HTTP-Routen
# ============================================================================ #
async def index(_request):
    if not INDEX_HTML.exists():
        return web.Response(status=500, text=f"index.html fehlt: {INDEX_HTML}")
    return web.FileResponse(INDEX_HTML)


async def healthz(_request):
    return web.json_response({
        "ok": True,
        "stage": STAGE,
        "ts": time.time(),
        "backends": {
            "stt": CFG.stt_backend if CFG else None,
            "tts": CFG.tts_backend if CFG else None,
            "llm": CFG.llm_backend if CFG else None,
        },
        "stt": (STT.describe() if STT else {}),
        "agent": {
            "name": CFG.agent_name if CFG else None,
            "agent_id": _agent_id(),
            "user_id": _default_user(),
            "session_key": _session_key_for_user(_default_user()),
            "shared_with_telegram": False,
            "ready": CONV is not None and getattr(CONV.llm, "loaded", True),
            **(CONV.llm.describe() if CONV else {}),
        },
        "tts": (TTS.describe() if TTS else {}),
        "turn": {"debounce_ms_default": CFG.debounce_ms if CFG else None},
        "telegram_bridge": {
            "enabled": bool(BRIDGE and BRIDGE.enabled),
            "account_id": BRIDGE.account_id if BRIDGE else None,
            "target_chat_id": BRIDGE.target_chat_id if BRIDGE else None,
            "connected_browsers": len(BRIDGE._broadcast_channels) if BRIDGE else 0,
        },
    })


async def upload_image(request):
    """Nimmt ein Bild als multipart/form-data entgegen, gibt eine URL zurück."""
    try:
        reader = await request.multipart()
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"multipart parse: {exc}"}, status=400)

    field_part = await reader.next()
    if field_part is None or field_part.name != "file":
        return web.json_response({"ok": False, "error": "missing field 'file'"}, status=400)

    content_type = (field_part.headers.get("Content-Type") or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_IMAGE_MIME:
        return web.json_response(
            {"ok": False, "error": f"unsupported content type: {content_type or '<none>'}"},
            status=400)

    orig_name = field_part.filename or "upload"
    # Suffix NIE vom User-Filename übernehmen (XSS-Schutz: evil.html als image/jpeg).
    suffix = {
        "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
        "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp",
    }.get(content_type, ".bin")
    safe_id = uuid.uuid4().hex
    out_path = UPLOAD_DIR / f"{safe_id}{suffix}"

    total = 0
    with out_path.open("wb") as f:
        while True:
            chunk = await field_part.read_chunk(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                f.close()
                try:
                    out_path.unlink()
                except Exception:
                    pass
                return web.json_response(
                    {"ok": False, "error": f"file too large (>{MAX_UPLOAD_BYTES} bytes)"},
                    status=413)
            f.write(chunk)

    rel_url = f"/uploads/{out_path.name}"
    LOG.info("upload accepted: name=%r bytes=%d type=%s -> %s",
             orig_name, total, content_type, rel_url)
    return web.json_response({
        "ok": True, "url": rel_url, "name": orig_name,
        "bytes": total, "contentType": content_type,
    })


def _image_url_to_data_url(url: str) -> str | None:
    if not url:
        return None
    if url.startswith("data:") or url.startswith("http://") or url.startswith("https://"):
        return url
    if not url.startswith("/uploads/"):
        return None
    name = url[len("/uploads/"):]
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        return None
    path = UPLOAD_DIR / name
    if not path.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "application/octet-stream"
    try:
        b = path.read_bytes()
    except Exception:
        return None
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def _resolve_image_urls(image_urls: list, log_tag: str) -> list:
    if not image_urls:
        return []
    results = await asyncio.gather(*(
        asyncio.to_thread(_image_url_to_data_url, u) for u in image_urls
    ))
    out: list = []
    for u, durl in zip(image_urls, results):
        if durl:
            out.append(durl)
        else:
            LOG.warning("%s image %r konnte nicht aufgelöst werden", log_tag, u)
    return out


# ============================================================================ #
# Turn-Orchestrierung
# ============================================================================ #
def _combine_user_input(voice_merged: str, text_parts: list) -> str:
    parts: list = []
    if voice_merged and voice_merged.strip():
        parts.append(voice_merged.strip())
    for tp in text_parts:
        s = (tp or "").strip()
        if s:
            parts.append(s)
    return "\n\n".join(parts).strip()


def _queue_has_more_work(state: TurnState, exclude_task=None) -> bool:
    if state.agent_task and state.agent_task is not exclude_task and not state.agent_task.done():
        return True
    if state.debounce_task and not state.debounce_task.done():
        return True
    for tt in state.text_tasks:
        if tt is exclude_task:
            continue
        if not tt.done():
            return True
    return False


async def _agent_chat_with_retry(combined: str, *, image_urls, allow_retry: bool,
                                 user_key: str):
    """Ruft den ConversationManager; bei 408 einmal lautlos retryen.
    Gibt (reply, meta, retried) zurück oder wirft beim zweiten Fehler."""
    try:
        reply, meta = await CONV.chat(combined, user_key=user_key, image_urls=image_urls)
        return reply, meta, False
    except UpstreamTimeoutError as exc:
        if not allow_retry:
            raise
        LOG.warning("agent upstream timeout, retry once: %s", exc)
        await asyncio.sleep(0.5)
        reply, meta = await CONV.chat(combined, user_key=user_key, image_urls=image_urls)
        return reply, meta, True


async def _send_turn_pending(ws, state: TurnState):
    voice_merged = sanitizer.merge_transcripts(state.pending_texts)
    combined = _combine_user_input(voice_merged, state.pending_text_parts)
    await ws.send_json({
        "type": "turn.pending",
        "turnId": state.turn_id,
        "text": combined,
        "segmentIds": list(state.pending_segment_ids),
        "imageCount": len(state.pending_image_urls),
        "hasText": bool(state.pending_text_parts),
        "hasVoice": bool(state.pending_texts),
        "debounceMs": state.debounce_ms,
        "ts": time.time(),
    })


async def _run_turn(ws, state: TurnState):
    if not state.has_pending():
        return
    turn_id = state.turn_id
    voice_merged = sanitizer.merge_transcripts(state.pending_texts)
    segment_ids = list(state.pending_segment_ids)
    text_parts = list(state.pending_text_parts)
    image_urls = list(state.pending_image_urls)
    combined = _combine_user_input(voice_merged, text_parts)
    LOG.info("turn=%s commit: voice_segs=%d text_sends=%d imgs=%d combined=%r",
             turn_id, len(state.pending_texts), len(text_parts), len(image_urls), combined[:200])

    if BRIDGE:
        BRIDGE.begin_local_call()
    try:
        await _run_turn_inner(ws, state, turn_id, combined=combined,
                              voice_merged=voice_merged, text_parts=text_parts,
                              image_urls=image_urls, segment_ids=segment_ids)
    finally:
        if BRIDGE:
            BRIDGE.end_local_call()


async def _run_turn_inner(ws, state, turn_id, *, combined, voice_merged,
                          text_parts, image_urls, segment_ids):
    resolved_imgs = await _resolve_image_urls(image_urls, f"turn={turn_id}")

    if voice_merged and (text_parts or image_urls):
        source = "mixed"
    elif voice_merged:
        source = "voice"
    else:
        source = "text"

    await ws.send_json({
        "type": "turn.commit", "turnId": turn_id, "text": combined,
        "segmentIds": segment_ids, "source": source,
        "imageCount": len(resolved_imgs), "ts": time.time(),
    })

    # Telegram-Spiegelung des User-Inputs.
    if BRIDGE and BRIDGE.enabled:
        prefix = {"voice": "👤 (Voice)", "text": "👤 (Voice-Chat)",
                  "mixed": "👤 (Voice+Text)"}[source]
        mirror_parts = []
        if combined:
            mirror_parts.append(combined)
        for _ in range(len(resolved_imgs)):
            mirror_parts.append("📷 Bild gesendet")
        mirror_text = " · ".join(mirror_parts).strip()
        if mirror_text:
            if combined:
                BRIDGE.remember_self_sent(combined)
            asyncio.create_task(BRIDGE.send(f"{prefix}: {mirror_text}", echo_text=mirror_text))

    if CONV is None:
        return
    if not combined and not resolved_imgs:
        return
    reply_id = f"reply-{turn_id}"
    await ws.send_json({"type": "reply.start", "turnId": turn_id,
                        "replyId": reply_id, "ts": time.time()})

    # --- Streaming-Pfad (A1+A2): LLM-Token live → satzweises TTS → PCM-Chunks ---
    if CFG and CFG.streaming and hasattr(CONV, "chat_stream"):
        await _stream_reply_and_tts(
            ws, state, turn_id, reply_id,
            combined=combined, resolved_imgs=resolved_imgs)
        return

    # --- Klassischer Pfad (Fallback, STREAMING=0): erst komplett, dann 1 WAV ---
    t_llm = time.time()
    try:
        allow_retry = (CFG.llm_retry_timeout_on_idle
                       and not _queue_has_more_work(state, exclude_task=state.agent_task))
        reply_text, meta_agent, retried = await _agent_chat_with_retry(
            combined, image_urls=resolved_imgs if resolved_imgs else None,
            allow_retry=allow_retry, user_key=state.session_user or _default_user())
        if retried:
            LOG.info("turn=%s agent retry succeeded", turn_id)
    except asyncio.CancelledError:
        LOG.info("turn=%s agent-call cancelled", turn_id)
        raise
    except UpstreamTimeoutError as exc:
        LOG.warning("turn=%s agent upstream timeout (no retry): %s", turn_id, exc)
        await ws.send_json({"type": "reply.error", "turnId": turn_id, "replyId": reply_id,
                            "error": "upstream provider timeout",
                            "errorKind": "upstream_timeout", "ts": time.time()})
        return
    except Exception as exc:
        LOG.exception("turn=%s agent failed", turn_id)
        await ws.send_json({"type": "reply.error", "turnId": turn_id, "replyId": reply_id,
                            "error": str(exc), "ts": time.time()})
        return

    if sanitizer.is_no_reply(reply_text):
        llm_ms = int((time.time() - t_llm) * 1000)
        LOG.info("turn=%s agent NO_REPLY llm=%dms", turn_id, llm_ms)
        await ws.send_json({
            "type": "reply.silent", "turnId": turn_id, "replyId": reply_id,
            "reason": "no_reply", "llmMs": llm_ms,
            "usage": meta_agent.get("usage"),
            "finishReason": meta_agent.get("finish_reason"), "ts": time.time(),
        })
        return

    llm_ms = int((time.time() - t_llm) * 1000)
    cleaned = sanitizer.sanitize_for_tts(
        reply_text, pronunciations_file=CFG.pronunciations_file if CFG else None)
    LOG.info("turn=%s agent ok llm=%dms text=%r", turn_id, llm_ms, reply_text[:120])
    await ws.send_json({
        "type": "reply", "turnId": turn_id, "replyId": reply_id,
        "text": reply_text, "cleanedText": cleaned, "llmMs": llm_ms,
        "finishReason": meta_agent.get("finish_reason"),
        "usage": meta_agent.get("usage"), "ts": time.time(),
    })

    if BRIDGE and BRIDGE.enabled and cleaned:
        BRIDGE.remember_self_sent(reply_text)
        asyncio.create_task(BRIDGE.send(f"❤️ {CFG.agent_name}: {cleaned}", echo_text=reply_text))

    # --- TTS ---
    if TTS is None or not cleaned:
        return
    audio_id = f"audio-{turn_id}"
    state.audio_ids.add(audio_id)
    anchor = getattr(state, "speech_end_ts", 0.0) or 0.0
    now = time.time()
    start_evt = {"type": "audio.start", "turnId": turn_id,
                 "audioId": audio_id, "ts": now}
    if anchor:
        start_evt["e2eMs"] = int((now - anchor) * 1000)
        start_evt["debounceMs"] = state.debounce_ms  # Anteil Pause an e2e
    await ws.send_json(start_evt)
    t_tts = time.time()
    try:
        pcm_bytes, sr = await TTS.synth(cleaned, speed=state.speed)
    except asyncio.CancelledError:
        LOG.info("turn=%s tts cancelled", turn_id)
        state.audio_ids.discard(audio_id)
        raise
    except Exception as exc:
        LOG.exception("turn=%s tts failed", turn_id)
        state.audio_ids.discard(audio_id)
        await ws.send_json({"type": "audio.error", "turnId": turn_id, "audioId": audio_id,
                            "error": str(exc), "ts": time.time()})
        return

    tts_ms = int((time.time() - t_tts) * 1000)
    num_samples = len(pcm_bytes) // 2
    audio_s = num_samples / float(sr) if sr else 0.0
    wav_bytes = await asyncio.to_thread(audio_utils.pcm16_to_wav_bytes, pcm_bytes, sr)
    framed = audio_utils.wrap_wav_with_turn_id(wav_bytes, turn_id)
    LOG.info("turn=%s tts ok tts=%dms audio=%.2fs bytes=%d speed=%.2f",
             turn_id, tts_ms, audio_s, len(wav_bytes), state.speed)
    try:
        await ws.send_json({
            "type": "audio.meta", "turnId": turn_id, "audioId": audio_id,
            "sampleRate": sr, "durationS": round(audio_s, 3), "ttsMs": tts_ms,
            "bytes": len(wav_bytes), "speed": state.speed, "framed": True,
            "ts": time.time(),
        })
        await ws.send_bytes(framed)
    except asyncio.CancelledError:
        LOG.info("turn=%s tts send cancelled mid-flight", turn_id)
        raise


async def _tts_synth_stream(text: str, speed: float):
    """TTS-Streaming mit Fallback: nutzt ``synth_stream`` falls vorhanden, sonst
    das klassische ``synth`` (ein Häppchen). Hält den Orchestrator backend-agnostisch."""
    fn = getattr(TTS, "synth_stream", None)
    if fn is not None:
        async for item in fn(text, speed=speed):
            yield item
    else:
        pcm, sr = await TTS.synth(text, speed=speed)
        if pcm:
            yield pcm, sr


async def _stream_reply_and_tts(ws, state: TurnState, turn_id, reply_id, *,
                                combined, resolved_imgs):
    """Streaming-Turn (A1+A2).

    LLM-Token werden live gelesen; sobald ein Satz vollständig ist, geht er in
    die TTS-Queue. Ein paralleler TTS-Worker synthetisiert satzweise und schickt
    das Audio als PCM-Chunks (VCT2) progressiv an den Client — Satz 1 wird
    abgespielt, während Satz 2 noch generiert/synthetisiert wird.
    """
    pron = CFG.pronunciations_file if CFG else None
    max_chars = CFG.tts_max_chars_per_chunk if CFG else 220
    audio_id = f"audio-{turn_id}"
    sentence_q: asyncio.Queue = asyncio.Queue()

    parts: list[str] = []          # alle bisher empfangenen LLM-Deltas
    held: list[str] = []           # fertige Sätze, die noch auf TTS-Freigabe warten
    flushed_any = {"v": False}
    sent_len = {"v": 0}            # bereits an den Client gesendete Text-Länge
    started = {"audio": False}
    sr_box = {"sr": None}
    t_tts0 = {"t": None}
    t_first = {"t": None}          # Zeitpunkt des ersten LLM-Tokens
    # E2E-Anker ("User fertig mit Sprechen"); jetzt einfrieren, damit ein
    # nachrückendes Segment den Messpunkt dieses Turns nicht verschiebt.
    anchor = getattr(state, "speech_end_ts", 0.0) or 0.0

    async def _tts_worker() -> int:
        seq = 0
        while True:
            sentence = await sentence_q.get()
            if sentence is None:
                break
            if t_tts0["t"] is None:
                t_tts0["t"] = time.time()
            try:
                async for pcm, sr in _tts_synth_stream(sentence, state.speed):
                    if not pcm:
                        continue
                    if not started["audio"]:
                        started["audio"] = True
                        sr_box["sr"] = sr
                        now = time.time()
                        start_evt = {
                            "type": "audio.start", "turnId": turn_id, "audioId": audio_id,
                            "sampleRate": sr, "ts": now}
                        # Latenz-Aufschlüsselung bis zur ERSTEN Wiedergabe (≠ Gesamtzeit):
                        if anchor:
                            start_evt["e2eMs"] = int((now - anchor) * 1000)
                            start_evt["debounceMs"] = state.debounce_ms  # Anteil Pause an e2e
                        if t_first["t"] is not None:
                            start_evt["llmFirstMs"] = int((t_first["t"] - t_llm) * 1000)
                        if t_tts0["t"] is not None:
                            start_evt["ttsFirstMs"] = int((now - t_tts0["t"]) * 1000)
                        await ws.send_json(start_evt)
                    frame_bytes = max(2, int(sr * (CFG.tts_chunk_ms / 1000.0)) * 2)
                    for frame in audio_utils.iter_pcm_frames(pcm, frame_bytes):
                        seq += 1
                        await ws.send_bytes(audio_utils.wrap_pcm_chunk(turn_id, seq, frame))
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("turn=%s tts chunk failed", turn_id)
        return seq

    async def _emit_text():
        """Sendet neuen Antworttext an den Client — aber erst, wenn klar ist,
        dass es kein reines NO_REPLY ist (sonst würde 'NO_REPLY' kurz aufblitzen)."""
        full = "".join(parts)
        if not flushed_any["v"] and sanitizer.is_no_reply(full.strip()):
            return
        if len(full) > sent_len["v"]:
            await ws.send_json({
                "type": "reply.delta", "turnId": turn_id, "replyId": reply_id,
                "delta": full[sent_len["v"]:], "ts": time.time()})
            sent_len["v"] = len(full)

    async def _release(force: bool = False):
        full = "".join(parts).strip()
        if not flushed_any["v"] and not force and sanitizer.is_no_reply(full):
            return  # sieht (noch) wie NO_REPLY aus → Sätze zurückhalten
        while held:
            cleaned = sanitizer.sanitize_for_tts(held.pop(0), pronunciations_file=pron)
            if cleaned:
                await sentence_q.put(cleaned)
                flushed_any["v"] = True

    pending = ""

    async def _consume_stream():
        nonlocal pending
        agen = CONV.chat_stream(
            combined, user_key=state.session_user or _default_user(),
            image_urls=resolved_imgs if resolved_imgs else None)
        async for delta in agen:
            if t_first["t"] is None:
                t_first["t"] = time.time()
            parts.append(delta)
            await _emit_text()
            pending += delta
            sents, pending = audio_utils.split_stream_sentences(pending, max_chars)
            if sents:
                held.extend(sents)
                await _release()

    # t_llm VOR dem Worker-Start binden: der Worker liest es in seiner Closure
    # (für llmFirstMs). Würde später ein `await` zwischen create_task und dieser
    # Zuweisung stehen, könnte der Worker es sonst ungebunden lesen (NameError).
    t_llm = time.time()
    tts_task = asyncio.create_task(_tts_worker())
    state.audio_ids.add(audio_id)
    allow_retry = (CFG.llm_retry_timeout_on_idle
                   and not _queue_has_more_work(state, exclude_task=state.agent_task))

    async def _drain_tts():
        sentence_q.put_nowait(None)
        return await asyncio.gather(tts_task, return_exceptions=True)

    try:
        try:
            await _consume_stream()
        except UpstreamTimeoutError:
            if parts or not allow_retry:
                raise
            LOG.warning("turn=%s agent upstream timeout, retry once (stream)", turn_id)
            await asyncio.sleep(0.5)
            pending = ""
            await _consume_stream()
    except asyncio.CancelledError:
        LOG.info("turn=%s stream cancelled", turn_id)
        tts_task.cancel()
        await asyncio.gather(tts_task, return_exceptions=True)
        # audio_id NICHT verwerfen: _cancel_in_flight schickt daraufhin noch ein
        # audio.stop an den Client (Backup zum lokalen Barge-In-Stopp).
        raise
    except UpstreamTimeoutError as exc:
        await _drain_tts()
        state.audio_ids.discard(audio_id)
        LOG.warning("turn=%s agent upstream timeout (stream, no retry): %s", turn_id, exc)
        await ws.send_json({"type": "reply.error", "turnId": turn_id, "replyId": reply_id,
                            "error": "upstream provider timeout",
                            "errorKind": "upstream_timeout", "ts": time.time()})
        return
    except Exception as exc:
        await _drain_tts()
        state.audio_ids.discard(audio_id)
        LOG.exception("turn=%s agent stream failed", turn_id)
        await ws.send_json({"type": "reply.error", "turnId": turn_id, "replyId": reply_id,
                            "error": str(exc), "ts": time.time()})
        return

    # Restpuffer als letzten Satz übernehmen.
    if pending.strip():
        held.append(pending.strip())
    full_reply = "".join(parts).strip()
    llm_ms = int((time.time() - t_llm) * 1000)
    meta = dict(getattr(CONV, "last_stream_meta", {}) or {})

    if sanitizer.is_no_reply(full_reply):
        await _drain_tts()
        state.audio_ids.discard(audio_id)
        LOG.info("turn=%s agent NO_REPLY llm=%dms (stream)", turn_id, llm_ms)
        await ws.send_json({
            "type": "reply.silent", "turnId": turn_id, "replyId": reply_id,
            "reason": "no_reply", "llmMs": llm_ms, "usage": meta.get("usage"),
            "finishReason": meta.get("finish_reason"), "ts": time.time()})
        return

    await _release(force=True)            # zurückgehaltene Sätze freigeben
    await _emit_text()                    # restlichen Text an den Client
    cleaned_full = sanitizer.sanitize_for_tts(full_reply, pronunciations_file=pron)
    LOG.info("turn=%s agent ok llm=%dms text=%r (stream)", turn_id, llm_ms, full_reply[:120])
    await ws.send_json({
        "type": "reply", "turnId": turn_id, "replyId": reply_id,
        "text": full_reply, "cleanedText": cleaned_full, "llmMs": llm_ms,
        "finishReason": meta.get("finish_reason"), "usage": meta.get("usage"),
        "streamed": True, "ts": time.time()})

    if BRIDGE and BRIDGE.enabled and cleaned_full:
        BRIDGE.remember_self_sent(full_reply)
        asyncio.create_task(BRIDGE.send(f"❤️ {CFG.agent_name}: {cleaned_full}", echo_text=full_reply))

    results = await _drain_tts()
    state.audio_ids.discard(audio_id)
    total_chunks = next((r for r in results if isinstance(r, int)), 0)
    if started["audio"]:
        tts_ms = int((time.time() - t_tts0["t"]) * 1000) if t_tts0["t"] else 0
        LOG.info("turn=%s tts ok chunks=%d sr=%s tts=%dms speed=%.2f (stream)",
                 turn_id, total_chunks, sr_box["sr"], tts_ms, state.speed)
        await ws.send_json({
            "type": "audio.end", "turnId": turn_id, "audioId": audio_id,
            "chunks": total_chunks, "sampleRate": sr_box["sr"], "ttsMs": tts_ms,
            "speed": state.speed, "ts": time.time()})


async def _debounce_then_run(ws, state: TurnState):
    try:
        await asyncio.sleep(state.debounce_ms / 1000.0)
    except asyncio.CancelledError:
        return
    turn_id_at_start = state.turn_id
    try:
        state.agent_task = asyncio.create_task(_run_turn(ws, state))
        await state.agent_task
    except asyncio.CancelledError:
        LOG.debug("turn=%s debounce-task cancelled while running", turn_id_at_start)
    finally:
        state.agent_task = None
        if state.turn_id == turn_id_at_start:
            state.reset()


async def _cancel_in_flight(state: TurnState, ws):
    # Ein Abbruch beendet die laufende Antwort → ein evtl. gesetztes
    # "Fenster-nicht-wieder-öffnen"-Flag ist damit gegenstandslos.
    state.wake_suppress_reopen = False
    cancelled = False
    tasks_to_await: list = []
    if state.debounce_task and not state.debounce_task.done():
        state.debounce_task.cancel()
        tasks_to_await.append(state.debounce_task)
        cancelled = True
    if state.agent_task and not state.agent_task.done():
        state.agent_task.cancel()
        tasks_to_await.append(state.agent_task)
        cancelled = True
    for tt in list(state.text_tasks):
        if not tt.done():
            tt.cancel()
            tasks_to_await.append(tt)
            cancelled = True
    if cancelled:
        await ws.send_json({"type": "turn.discarded", "turnId": state.turn_id,
                            "reason": "new-speech-detected", "ts": time.time()})
    if tasks_to_await:
        await asyncio.gather(*tasks_to_await, return_exceptions=True)
    if state.audio_ids:
        for aid in list(state.audio_ids):
            await ws.send_json({"type": "audio.stop", "audioId": aid, "ts": time.time()})
        state.audio_ids.clear()


def _speaker_tag(text: str, speaker_info: dict | None) -> str:
    if speaker_info is None:
        return text
    nm = speaker_info["name"]
    role = speaker_info["role"]
    rel = speaker_info.get("relation") or ""
    if role == "admin":
        q = f"admin, {rel}" if rel else "admin"
        return f"[Sprecher: {nm} ({q})] {text}"
    if speaker_info["known"]:
        return f"[Sprecher: {nm} ({rel})] {text}" if rel else f"[Sprecher: {nm}] {text}"
    return f"[Sprecher: {nm}, unbekannt] {text}"


# Nach manuellem Schließen des Gesprächsfensters für so viele Sekunden KEIN
# automatisches Wieder-Öffnen (schluckt nachlaufende Partials / Echo / Störer).
WAKE_CLOSE_GUARD_S = 2.0


def _wake_mode() -> str:
    """Normalisierter Wake-Modus: 'alexa' (One-Shot) oder 'conversation'."""
    m = (getattr(CFG, "wake_mode", "conversation") or "conversation").lower() if CFG else "conversation"
    return "alexa" if m in ("alexa", "oneshot", "one-shot", "single") else "conversation"


def _wake_oneshot() -> bool:
    """Alexa-Modus: nach jeder Antwort schließt das Konversationsfenster sofort
    (kein Folgefragen-Fenster — jeder Befehl braucht wieder das Weckwort)."""
    return _wake_mode() == "alexa"


def _wake_closed_active(state: TurnState, now: float | None = None) -> bool:
    """True, solange ein manuell geschlossenes Fenster bewusst zu bleiben soll:
    während der noch laufenden Antwort (``wake_suppress_reopen``) ODER im kurzen
    Nachlauf-Guard (``wake_closed_until``). In dieser Zeit ignoriert die Pipeline
    die Wake-Erkennung KOMPLETT — sonst reißen Echo der eigenen TTS, B2-Partials
    oder Folgegerede (inkl. Fuzzy-Fehltreffer) das Fenster sofort wieder auf."""
    now = time.time() if now is None else now
    return bool(state.wake_suppress_reopen) or now < state.wake_closed_until


async def _open_wake_window(ws, state: TurnState, now: float | None = None,
                            reason: str = "command"):
    """Konversationsfenster (neu) öffnen/auffrischen und den Client informieren.

    `reason` sagt dem Client, ob Antonia jetzt IDLE ist (`armed` = wartet auf den
    Befehl, `done` = Antwort fertig gesprochen) → Idle-Timer für das akustische
    Auslauf-Feedback starten — oder ob gerade ein Befehl reinkam (`command`) und
    eine Antwort folgt → Timer NICHT laufen lassen. So schließt das Fenster erst,
    nachdem die Antwort komplett ausgesprochen wurde, nicht währenddessen."""
    now = time.time() if now is None else now
    # Guard nach manuellem Schließen: nicht automatisch wieder aufmachen.
    if _wake_closed_active(state, now):
        return
    state.wake_until = now + CFG.wake_word_window_s
    await ws.send_json({
        "type": "wake.window", "turnId": state.turn_id, "reason": reason,
        "windowS": CFG.wake_word_window_s, "ts": now})


async def _emit_wake_detected(ws, state: TurnState, segment_id):
    """Akustisches Früh-Feedback: einmal pro Segment ein `wake.detected` senden,
    sobald das Weckwort erkannt wurde (Partial ODER finales Segment)."""
    if state.wake_detected_seg == segment_id:
        return
    # Guard nach manuellem Schließen: kein Früh-Feedback / Öffnen durch Nachläufer.
    if _wake_closed_active(state):
        return
    state.wake_detected_seg = segment_id
    await ws.send_json({
        "type": "wake.detected", "segmentId": segment_id,
        "turnId": state.turn_id, "ts": time.time()})


async def _handle_audio_segment(ws, state: TurnState, pcm_bytes, segment_id, peer,
                                speech_start_ts=None, barge_in=False):
    num_samples = len(pcm_bytes) // 4
    duration_s = num_samples / SAMPLE_RATE
    force_hold = bool(state.has_pending() or barge_in)
    LOG.info("segment from %s: id=%s samples=%d dur=%.2fs (turn=%s)",
             peer, segment_id, num_samples, duration_s, state.turn_id)

    t0 = time.time()
    await ws.send_json({
        "type": "segment.received", "segmentId": segment_id, "turnId": state.turn_id,
        "samples": num_samples, "durationS": round(duration_s, 3), "ts": t0,
    })

    # Bei aktivem Wake-Word wird ein laufender Turn NICHT sofort abgebrochen —
    # erst wenn das Segment das Gate passiert (sonst würde Fremdgerede Antonia
    # mitten im Satz unterbrechen, obwohl es gar nicht an sie gerichtet ist).
    wake_gating = bool(CFG and getattr(state, "wake_word_enabled", False))
    if not wake_gating:
        await _cancel_in_flight(state, ws)

    # STT
    try:
        t_stt = time.time()
        text = await STT.transcribe(pcm_bytes, SAMPLE_RATE)
        no_speech_prob = getattr(STT, "last_no_speech_prob", None)
        stt_ms = int((time.time() - t_stt) * 1000)
    except Exception as exc:
        LOG.exception("STT failed for segment %s", segment_id)
        await ws.send_json({"type": "transcript.error", "segmentId": segment_id,
                            "turnId": state.turn_id, "error": str(exc), "ts": time.time()})
        return
    LOG.info("stt seg=%s text=%r (%dms)", segment_id, text, stt_ms)

    # Halluzinations-Filter
    if text and GHOST and GHOST.is_hallucination(
            text, no_speech_prob=no_speech_prob, duration_s=duration_s):
        LOG.info("ghost gefiltert seg=%s text=%r", segment_id, text)
        await ws.send_json({
            "type": "transcript", "segmentId": segment_id, "turnId": state.turn_id,
            "text": "", "filtered": "hallucination", "filteredText": text,
            "sttMs": stt_ms, "totalMs": int((time.time() - t0) * 1000), "ts": time.time(),
        })
        if state.has_pending():
            state.debounce_task = asyncio.create_task(_debounce_then_run(ws, state))
        return

    # --- Wake-Word-Gate (Prefix). Nur Voice; Tipp-Eingaben sind immer gemeint.
    if wake_gating and text:
        now = time.time()
        # Manuell geschlossen + läuft noch / Guard → Segment komplett ignorieren.
        # KEIN Wake-Match (auch keine Fuzzy-Treffer), KEIN Cancel der laufenden
        # Antwort. So kann während der Verarbeitung nichts das Fenster aufreißen.
        if _wake_closed_active(state, now):
            LOG.info("wake: geschlossen → ignoriert seg=%s text=%r", segment_id, text)
            await ws.send_json({
                "type": "transcript.ignored", "segmentId": segment_id,
                "turnId": state.turn_id, "text": text,
                "reason": "wake_closed", "ts": time.time()})
            if state.has_pending():
                state.debounce_task = asyncio.create_task(_debounce_then_run(ws, state))
            return
        # „Fenster offen?" am SPRECH-BEGINN messen, nicht am Commit-Zeitpunkt:
        # Wer innerhalb des Fensters zu sprechen beginnt, dessen (evtl. langer)
        # Satz darf auch dann noch durch, wenn das Fenster während des Sprechens
        # bzw. der Transkription abläuft (sonst Race: Eingabe erkannt, aber
        # verworfen). speech_start_ts ist bereits clock-skew-korrigiert (Server-
        # zeit); Fallback auf now, falls kein Zeitstempel vorliegt.
        ref_ts = speech_start_ts if speech_start_ts is not None else now
        if ref_ts < state.wake_until:
            # Konversationsfenster war beim Sprech-Beginn offen → Folgefrage durch.
            command_text = text
        else:
            matched, remainder = wake.match_wake(
                text, CFG.wake_word, fuzzy=CFG.wake_word_fuzzy,
                anywhere=CFG.wake_word_anywhere, ratio=CFG.wake_word_ratio)
            if not matched:
                LOG.info("wake: ignoriert seg=%s text=%r", segment_id, text)
                await ws.send_json({
                    "type": "transcript.ignored", "segmentId": segment_id,
                    "turnId": state.turn_id, "text": text,
                    "reason": "no_wake_word", "ts": time.time()})
                if state.has_pending():
                    state.debounce_task = asyncio.create_task(_debounce_then_run(ws, state))
                return
            # Wake erkannt → ggf. Früh-Pling nachholen (falls kein Partial es
            # schon gesendet hat), Wake-Word abschneiden.
            await _emit_wake_detected(ws, state, segment_id)
            command_text = remainder if CFG.wake_word_strip else text
            if not (command_text or "").strip():
                # Nur das Wake-Word gesagt → „scharf", Antonia wartet (idle) auf
                # den Befehl → Idle-Timer beim Client läuft (reason=armed).
                await _open_wake_window(ws, state, now, reason="armed")
                await ws.send_json({
                    "type": "wake.armed", "turnId": state.turn_id,
                    "windowS": CFG.wake_word_window_s, "ts": time.time()})
                return

        # Stop-Kommando („stop", „ok stopp", …) → laufende Antwort stoppen und
        # Konversationsfenster schließen (Wake-Word wird wieder gebraucht).
        if wake.is_stop_command(command_text):
            await _cancel_in_flight(state, ws)
            state.wake_until = 0.0
            state.wake_detected_seg = None
            LOG.info("wake: stop seg=%s → Fenster geschlossen", segment_id)
            await ws.send_json({
                "type": "wake.closed", "turnId": state.turn_id,
                "reason": "stop_command", "ts": time.time()})
            return

        # Befehl an die KI → Fenster auffrischen; eine Antwort folgt, also läuft
        # der Idle-Timer NICHT (reason=command). Dann laufenden Turn abräumen.
        await _open_wake_window(ws, state, now, reason="command")
        text = command_text
        await _cancel_in_flight(state, ws)

    # Speaker-ID (House-Mode, optional, failsafe)
    speaker_info = None
    if text:
        identifier = get_speaker_identifier()
        if identifier is not None:
            try:
                audio = audio_utils.pcm_bytes_to_float32_array(pcm_bytes)
                res = await asyncio.to_thread(
                    identifier.identify, audio, SAMPLE_RATE, None, speech_start_ts, force_hold)
                speaker_info = {
                    "name": res.name, "role": res.role,
                    "relation": getattr(res, "relation", ""),
                    "score": round(res.score, 4), "known": res.known, "held": res.held,
                }
            except Exception:
                LOG.exception("speaker-id failed for segment %s (ignoriert)", segment_id)

    transcript_evt = {
        "type": "transcript", "segmentId": segment_id, "turnId": state.turn_id,
        "text": text, "sttMs": stt_ms, "totalMs": int((time.time() - t0) * 1000),
        "ts": time.time(),
    }
    if speaker_info is not None:
        transcript_evt["speaker"] = speaker_info
    await ws.send_json(transcript_evt)

    if not text:
        if state.has_pending():
            state.debounce_task = asyncio.create_task(_debounce_then_run(ws, state))
        return

    state.pending_texts.append(_speaker_tag(text, speaker_info))
    state.pending_segment_ids.append(segment_id)
    # E2E-Anker: Empfangszeit dieses (letzten) Segments. Bei Coalescing gewinnt
    # das jeweils letzte Segment → Messung ab "User hat zuletzt gesprochen".
    state.speech_end_ts = t0
    await _send_turn_pending(ws, state)
    state.debounce_task = asyncio.create_task(_debounce_then_run(ws, state))


# ============================================================================ #
# B2: Streaming-STT (Zwischen-Transkripte während ein Segment eingestreamt wird)
# ============================================================================ #
async def _do_partial(ws, state: TurnState, seg: dict, pcm: bytes):
    """Transkribiert den bisher angesammelten Puffer und schickt ein
    transcript.partial — nur fürs UI, ändert keinen Turn-State."""
    try:
        text = await STT.transcribe(pcm, SAMPLE_RATE)
    except Exception:
        LOG.exception("partial STT failed seg=%s", seg.get("id"))
        text = ""
    finally:
        seg["partial_running"] = False
    # Segment inzwischen committed/abgebrochen? Dann Partial verwerfen.
    if seg.get("done") or not text:
        return
    seg["partial_text"] = text
    # Früh-Pling: Weckwort schon im wachsenden Partial erkennen (nur Wake-Modus
    # und nur solange das Fenster zu ist — bei offenem Fenster ist kein Weckwort
    # nötig). So weiß der Nutzer SOFORT, dass es getriggert hat, statt erst nach
    # dem Aussprechen.
    if (CFG and state.wake_word_enabled and time.time() >= state.wake_until
            and state.wake_detected_seg != seg.get("id")
            and not _wake_closed_active(state)):
        matched, _ = wake.match_wake(
            text, CFG.wake_word, fuzzy=CFG.wake_word_fuzzy,
            anywhere=CFG.wake_word_anywhere, ratio=CFG.wake_word_ratio)
        if matched:
            await _emit_wake_detected(ws, state, seg.get("id"))
    await ws.send_json({
        "type": "transcript.partial", "segmentId": seg.get("id"),
        "turnId": state.turn_id, "text": text, "ts": time.time(),
    })


def _maybe_spawn_partial(ws, state: TurnState, seg: dict):
    """Throttle-Check + Start einer Partial-Transkription (B2). Sync — startet
    bei Bedarf einen Task, blockiert die Frame-Schleife nicht."""
    if not (CFG and CFG.stt_partial and STT is not None):
        return
    if seg.get("partial_running") or seg.get("done"):
        return
    buf_len = len(seg["buf"])
    now = time.time()
    # f32 = 4 Bytes/Sample.
    new_bytes = buf_len - seg.get("last_partial_len", 0)
    min_new = int(SAMPLE_RATE * 4 * (CFG.stt_partial_min_new_ms / 1000.0))
    if new_bytes < min_new:
        return
    if (now - seg.get("last_partial_ts", 0)) * 1000.0 < CFG.stt_partial_min_interval_ms:
        return
    seg["partial_running"] = True
    seg["last_partial_ts"] = now
    seg["last_partial_len"] = buf_len
    asyncio.create_task(_do_partial(ws, state, seg, bytes(seg["buf"])))


# ============================================================================ #
# WebSocket-Handler
# ============================================================================ #
async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)
    peer = request.remote
    LOG.info("ws connect: %s", peer)

    state = TurnState()
    state.session_user = _default_user()
    state.speed = CFG.tts_speed if CFG else 1.0
    state.debounce_ms = CFG.debounce_ms if CFG else 1200
    # Start-Default des Wake-Modus; der Client schaltet ihn via 'settings' um.
    state.wake_word_enabled = bool(CFG.wake_word_enabled) if CFG else False
    pending_segment_meta: deque = deque()
    # B1: gestreamtes Eingangs-Segment (Frames laufen ein, während gesprochen
    # wird). None = kein aktives Stream-Segment → Binärframes sind Voll-Segmente.
    active_stream: dict | None = None

    if BRIDGE is not None:
        BRIDGE.register_broadcast(ws)

    await ws.send_json({
        "type": "hello", "stage": STAGE,
        "msg": f"Server bereit – Agent: {CFG.agent_name}.",
        "agent_name": CFG.agent_name,
        "stt": (STT.describe() if STT else {}),
        "agent": {
            "name": CFG.agent_name, "agent_id": _agent_id(),
            "user_id": state.session_user,
            "session_key": _session_key_for_user(state.session_user),
            "shared_with_telegram": False,
        },
        "telegram_bridge": {
            "enabled": bool(BRIDGE and BRIDGE.enabled),
            "account_id": BRIDGE.account_id if BRIDGE else None,
            "target_chat_id": BRIDGE.target_chat_id if BRIDGE else None,
        },
        "tts": {"sample_rate": TTS.sample_rate if TTS else None, "speed": state.speed},
        "streaming": bool(CFG.streaming) if CFG else False,
        "streamInput": bool(CFG.streaming) if CFG else False,
        "sttPartial": bool(CFG.stt_partial) if CFG else False,
        "wakeWord": {
            "available": True,             # Wake-Modus ist immer wählbar (UI)
            "enabled": bool(CFG.wake_word_enabled) if CFG else False,  # Start-Default
            "word": CFG.wake_word if CFG else "",
            "windowS": CFG.wake_word_window_s if CFG else 0,
            "mode": _wake_mode(),          # "conversation" | "alexa" (One-Shot)
        },
        "turn": {
            "debounce_ms": state.debounce_ms,
            "debounce_ms_min": CFG.debounce_ms_min, "debounce_ms_max": CFG.debounce_ms_max,
            "vad": vad_params_for_debounce(state.debounce_ms),
        },
        "ts": time.time(),
    })

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    LOG.warning("ws non-json text from %s: %r", peer, msg.data)
                    continue
                t = data.get("type")
                if t == "ping":
                    await ws.send_json({"type": "ack", "received": data, "ts": time.time()})
                elif t == "playback.done":
                    # Wake-Word: Antwort fertig ausgesprochen → Konversations-
                    # fenster (neu) öffnen; jetzt erst läuft der Idle-Timer.
                    # Ausnahme: Der User hat den Kanal währenddessen selbst
                    # geschlossen → nicht wieder aufmachen (Flag einmalig zurück).
                    if CFG and state.wake_word_enabled:
                        if state.wake_suppress_reopen:
                            # Manuell geschlossen während der Antwort: nicht wieder
                            # öffnen. Kurzer Nachlauf-Guard, damit das Echo direkt
                            # nach der Antwort das Fenster nicht doch aufreißt.
                            state.wake_suppress_reopen = False
                            state.wake_closed_until = max(
                                state.wake_closed_until, time.time() + WAKE_CLOSE_GUARD_S)
                        elif _wake_oneshot():
                            # Alexa-Modus: nach der Antwort Fenster sofort schließen,
                            # neuer Befehl braucht wieder das Weckwort.
                            state.wake_until = 0.0
                            state.wake_detected_seg = None
                            await ws.send_json({
                                "type": "wake.closed", "turnId": state.turn_id,
                                "reason": "oneshot", "ts": time.time()})
                        else:
                            await _open_wake_window(ws, state, reason="done")
                    ident = get_speaker_identifier()
                    if ident is not None:
                        try:
                            ident.mark_playback_finished()
                        except Exception:
                            LOG.exception("mark_playback_finished failed (ignoriert)")
                elif t == "barge_in":
                    reason = (data.get("reason") or "speech")
                    # Wake-Modus + manuell geschlossen/Guard: die Sprache ist NICHT
                    # an Antonia gerichtet → laufende Antwort NICHT abbrechen.
                    if CFG and state.wake_word_enabled and _wake_closed_active(state):
                        LOG.info("barge-in ignoriert (wake zu) von %s (reason=%s)", peer, reason)
                    else:
                        in_flight = (state.agent_task and not state.agent_task.done()) or bool(state.audio_ids)
                        if in_flight:
                            LOG.info("barge-in from %s (reason=%s)", peer, reason)
                            await _cancel_in_flight(state, ws)
                        # Wer eine LAUFENDE Antwort unterbricht (serverseitig in-flight
                        # ODER clientseitig noch am Abspielen), führt das Gespräch fort
                        # → Fenster offen halten, damit die folgende (unterbrechende)
                        # Eingabe nicht als „kein Weckwort" verworfen wird. NICHT bei
                        # bloßem Geräusch ohne laufende Antwort (sonst nie wieder Gate).
                        if CFG and state.wake_word_enabled and (in_flight or bool(data.get("playing"))):
                            await _open_wake_window(ws, state, reason="command")
                elif t == "wake.close":
                    # Voice-Kanal (Konversationsfenster) manuell schließen, OHNE
                    # laufende Verarbeitung abzubrechen (das macht 'barge_in').
                    # Danach ist wieder das Weckwort nötig.
                    if CFG and state.wake_word_enabled:
                        in_flight = (state.agent_task and not state.agent_task.done()) or bool(state.audio_ids)
                        state.wake_until = 0.0
                        state.wake_detected_seg = None
                        # Kurzer Guard: nachlaufende Partials/Echo dürfen das Fenster
                        # jetzt nicht sofort wieder aufmachen.
                        state.wake_closed_until = time.time() + WAKE_CLOSE_GUARD_S
                        # Läuft noch eine Antwort (server-seitig in_flight ODER
                        # der Client spielt noch Nachklang), folgt danach ein
                        # playback.done, das das Fenster sonst wieder öffnet →
                        # unterdrücken.
                        state.wake_suppress_reopen = bool(in_flight or data.get("playing"))
                        LOG.info("wake: manuell geschlossen (%s, in_flight=%s)", peer, in_flight)
                        await ws.send_json({
                            "type": "wake.closed", "turnId": state.turn_id,
                            "reason": "manual", "ts": time.time()})
                elif t == "segment.start":
                    _sst = data.get("speechStartTs")
                    _cnow = data.get("clientNow")
                    speech_start_ts = None
                    if isinstance(_sst, (int, float)) and isinstance(_cnow, (int, float)):
                        speech_start_ts = time.time() - (float(_cnow) - float(_sst)) / 1000.0
                    pending_segment_meta.append({
                        "id": data.get("segmentId") or uuid.uuid4().hex[:8],
                        "speech_start_ts": speech_start_ts,
                        "barge_in": bool(data.get("bargeIn")),
                    })
                elif t == "segment.stream.start":
                    # B1: Beginn eines gestreamten Segments. Folgende Binärframes
                    # (roh-f32le) werden angehängt, bis segment.stream.commit kommt.
                    _sst = data.get("speechStartTs")
                    _cnow = data.get("clientNow")
                    speech_start_ts = None
                    if isinstance(_sst, (int, float)) and isinstance(_cnow, (int, float)):
                        speech_start_ts = time.time() - (float(_cnow) - float(_sst)) / 1000.0
                    active_stream = {
                        "id": data.get("segmentId") or uuid.uuid4().hex[:8],
                        "buf": bytearray(),
                        "speech_start_ts": speech_start_ts,
                        "barge_in": bool(data.get("bargeIn")),
                        # B2: Partial-Throttle-State.
                        "partial_running": False, "last_partial_ts": 0.0,
                        "last_partial_len": 0, "partial_text": "", "done": False,
                    }
                elif t == "segment.stream.commit":
                    if active_stream is not None:
                        _seg = active_stream
                        _seg["done"] = True       # späte Partials unterdrücken
                        active_stream = None
                        pcm = bytes(_seg["buf"])
                        if pcm:
                            asyncio.create_task(_handle_audio_segment(
                                ws, state, pcm, _seg["id"], peer,
                                speech_start_ts=_seg.get("speech_start_ts"),
                                barge_in=_seg.get("barge_in", False)))
                elif t == "segment.stream.abort":
                    if active_stream is not None:
                        active_stream["done"] = True
                    active_stream = None
                elif t == "text.message":
                    text_val = (data.get("text") or "").strip()
                    img_urls = data.get("imageUrls") or []
                    if not isinstance(img_urls, list):
                        img_urls = []
                    img_urls = [str(u) for u in img_urls if u]
                    if not text_val and not img_urls:
                        continue
                    if state.agent_task and not state.agent_task.done():
                        await _cancel_in_flight(state, ws)
                    if text_val:
                        state.pending_text_parts.append(text_val)
                    if img_urls:
                        state.pending_image_urls.extend(img_urls)
                    await _send_turn_pending(ws, state)
                    if state.debounce_task and not state.debounce_task.done():
                        state.debounce_task.cancel()
                    state.debounce_task = asyncio.create_task(_debounce_then_run(ws, state))
                elif t == "session.reset":
                    await _cancel_in_flight(state, ws)
                    state.pending_texts.clear()
                    state.pending_segment_ids.clear()
                    pending_segment_meta.clear()
                    if active_stream is not None:
                        active_stream["done"] = True
                    active_stream = None
                    state.turn_id = uuid.uuid4().hex[:8]
                    _ident = get_speaker_identifier()
                    if _ident is not None:
                        try:
                            _ident.reset()
                        except Exception:
                            LOG.exception("speaker identifier reset failed (ignoriert)")
                    new_user = f"{_default_user()}-{uuid.uuid4().hex[:8]}"
                    state.session_user = new_user
                    if CONV is not None:
                        CONV.reset(new_user)
                    LOG.info("session reset for %s: user=%s", peer, new_user)
                    await ws.send_json({
                        "type": "session.reset.ack", "sessionUser": new_user,
                        "sessionKey": _session_key_for_user(new_user),
                        "sharedWithTelegram": False, "ts": time.time(),
                    })
                elif t == "settings":
                    if "speed" in data:
                        try:
                            state.speed = max(0.5, min(3.0, float(data["speed"])))
                        except (TypeError, ValueError):
                            pass
                    if "debounceMs" in data:
                        try:
                            state.debounce_ms = max(CFG.debounce_ms_min,
                                                    min(CFG.debounce_ms_max, int(data["debounceMs"])))
                        except (TypeError, ValueError):
                            pass
                    if "wakeWordEnabled" in data:
                        was_on = state.wake_word_enabled
                        state.wake_word_enabled = bool(data["wakeWordEnabled"])
                        # Wake-Modus verlassen → offenes Konversationsfenster schließen.
                        if was_on and not state.wake_word_enabled:
                            state.wake_until = 0.0
                    vad_params = vad_params_for_debounce(state.debounce_ms)
                    LOG.info("settings updated: speed=%.2f debounce=%dms wake=%s",
                             state.speed, state.debounce_ms, state.wake_word_enabled)
                    await ws.send_json({
                        "type": "settings.ack", "speed": state.speed,
                        "debounceMs": state.debounce_ms,
                        "wakeWordEnabled": state.wake_word_enabled,
                        "vad": vad_params, "ts": time.time(),
                    })
                else:
                    LOG.debug("ws text: %r", data)
            elif msg.type == WSMsgType.BINARY:
                if active_stream is not None:
                    # B1: Frame eines gestreamten Segments — anhängen. Die finale
                    # Transkription passiert beim commit; B2 schiebt zwischendurch
                    # Partials (gedrosselt) ein.
                    active_stream["buf"].extend(msg.data)
                    _maybe_spawn_partial(ws, state, active_stream)
                    continue
                if pending_segment_meta:
                    _meta = pending_segment_meta.popleft()
                else:
                    _meta = {"id": uuid.uuid4().hex[:8], "speech_start_ts": None, "barge_in": False}
                asyncio.create_task(_handle_audio_segment(
                    ws, state, msg.data, _meta["id"], peer,
                    speech_start_ts=_meta.get("speech_start_ts"),
                    barge_in=_meta.get("barge_in", False)))
            elif msg.type == WSMsgType.ERROR:
                LOG.warning("ws error from %s: %s", peer, ws.exception())
                break
    finally:
        if BRIDGE is not None:
            BRIDGE.unregister_broadcast(ws)
        if state.debounce_task and not state.debounce_task.done():
            state.debounce_task.cancel()
        if state.agent_task and not state.agent_task.done():
            state.agent_task.cancel()
        for tt in state.text_tasks:
            if not tt.done():
                tt.cancel()
        LOG.info("ws close: %s", peer)
        if not ws.closed:
            await ws.close()
    return ws


# ============================================================================ #
# App-Boot
# ============================================================================ #
@web.middleware
async def _security_headers_mw(request, handler):
    resp = await handler(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    return resp


def build_app() -> web.Application:
    app = web.Application(client_max_size=16 * 1024 * 1024,
                          middlewares=[_security_headers_mw])
    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/upload", upload_image)
    if STATIC_DIR.exists():
        app.router.add_static("/static/", STATIC_DIR, show_index=False)
    app.router.add_static("/uploads/", UPLOAD_DIR, show_index=False)
    return app


async def init_backends(cfg: Config):
    """Baut und lädt die gewählten Backends. Wirft bei Fehler (Caller beendet)."""
    stt = STTBackend.from_config(cfg)
    tts = TTSBackend.from_config(cfg)
    llm = LLMBackend.from_config(cfg)

    await stt.load()
    await tts.load()
    await llm.load()

    if cfg.stt_warmup:
        try:
            import numpy as np
            t0 = time.time()
            await stt.transcribe(np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32).tobytes(),
                                 SAMPLE_RATE)
            LOG.info("STT Warm-up: %.2fs", time.time() - t0)
        except Exception as exc:
            LOG.warning("STT Warm-up: %s", exc)
    if cfg.tts_warmup:
        try:
            t0 = time.time()
            await tts.synth("Hallo.", speed=cfg.tts_speed)
            LOG.info("TTS Warm-up: %.2fs", time.time() - t0)
        except Exception as exc:
            LOG.warning("TTS Warm-up: %s", exc)

    conv = ConversationManager(llm, system_prompt=cfg.resolved_voice_system(),
                               history_turns=cfg.llm_history_turns)
    return stt, tts, conv


async def main():
    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    try:
        cfg.validate()
    except Exception:
        LOG.exception("Konfiguration ungültig")
        sys.exit(2)

    if cfg.house_mode:
        LOG.info("🏠 HOUSE_MODE aktiv — speaker_id=%d wake_word=%d auth=%d",
                 cfg.house_speaker_id, cfg.house_wake_word, cfg.house_auth)

    try:
        stt, tts, conv = await init_backends(cfg)
    except Exception:
        LOG.exception("Backend-Initialisierung fehlgeschlagen")
        sys.exit(3)

    bridge = None  # Telegram-Bridge ist Legacy/optional; per Default aus.

    configure(cfg, stt=stt, tts=tts, conv=conv, bridge=bridge)

    LOG.info("STT-Backend: %s · %s", cfg.stt_backend, stt.describe())
    LOG.info("TTS-Backend: %s · %s", cfg.tts_backend, tts.describe())
    LOG.info("LLM-Backend: %s · %s", cfg.llm_backend, conv.llm.describe())

    app = build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, cfg.host, cfg.port)
    await site.start()

    LOG.info("🎙️  Voice-Chat Server läuft auf http://%s:%s (Agent: %s)",
             cfg.host, cfg.port, cfg.agent_name)
    LOG.info("    Debounce: %d ms · TTS speed: %.2f", cfg.debounce_ms, cfg.tts_speed)

    stop_event = asyncio.Event()

    def _request_stop(*_):
        LOG.info("Shutdown angefordert.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, _request_stop)
    await stop_event.wait()

    LOG.info("Stoppe Server …")
    if BRIDGE is not None:
        await BRIDGE.stop()
    await runner.cleanup()
    if conv is not None and hasattr(conv.llm, "close"):
        await conv.llm.close()


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    run()
