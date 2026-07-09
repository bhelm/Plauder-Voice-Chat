#!/usr/bin/env python3
"""Voice-Chat Server — HTTP/WebSocket layer.

Transport + turn orchestration only. STT/TTS/LLM sit behind pluggable
backends (see plauder.backends), chosen via .env. Text processing
(sanitizer, hallucination filter, merging) and turn state are separate modules.

Pipeline (STREAMING=1, the default — see _stream_reply_and_tts):
  Browser (16 kHz float32 PCM via VAD/push-to-talk/wake word)
    └─ WebSocket → TurnState (debounce + coalescing)
                 → STT.transcribe → hallucination filter
                 → speaker-lock gate → wake-word gate
                 → ConversationManager.chat_stream (LLM tokens live)
                 → sanitizer → sentence-wise TTS → VCT2 PCM chunks → browser
                   (VCT3 opus chunks when the client negotiated audioCodec=opus)
STREAMING=0 falls back to chat() → TTS.synth → one WAV (VCT1).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from collections import deque
from pathlib import Path

from aiohttp import WSMsgType, web

from . import audio as audio_utils
from . import opus_codec
from . import sanitizer
from . import voice_clone
from . import voices as voices_mod
from . import wake
from .backends import LLMBackend, STTBackend, TTSBackend, UpstreamTimeoutError
from .config import SAMPLE_RATE, Config
from .hermes_history import fetch_history
from .images import _resolve_image_urls
from .session import ConversationManager
from .speaker_verify import foreign_regions, keep_regions
from .telegram_bridge import TelegramBridge
from .turn_state import (TurnState, current_hermes_session_id,
                         rotate_hermes_session_id, vad_params_for_debounce)

LOG = logging.getLogger("voice-chat")

HERE = Path(__file__).resolve().parent.parent
STATIC_DIR = HERE / "static"
INDEX_HTML = STATIC_DIR / "index.html"

WS_MAX_MSG_BYTES = 16 * 1024 * 1024   # max size of a single WebSocket frame (image data URLs)
WS_HEARTBEAT_S = 20.0                 # WebSocket ping interval
AGENT_RETRY_DELAY_S = 0.5             # pause before the single silent retry on an idle timeout
SEGMENT_ID_LEN = 8                    # length of the short hex ids used for segments/turns
TTS_PREFETCH = 1                      # sentences synthesized ahead while the current one ships
                                      # (hides the TTS server's per-request time-to-first-byte)

STAGE = 6  # protocol version (client-compatible)


def _short_id() -> str:
    """Short random hex id for segments / turns / ephemeral users."""
    return uuid.uuid4().hex[:SEGMENT_ID_LEN]


def _opus_active() -> bool:
    """Opus compression usable on this server: enabled via AUDIO_OPUS AND the
    codec module actually loads (opuslib + system libopus). Drives the hello
    advertisement and every negotiation checkpoint."""
    return bool(CFG and CFG.audio_opus and opus_codec.is_available())


# Voice library / cloning handlers live in plauder.voice_clone (they read the
# runtime state below via `server.<name>` at call time).

# --------------------------------------------------------------------------- #
# Runtime state (filled by configure() / main()). Module globals, so the
# turn handlers can access them without a request context.
# --------------------------------------------------------------------------- #
CFG: Config | None = None
STT: STTBackend | None = None
TTS: TTSBackend | None = None
CONV: ConversationManager | None = None
BRIDGE: TelegramBridge | None = None
GHOST: sanitizer.HallucinationFilter | None = None
# Speaker lock (voice gate). None = disabled; otherwise a SpeakerVerifier whose
# .active() says whether it currently gates (model loaded AND a profile enrolled).
SPEAKER = None
# Voice library (cloned-voice CRUD + active-voice selection). None = disabled;
# a VoiceLibrary when TTS_CLONE_ENABLED and TTS points at the OmniVoice wrapper.
VOICES = None

# House-Mode speaker-ID (lazy, optional — the speaker_id module is not part of
# this repo; stays disabled when it is missing).
_SPEAKER_IDENTIFIER = None
_SPEAKER_INIT_FAILED = False


def configure(cfg: Config, *, stt=None, tts=None, conv=None, bridge=None, ghost=None,
              speaker=None, voices=None):
    """Sets the runtime state. Tests can inject mock backends here."""
    global CFG, STT, TTS, CONV, BRIDGE, GHOST, SPEAKER, VOICES
    CFG = cfg
    STT = stt
    TTS = tts
    CONV = conv
    BRIDGE = bridge
    GHOST = ghost if ghost is not None else sanitizer.HallucinationFilter.from_config(cfg)
    SPEAKER = speaker
    VOICES = voices


def get_speaker_identifier():
    """Lazy-load of the optional House-Mode speaker identifier."""
    global _SPEAKER_IDENTIFIER, _SPEAKER_INIT_FAILED
    if CFG is None or not CFG.house_speaker_id or _SPEAKER_INIT_FAILED:
        return None
    if _SPEAKER_IDENTIFIER is None:
        try:
            import speaker_id as _spk  # optional, not in the repo
            data_dir = Path(CFG.house_data_dir)
            embedder = _spk.SpeakerEmbedder(str(data_dir / "models" / "campplus_multilingual.onnx"))
            store = _spk.SpeakerStore(str(data_dir / "speakers.json"))
            _SPEAKER_IDENTIFIER = _spk.SpeakerIdentifier(embedder, store)
            LOG.info("🔊 Speaker-ID active (%d speakers)", len(store.all()))
        except Exception as exc:
            _SPEAKER_INIT_FAILED = True
            LOG.warning("Speaker-ID init failed, disabled: %s", exc)
            return None
    return _SPEAKER_IDENTIFIER


# --------------------------------------------------------------------------- #
# Identity helpers (for hello/healthz frames)
# --------------------------------------------------------------------------- #
def _agent_id() -> str:
    return CFG.openclaw_agent_id if CFG else "antonia"


def _default_user() -> str:
    return CFG.openclaw_user_id if CFG else "voice-user"


def _session_key_for_user(user_id: str) -> str:
    return f"agent:{_agent_id()}:openai-user:{user_id}"


def _hermes_session_key_for_mode(mode: str) -> str:
    """Return the Hermes session key. Always uses the 'separate' key — the main
    mode toggle was removed because without a gateway change the main key only
    provides Honcho peer-memory (identical to separate) but does NOT share the
    Telegram turn history."""
    if not CFG:
        return ""
    return CFG.hermes_session_key_separate


def _apply_hermes_headers(state: TurnState) -> None:
    """Set the Hermes session headers on the LLM backend right before a request.
    The LLM backend is a process-wide singleton; we update its mutable
    session_key / session_id so the next _build_request() includes the correct
    X-Hermes-Session-Key / X-Hermes-Session-Id headers.
    Always uses the 'separate' session (mode toggle removed)."""
    if CONV is None:
        return
    llm = CONV.llm
    llm.session_key = _hermes_session_key_for_mode("separate")
    # Re-read the ACTIVE session ID from the persistent store on every call: a
    # "new session" reset on any device must apply to every connection's next
    # turn immediately, not only after its reconnect. Without a configured
    # Hermes key there is no shared store — keep the per-connection ID.
    if CFG and CFG.hermes_session_key_separate:
        state.hermes_session_id_separate = current_hermes_session_id()
    llm.session_id = state.hermes_session_id_separate
    LOG.debug("hermes headers: key=%s sid=%s",
              llm.session_key, llm.session_id[:12] if llm.session_id else "")


# ============================================================================ #
# Cross-device sync
# ============================================================================ #
# All connected browsers (ws → TurnState). The voice session is shared across
# devices (one Hermes session), so committed messages and "new session" resets
# are mirrored to every other connection to keep all UIs on the same state.
WS_CLIENTS: dict = {}

# Fire-and-forget broadcast tasks (kept referenced so they aren't GC'd).
_SYNC_TASKS: set = set()

# --- STT warmup (self-hosted Whisper endpoints) ----------------------------
# A cold faster-whisper server loads its model on the FIRST request (~30 s
# observed) — without a warmup that lands on the user's first real segment,
# with zero feedback while transcribe() serializes everything behind its
# lock. On WS connect we pull the load forward with a short silence clip.
# Only for self-hosted endpoints (base_url set): the real OpenAI API is
# always warm and the call would cost money. Throttled globally.
_STT_WARM_INTERVAL_S = 300.0
_stt_warm = {"task": None, "ts": 0.0}


def _maybe_warmup_stt() -> None:
    if STT is None or not getattr(STT, "base_url", None):
        return
    now = time.time()
    task = _stt_warm["task"]
    if (task is not None and not task.done()) or \
            now - _stt_warm["ts"] < _STT_WARM_INTERVAL_S:
        return
    _stt_warm["ts"] = now

    async def _warm():
        t0 = time.time()
        try:
            # 0.5 s of f32 silence — enough to force the remote model load.
            await STT.transcribe(b"\x00" * (SAMPLE_RATE * 4 // 2), SAMPLE_RATE)
            LOG.info("stt warmup: done in %d ms", int((time.time() - t0) * 1000))
        except Exception as exc:
            LOG.warning("stt warmup failed after %d ms: %s",
                        int((time.time() - t0) * 1000), exc)

    _stt_warm["task"] = asyncio.create_task(_warm())


async def _broadcast_peers(origin_ws, payload: dict) -> None:
    """Best-effort fan-out to every OTHER connected browser."""
    for peer_ws in [w for w in WS_CLIENTS if w is not origin_ws]:
        try:
            await peer_ws.send_json(payload)
        except Exception:
            pass  # a dying socket is cleaned up by its own handler


def _broadcast_peers_bg(origin_ws, payload: dict) -> None:
    """Like _broadcast_peers, but off the latency-critical turn path."""
    task = asyncio.create_task(_broadcast_peers(origin_ws, payload))
    _SYNC_TASKS.add(task)
    task.add_done_callback(_SYNC_TASKS.discard)


# ============================================================================ #
# HTTP routes
# ============================================================================ #
async def index(_request):
    if not INDEX_HTML.exists():
        return web.Response(status=500, text=f"index.html missing: {INDEX_HTML}")
    # Inject the configured UI language (APP_LANGUAGE) so the page renders in the
    # right locale immediately, without a flash of the fallback language, and the
    # sub-path prefix (BASE_PATH) so the client builds WS/upload/asset URLs that
    # resolve behind a reverse proxy.
    lang = (CFG.app_language if CFG else "en")
    base = (CFG.base_path if CFG else "")
    html = (INDEX_HTML.read_text(encoding="utf-8")
            .replace("__APP_LANG__", lang)
            .replace("__BASE_PATH__", base))
    return web.Response(text=html, content_type="text/html")


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


# ============================================================================ #
# Turn orchestration
# ============================================================================ #
def _combine_user_input(voice_merged: str, text_parts: list, resume: str = "") -> str:
    parts: list = []
    if resume and resume.strip():
        # Carried-over input of a coalesced (barged-in) turn: its own
        # paragraph, not sentence-glued into the continued speech.
        parts.append(resume.strip())
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
    """Calls the ConversationManager; on a 408, retries once silently.
    Returns (reply, meta, retried) or raises on the second failure."""
    try:
        reply, meta = await CONV.chat(combined, user_key=user_key, image_urls=image_urls)
        return reply, meta, False
    except UpstreamTimeoutError as exc:
        if not allow_retry:
            raise
        LOG.warning("agent upstream timeout, retry once: %s", exc)
        await asyncio.sleep(AGENT_RETRY_DELAY_S)
        reply, meta = await CONV.chat(combined, user_key=user_key, image_urls=image_urls)
        return reply, meta, True


async def _send_turn_pending(ws, state: TurnState):
    voice_merged = sanitizer.merge_transcripts(state.pending_texts)
    combined = _combine_user_input(voice_merged, state.pending_text_parts,
                                   resume=state.pending_resume)
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
    combined = _combine_user_input(voice_merged, text_parts,
                                   resume=state.pending_resume)
    LOG.info("turn=%s commit: voice_segs=%d text_sends=%d imgs=%d combined=%r",
             turn_id, len(state.pending_texts), len(text_parts), len(image_urls), combined[:200])

    state.inflight_combined = combined   # for coalescing (see _capture_coalesce)
    state.inflight_images = list(image_urls)
    state.audio_started = False          # no audio of THIS turn emitted yet
    if BRIDGE:
        BRIDGE.begin_local_call()
    try:
        # A carried-over resume counts as voice for the source classification
        # (it originated from speech; _run_turn_inner only uses it for that).
        await _run_turn_inner(ws, state, turn_id, combined=combined,
                              voice_merged=voice_merged or state.pending_resume,
                              text_parts=text_parts,
                              image_urls=image_urls, segment_ids=segment_ids)
    finally:
        state.inflight_combined = None
        state.inflight_images = []
        if BRIDGE:
            BRIDGE.end_local_call()


async def _run_turn_inner(ws, state, turn_id, *, combined, voice_merged,
                          text_parts, image_urls, segment_ids):
    _apply_hermes_headers(state)
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
    # Mirror the committed user input to other connected devices.
    if combined or resolved_imgs:
        _broadcast_peers_bg(ws, {
            "type": "chat.remote", "role": "user", "text": combined,
            "imageCount": len(resolved_imgs), "ts": time.time(),
        })

    # Telegram mirroring of the user input.
    if BRIDGE and BRIDGE.enabled:
        prefix = {"voice": "👤 (Voice)", "text": "👤 (Voice-Chat)",
                  "mixed": "👤 (Voice+Text)"}[source]
        mirror_parts = []
        if combined:
            mirror_parts.append(combined)
        for _ in range(len(resolved_imgs)):
            mirror_parts.append("📷 Image sent")
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

    # --- Streaming path (A1+A2): LLM tokens live → sentence-wise TTS → PCM chunks ---
    if CFG and CFG.streaming and hasattr(CONV, "chat_stream"):
        await _stream_reply_and_tts(
            ws, state, turn_id, reply_id,
            combined=combined, resolved_imgs=resolved_imgs)
        return

    # --- Classic path (fallback, STREAMING=0): fully generate first, then 1 WAV ---
    t_llm = time.time()
    try:
        allow_retry = (CFG.llm_retry_timeout_on_idle
                       and not _queue_has_more_work(state, exclude_task=state.agent_task))
        _apply_hermes_headers(state)   # see _consume_stream: singleton race
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
        await _reopen_wake_window_after_silent(ws, state)
        return
    except Exception as exc:
        LOG.exception("turn=%s agent failed", turn_id)
        await ws.send_json({"type": "reply.error", "turnId": turn_id, "replyId": reply_id,
                            "error": str(exc), "ts": time.time()})
        await _reopen_wake_window_after_silent(ws, state)
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
        await _reopen_wake_window_after_silent(ws, state)
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
    _broadcast_peers_bg(ws, {"type": "chat.remote", "role": "assistant",
                             "text": reply_text, "ts": time.time()})

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
        start_evt["debounceMs"] = state.debounce_ms  # pause share of e2e
    state.audio_started = True   # reply is audible now (classic path)
    state.client_playing = True
    await ws.send_json(start_evt)
    t_tts = time.time()
    _synth_kw = {"speed": state.speed}
    _vid = voice_clone.active_voice_id()
    if _vid is not None:
        _synth_kw["voice"] = _vid
    try:
        pcm_bytes, sr = await TTS.synth(cleaned, **_synth_kw)
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


async def _tts_synth_stream(text: str, speed: float, voice: str | None = None):
    """TTS streaming with fallback: uses ``synth_stream`` if present, otherwise
    the classic ``synth`` (one chunk). Keeps the orchestrator backend-agnostic.
    ``voice`` optionally overrides the default voice (cloned-voice id); it is
    only forwarded when set, so duck-typed backends without a ``voice`` kwarg
    keep working."""
    kw = {"speed": speed}
    if voice is not None:
        kw["voice"] = voice
    fn = getattr(TTS, "synth_stream", None)
    if fn is not None:
        async for item in fn(text, **kw):
            yield item
    else:
        pcm, sr = await TTS.synth(text, **kw)
        if pcm:
            yield pcm, sr


async def _stream_reply_and_tts(ws, state: TurnState, turn_id, reply_id, *,
                                combined, resolved_imgs):
    """Streaming turn (A1+A2).

    LLM tokens are read live; as soon as a sentence is complete, it goes into
    the TTS queue. A parallel TTS worker synthesizes sentence by sentence and
    sends the audio as PCM chunks (VCT2) progressively to the client — sentence 1
    is played back while sentence 2 is still being generated/synthesized.
    """
    pron = CFG.pronunciations_file if CFG else None
    # Cloned voice for this whole reply, resolved once (None = backend default).
    reply_voice = voice_clone.active_voice_id()
    max_chars = CFG.tts_max_chars_per_chunk if CFG else 220
    # The FIRST sentence gates time-to-first-audio: force-flush it earlier so a
    # long punctuation-free opener doesn't stall TTS for tens of tokens.
    first_max = CFG.tts_first_chunk_chars if CFG else 100
    if first_max <= 0:
        first_max = max_chars
    first_max = min(first_max, max_chars)
    got_sentence = {"v": False}
    audio_id = f"audio-{turn_id}"
    sentence_q: asyncio.Queue = asyncio.Queue()

    parts: list[str] = []          # all LLM deltas received so far
    held: list[str] = []           # finished sentences still waiting for TTS release
    flushed_any = {"v": False}
    sent_len = {"v": 0}            # text length already sent to the client
    started = {"audio": False}
    sr_box = {"sr": None}
    t_tts0 = {"t": None}
    t_first = {"t": None}          # timestamp of the first LLM token
    # E2E anchor ("user done speaking"); freeze it now so a later incoming
    # segment doesn't shift this turn's measurement point.
    anchor = getattr(state, "speech_end_ts", 0.0) or 0.0

    # Downlink codec: the client opted into opus via settings.audioCodec (only
    # stored as "opus" when the codec module is usable — see the settings
    # handler). One encoder per audio stream (turn), created at first chunk
    # with the real TTS sample rate; packets are batched so one VCT3 frame
    # covers ~tts_chunk_ms of audio, and the encoder is flushed at stream end.
    want_opus = getattr(state, "audio_codec", "pcm") == "opus" and _opus_active()

    async def _tts_worker() -> int:
        seq = 0
        opus_enc = None
        opus_batch: list[bytes] = []
        chunk_ms = CFG.tts_chunk_ms if CFG else 400
        batch_packets = max(1, int(chunk_ms) // opus_codec.OpusEncoder.FRAME_MS)

        async def _ship_batch():
            nonlocal seq
            if opus_batch:
                seq += 1
                await ws.send_bytes(audio_utils.wrap_opus_chunk(turn_id, seq, opus_batch))
                opus_batch.clear()

        # --- Sentence prefetch -------------------------------------------------
        # The TTS server has a fixed per-request time-to-first-byte (~0.8 s for
        # Kokoro), so requesting sentence N+1 only after N finished streaming
        # creates audible gaps whenever a sentence's audio is shorter than that
        # overhead. A feeder therefore keeps up to TTS_PREFETCH+1 synth requests
        # in flight; each writes into its own chunk queue, and the shipper below
        # drains those queues strictly IN ORDER — playback order is unaffected.
        synth_tasks: set = set()
        sem = asyncio.Semaphore(1 + TTS_PREFETCH)
        order_q: asyncio.Queue = asyncio.Queue()

        async def _synth_to_queue(sentence: str, chunk_q: asyncio.Queue):
            if t_tts0["t"] is None:
                t_tts0["t"] = time.time()
            # Per-sentence lead gate: OmniVoice prepends a faint noise blob to
            # every synthesis — audible as a tiny "hah" at each sentence seam.
            gate = None
            last_sr = None
            try:
                async for pcm, sr in _tts_synth_stream(sentence, state.speed, reply_voice):
                    if gate is None:
                        gate = audio_utils.StreamLeadGate(sr)
                    last_sr = sr
                    pcm = gate.process(pcm)
                    if pcm:
                        await chunk_q.put((pcm, sr))
                # Never-opened gate: ship the held tail as silence so the
                # sentence's duration (and thus pause timing) is preserved.
                if gate is not None and last_sr is not None:
                    tail = gate.flush()
                    if tail:
                        await chunk_q.put((tail, last_sr))
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("turn=%s tts sentence failed", turn_id)
            finally:
                sem.release()
                await chunk_q.put(None)

        async def _feeder():
            while True:
                sentence = await sentence_q.get()
                if sentence is None:
                    await order_q.put(None)
                    return
                await sem.acquire()
                chunk_q: asyncio.Queue = asyncio.Queue()
                task = asyncio.create_task(_synth_to_queue(sentence, chunk_q))
                synth_tasks.add(task)
                task.add_done_callback(synth_tasks.discard)
                await order_q.put(chunk_q)

        feeder_task = asyncio.create_task(_feeder())

        async def _iter_chunks_in_order():
            while True:
                chunk_q = await order_q.get()
                if chunk_q is None:
                    return
                while True:
                    item = await chunk_q.get()
                    if item is None:
                        break
                    yield item

        try:
            async for pcm, sr in _iter_chunks_in_order():
                if not pcm:
                    continue
                if not started["audio"]:
                    # Create the encoder BEFORE audio.start so the codec
                    # announced there is the one actually used (an init
                    # failure falls back to raw VCT2 for the whole turn).
                    if want_opus and opus_enc is None:
                        try:
                            opus_enc = opus_codec.OpusEncoder(
                                sr, bitrate=opus_codec.DOWNLINK_BITRATE)
                        except Exception:
                            LOG.exception(
                                "turn=%s opus encoder init failed — "
                                "falling back to PCM", turn_id)
                    started["audio"] = True
                    state.audio_started = True   # reply is audible now
                    state.client_playing = True
                    sr_box["sr"] = sr
                    now = time.time()
                    start_evt = {
                        "type": "audio.start", "turnId": turn_id, "audioId": audio_id,
                        "sampleRate": sr,
                        "codec": "opus" if opus_enc is not None else "pcm",
                        "ts": now}
                    # Latency breakdown up to the FIRST playback (≠ total time):
                    if anchor:
                        start_evt["e2eMs"] = int((now - anchor) * 1000)
                        start_evt["debounceMs"] = state.debounce_ms  # pause share of e2e
                    if t_first["t"] is not None:
                        start_evt["llmFirstMs"] = int((t_first["t"] - t_llm) * 1000)
                    if t_tts0["t"] is not None:
                        start_evt["ttsFirstMs"] = int((now - t_tts0["t"]) * 1000)
                    await ws.send_json(start_evt)
                if opus_enc is not None:
                    for pkt in opus_enc.encode_pcm16(pcm):
                        opus_batch.append(pkt)
                        if len(opus_batch) >= batch_packets:
                            await _ship_batch()
                    continue
                frame_bytes = max(2, int(sr * (CFG.tts_chunk_ms / 1000.0)) * 2)
                for frame in audio_utils.iter_pcm_frames(pcm, frame_bytes):
                    seq += 1
                    await ws.send_bytes(audio_utils.wrap_pcm_chunk(turn_id, seq, frame))
        finally:
            # Normal end, barge-in cancel or a dead socket: never orphan the
            # feeder or an in-flight prefetch synthesis.
            feeder_task.cancel()
            for t in list(synth_tasks):
                t.cancel()
            await asyncio.gather(feeder_task, *synth_tasks, return_exceptions=True)
        # Stream over: flush the encoder's trailing partial frame and ship the
        # remaining batch before audio.end is emitted by the caller.
        if opus_enc is not None:
            try:
                opus_batch.extend(opus_enc.flush())
            except Exception:
                LOG.exception("turn=%s opus flush failed", turn_id)
            await _ship_batch()
        return seq

    async def _emit_text():
        """Sends new reply text to the client — but only once it's clear that
        it's not a pure NO_REPLY (otherwise 'NO_REPLY' — or a partial 'NO_'
        token — would briefly flash up)."""
        full = "".join(parts)
        if not flushed_any["v"] and sanitizer.is_no_reply_prefix(full.strip()):
            return
        if len(full) > sent_len["v"]:
            await ws.send_json({
                "type": "reply.delta", "turnId": turn_id, "replyId": reply_id,
                "delta": full[sent_len["v"]:], "ts": time.time()})
            sent_len["v"] = len(full)

    async def _release(force: bool = False):
        full = "".join(parts).strip()
        if not flushed_any["v"] and not force and sanitizer.is_no_reply_prefix(full):
            return  # could still become NO_REPLY → hold back the sentences
        while held:
            cleaned = sanitizer.sanitize_for_tts(held.pop(0), pronunciations_file=pron)
            if cleaned:
                await sentence_q.put(cleaned)
                flushed_any["v"] = True

    pending = ""

    async def _consume_stream():
        nonlocal pending
        # Re-apply THIS connection's Hermes session id right before the call:
        # it lives on the process-wide LLM singleton, and awaits between turn
        # start and request build let a concurrent connection overwrite it
        # (cross-session contamination). This shrinks the race window to the
        # request build itself.
        _apply_hermes_headers(state)
        agen = CONV.chat_stream(
            combined, user_key=state.session_user or _default_user(),
            image_urls=resolved_imgs if resolved_imgs else None)
        async for delta in agen:
            if t_first["t"] is None:
                t_first["t"] = time.time()
            parts.append(delta)
            await _emit_text()
            pending += delta
            sents, pending = audio_utils.split_stream_sentences(
                pending, max_chars if got_sentence["v"] else first_max)
            if sents:
                got_sentence["v"] = True
                held.extend(sents)
                await _release()

    # Bind t_llm BEFORE starting the worker: the worker reads it in its closure
    # (for llmFirstMs). If an `await` later sat between create_task and this
    # assignment, the worker could otherwise read it unbound (NameError).
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
            await asyncio.sleep(AGENT_RETRY_DELAY_S)
            pending = ""
            await _consume_stream()
    except asyncio.CancelledError:
        LOG.info("turn=%s stream cancelled", turn_id)
        tts_task.cancel()
        await asyncio.gather(tts_task, return_exceptions=True)
        # Do NOT discard audio_id: _cancel_in_flight then still sends an
        # audio.stop to the client (backup for the local barge-in stop).
        raise
    except UpstreamTimeoutError as exc:
        await _drain_tts()
        state.audio_ids.discard(audio_id)
        LOG.warning("turn=%s agent upstream timeout (stream, no retry): %s", turn_id, exc)
        await ws.send_json({"type": "reply.error", "turnId": turn_id, "replyId": reply_id,
                            "error": "upstream provider timeout",
                            "errorKind": "upstream_timeout", "ts": time.time()})
        await _reopen_wake_window_after_silent(ws, state)
        return
    except Exception as exc:
        await _drain_tts()
        state.audio_ids.discard(audio_id)
        LOG.exception("turn=%s agent stream failed", turn_id)
        await ws.send_json({"type": "reply.error", "turnId": turn_id, "replyId": reply_id,
                            "error": str(exc), "ts": time.time()})
        await _reopen_wake_window_after_silent(ws, state)
        return

    # From here on the TTS worker is still alive — a barge-in cancel landing in
    # any of the awaits below must not orphan it (it would keep sending audio
    # of the discarded turn and then block forever on the sentence queue).
    try:
        # Take the remaining buffer as the last sentence.
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
            await _reopen_wake_window_after_silent(ws, state)
            return

        await _release(force=True)            # release held-back sentences
        await _emit_text()                    # remaining text to the client
        cleaned_full = sanitizer.sanitize_for_tts(full_reply, pronunciations_file=pron)
        LOG.info("turn=%s agent ok llm=%dms text=%r (stream)", turn_id, llm_ms, full_reply[:120])
        await ws.send_json({
            "type": "reply", "turnId": turn_id, "replyId": reply_id,
            "text": full_reply, "cleanedText": cleaned_full, "llmMs": llm_ms,
            "finishReason": meta.get("finish_reason"), "usage": meta.get("usage"),
            "streamed": True, "ts": time.time()})
        _broadcast_peers_bg(ws, {"type": "chat.remote", "role": "assistant",
                                 "text": full_reply, "ts": time.time()})

        if BRIDGE and BRIDGE.enabled and cleaned_full:
            BRIDGE.remember_self_sent(full_reply)
            asyncio.create_task(BRIDGE.send(f"❤️ {CFG.agent_name}: {cleaned_full}", echo_text=full_reply))

        results = await _drain_tts()
        state.audio_ids.discard(audio_id)
    except asyncio.CancelledError:
        tts_task.cancel()
        await asyncio.gather(tts_task, return_exceptions=True)
        raise
    total_chunks = next((r for r in results if isinstance(r, int)), 0)
    if started["audio"]:
        tts_ms = int((time.time() - t_tts0["t"]) * 1000) if t_tts0["t"] else 0
        LOG.info("turn=%s tts ok chunks=%d sr=%s tts=%dms speed=%.2f (stream)",
                 turn_id, total_chunks, sr_box["sr"], tts_ms, state.speed)
        await ws.send_json({
            "type": "audio.end", "turnId": turn_id, "audioId": audio_id,
            "chunks": total_chunks, "sampleRate": sr_box["sr"], "ttsMs": tts_ms,
            "speed": state.speed, "ts": time.time()})


def _resume_debounce(ws, state: TurnState) -> None:
    """(Re)arm the debounce timer when coalesced input is still pending.

    Used by the segment paths that drop the current segment (ghost-filtered,
    wake-gated, speaker-rejected, empty transcript) but must not strand earlier
    pending input. Only needed on the NON-gated path, where the up-front
    `_cancel_in_flight` killed the ticking timer before the segment was dropped.

    Deliberately does NOT cancel an existing debounce_task (that slot also
    holds the *running* turn once the timer elapses — cancelling would abort an
    in-flight reply). Crucially it also must NOT re-arm while that slot is
    still alive: the pending lists are only cleared AFTER the running turn
    completes, so re-arming here would dispatch the SAME user input to the LLM
    a second time (duplicate turn, interleaved replies) whenever a foreign
    voice / ghost / empty segment is dropped mid-reply.
    """
    if not state.has_pending():
        return
    if state.debounce_task and not state.debounce_task.done():
        return  # timer still ticking or turn still running → nothing stranded
    if state.agent_task and not state.agent_task.done():
        return  # turn already dispatched; the pending lists belong to it
    state.debounce_task = asyncio.create_task(_debounce_then_run(ws, state))


def _spawn_tracked(state: TurnState, coro):
    """Launch a detached handler task but track it so the connection can cancel
    it on close (otherwise a late handler may send on a closed WebSocket)."""
    task = asyncio.create_task(coro)
    state.inflight_tasks.add(task)
    task.add_done_callback(state.inflight_tasks.discard)
    return task


def _cancel_connection_tasks(state: TurnState) -> None:
    """Cancel every task owned by a connection (debounce, agent, legacy text,
    and the tracked segment/partial handlers). Called when the WS closes."""
    for task in (state.debounce_task, state.agent_task):
        if task and not task.done():
            task.cancel()
    for task in list(state.text_tasks) + list(state.inflight_tasks):
        if not task.done():
            task.cancel()


# Minimum debounce sleep even when the pause window is already used up by
# STT/gate latency — leaves one tick for turn.pending to render and for a
# same-instant second segment to coalesce.
DEBOUNCE_MIN_WAIT_S = 0.05


async def _debounce_then_run(ws, state: TurnState):
    try:
        # Count the pause from the moment the user stopped speaking (anchor),
        # not from when STT + gates finished: their latency already consumed
        # part of the debounce window.
        delay = state.debounce_ms / 1000.0
        if state.debounce_anchor:
            delay -= time.time() - state.debounce_anchor
        await asyncio.sleep(max(DEBOUNCE_MIN_WAIT_S, delay))
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


def _capture_coalesce(state: TurnState) -> str | None:
    """Text of an in-flight turn whose reply has NOT started playing yet.
    The owner continuing to speak then means "I wasn't done" — the cancelled
    turn's input is re-queued into the next turn (coalescing) instead of an
    already-committed question silently vanishing. Once audio plays, more
    speech is a genuine interruption and nothing is re-queued."""
    if (state.agent_task and not state.agent_task.done()
            and not state.audio_started and state.inflight_combined):
        return state.inflight_combined
    return None


def _capture_coalesce_images(state: TurnState) -> list:
    """Image URLs belonging to the coalescable in-flight turn (see above)."""
    if (state.agent_task and not state.agent_task.done()
            and not state.audio_started):
        return list(state.inflight_images)
    return []


def _hermes_keeps_history() -> bool:
    """True when the LLM backend continues a server-side Hermes session
    (X-Hermes-Session-Id). The gateway then loads the transcript from ITS
    state.db and IGNORES the request-body history — and it persists the user
    message of a cancelled turn anyway (finalize_turn runs on interrupt). Re-
    sending that text would duplicate it in the model's real context, so
    coalescing is then VISUAL only (turn.discarded coalesced flag).

    Signal = a Hermes session KEY is configured (plain OpenAI-compatible
    backends get the session headers too, but ignore them — for those the
    request body is the only history and re-queuing IS needed)."""
    return bool(CFG is not None and CFG.hermes_session_key_separate
                and CONV is not None and getattr(CONV.llm, "session_id", ""))


async def _coalesce_cancel(state: TurnState, ws):
    """Cancel the in-flight turn; if its reply was not audible yet, carry the
    input over — visually always (flag), textually only when the backend does
    not keep server-side history."""
    resume = _capture_coalesce(state)
    resume_imgs = _capture_coalesce_images(state)
    requeue = bool(resume) and not _hermes_keeps_history()
    await _cancel_in_flight(state, ws, coalesced=bool(resume),
                            coalesced_text=resume, requeued=requeue)
    if requeue:
        # As its own paragraph, not merged into the continued speech
        # (see _combine_user_input / TurnState.pending_resume).
        state.pending_resume = resume
        if resume_imgs:
            state.pending_image_urls[:0] = resume_imgs
        LOG.info("turn coalesced: re-queued %r (+%d imgs)",
                 resume[:60], len(resume_imgs))


async def _cancel_in_flight(state: TurnState, ws, coalesced: bool = False,
                            coalesced_text: str | None = None,
                            requeued: bool = False):
    # A cancellation ends the running reply → any "do-not-reopen-window" flag
    # that may have been set is thereby moot.
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
        evt = {"type": "turn.discarded", "turnId": state.turn_id,
               "reason": "new-speech-detected",
               "coalesced": coalesced, "ts": time.time()}
        if coalesced:
            # requeued=True → the text rides back into the next turn.pending
            # server-side; False (Hermes keeps history) → the CLIENT must carry
            # the old text visually, coalescedText is its source.
            evt["requeued"] = requeued
            if coalesced_text:
                evt["coalescedText"] = coalesced_text
        await ws.send_json(evt)
    if tasks_to_await:
        await asyncio.gather(*tasks_to_await, return_exceptions=True)
    if state.audio_ids:
        for aid in list(state.audio_ids):
            await ws.send_json({"type": "audio.stop", "audioId": aid, "ts": time.time()})
        state.audio_ids.clear()
    elif state.client_playing:
        # All chunks were already SENT (audio_ids emptied at audio.end), but
        # the client is still playing buffered audio. An owner barge-in /
        # stop must silence that too — a bare audio.stop (no audioId) makes
        # the client stop everything.
        await ws.send_json({"type": "audio.stop", "ts": time.time()})
    state.client_playing = False


def _speaker_tag(text: str, speaker_info: dict | None) -> str:
    if speaker_info is None:
        return text
    nm = speaker_info["name"]
    role = speaker_info["role"]
    rel = speaker_info.get("relation") or ""
    if role == "admin":
        q = f"admin, {rel}" if rel else "admin"
        return f"[Speaker: {nm} ({q})] {text}"
    if speaker_info["known"]:
        return f"[Speaker: {nm} ({rel})] {text}" if rel else f"[Speaker: {nm}] {text}"
    return f"[Speaker: {nm}, unknown] {text}"


# After manually closing the conversation window, for this many seconds NO
# automatic reopening (swallows trailing partials / echo / interferers).
WAKE_CLOSE_GUARD_S = 2.0


def _wake_mode() -> str:
    """Normalized wake mode: 'alexa' (one-shot) or 'conversation'."""
    m = (getattr(CFG, "wake_mode", "conversation") or "conversation").lower() if CFG else "conversation"
    return "alexa" if m in ("alexa", "oneshot", "one-shot", "single") else "conversation"


def _wake_oneshot() -> bool:
    """Alexa mode: after every reply the conversation window closes immediately
    (no follow-up window — every command needs the wake word again)."""
    return _wake_mode() == "alexa"


def _wake_closed_active(state: TurnState, now: float | None = None) -> bool:
    """True as long as a manually closed window should deliberately stay closed:
    during the still-running reply (``wake_suppress_reopen``) OR within the short
    trailing guard (``wake_closed_until``). During this time the pipeline ignores
    wake detection COMPLETELY — otherwise echo of its own TTS, B2 partials or
    follow-up chatter (incl. fuzzy false matches) would reopen the window at once."""
    now = time.time() if now is None else now
    return bool(state.wake_suppress_reopen) or now < state.wake_closed_until


async def _open_wake_window(ws, state: TurnState, now: float | None = None,
                            reason: str = "command"):
    """(Re)open/refresh the conversation window and notify the client.

    `reason` tells the client whether Antonia is now IDLE (`armed` = waiting for
    the command, `done` = reply fully spoken) → start the idle timer for the
    acoustic wind-down feedback — or whether a command just came in (`command`)
    and a reply is coming → do NOT run the timer. This way the window only closes
    after the reply has been fully spoken, not during it."""
    now = time.time() if now is None else now
    # Guard after manual closing: do not reopen automatically.
    if _wake_closed_active(state, now):
        return
    state.wake_until = now + CFG.wake_word_window_s
    await ws.send_json({
        "type": "wake.window", "turnId": state.turn_id, "reason": reason,
        "windowS": CFG.wake_word_window_s, "ts": now})


async def _reopen_wake_window_after_silent(ws, state: TurnState):
    """A silent/failed reply produces no audio, so no playback.done will
    refresh the window — without this, the client-side window indicator and
    the server gate drift apart by the LLM latency (follow-ups get ignored
    while the mic still glows). Re-sync both with reason='done' (idle timer
    runs from now)."""
    if (CFG and state.wake_word_enabled and state.wake_until > 0
            and not _wake_oneshot()):
        await _open_wake_window(ws, state, reason="done")


async def _emit_wake_detected(ws, state: TurnState, segment_id):
    """Acoustic early feedback: send a `wake.detected` once per segment as soon
    as the wake word is recognized (partial OR final segment)."""
    if state.wake_detected_seg == segment_id:
        return
    # Guard after manual closing: no early feedback / reopening by stragglers.
    if _wake_closed_active(state):
        return
    state.wake_detected_seg = segment_id
    await ws.send_json({
        "type": "wake.detected", "segmentId": segment_id,
        "turnId": state.turn_id, "ts": time.time()})


async def _handle_audio_segment(ws, state: TurnState, pcm_bytes, segment_id, peer,
                                speech_start_ts=None, barge_in=False,
                                ptt=False):
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

    # A running turn is NOT cancelled up-front. Gated modes (wake word /
    # speaker lock) cancel only once the segment passes their gate; the plain
    # VAD path cancels right after the ghost filter below. Rationale: a
    # Whisper hallucination (mic noise → "Untertitelung des ZDF") must not
    # kill a thinking turn, and the extra STT latency is inaudible — audible
    # playback is already stopped client-side at VAD onset (bargeInStop).
    wake_gating = bool(CFG and getattr(state, "wake_word_enabled", False))
    speaker_gating = SPEAKER is not None and SPEAKER.active()

    # Speaker verification is local CPU work and independent of the transcript
    # — run it CONCURRENTLY with the (remote) STT instead of after it. The
    # result is consumed by the speaker gate below only if the segment survives
    # the text gates; on the drop paths the thread finishes in the background
    # and its outcome is discarded (the done-callback consumes any exception).
    verify_task = None
    if speaker_gating:
        verify_task = _spawn_tracked(state, asyncio.to_thread(
            SPEAKER.verify, pcm_bytes, SAMPLE_RATE, duration_s))
        verify_task.add_done_callback(lambda t: t.cancelled() or t.exception())

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
        # Nothing was cancelled yet (the barge-in is deferred past the ghost
        # filter), but resume defensively like every other drop path — the
        # guards inside make it a no-op while a timer ticks / a turn runs.
        _resume_debounce(ws, state)
        return
    LOG.info("stt seg=%s text=%r (%dms)", segment_id, text, stt_ms)

    # Hallucination filter — embedded first: a ghost sentence Whisper appended
    # at a mid-segment pause ("… reingeredet. Vielen Dank. Okay …") is pruned
    # without dropping the genuine rest.
    if text and GHOST:
        _stripped = GHOST.strip_ghost_sentences(text)
        if _stripped != text:
            LOG.info("ghost sentence stripped seg=%s %r → %r",
                     segment_id, text[:80], _stripped[:80])
            text = _stripped
    if text and GHOST and GHOST.is_hallucination(
            text, no_speech_prob=no_speech_prob, duration_s=duration_s):
        LOG.info("ghost filtered seg=%s text=%r", segment_id, text)
        await ws.send_json({
            "type": "transcript", "segmentId": segment_id, "turnId": state.turn_id,
            "text": "", "filtered": "hallucination", "filteredText": text,
            "sttMs": stt_ms, "totalMs": int((time.time() - t0) * 1000), "ts": time.time(),
        })
        _resume_debounce(ws, state)
        return

    # Plain-VAD barge-in — deferred to HERE (past STT + ghost filter) so that
    # hallucinated noise segments above can no longer cancel a running turn.
    # Empty transcripts drop before pending/dispatch anyway. The gated modes
    # cancel later still, once their gate passes.
    if text and not (wake_gating or speaker_gating):
        await _coalesce_cancel(state, ws)

    # --- Speaker lock (voice gate). Hard lock: only the enrolled owner voice is
    # transcribed; any other voice is dropped here, before the wake gate / LLM.
    # Voice only (typed input bypasses this path entirely).
    speaker_score = None
    speaker_trimmed = False
    speaker_full_text = None
    if speaker_gating and text:
        async def _reject(score):
            LOG.info("speaker: rejected seg=%s score=%.3f text=%r",
                     segment_id, score, text[:80])
            await _spk_dump(segment_id, pcm_bytes, score, "rej")
            await ws.send_json({
                "type": "transcript.ignored", "segmentId": segment_id,
                "turnId": state.turn_id, "text": text,
                "reason": "speaker_mismatch", "speakerScore": round(score, 4),
                "ts": time.time()})
            _resume_debounce(ws, state)

        # PTT is a deliberate button press — never hard-reject those segments.
        # The windowed rescue below still cleans foreign background out of them;
        # only the "drop everything" outcome turns into fail-open.
        fail_open = bool(ptt)

        # The commit gate ALWAYS re-verifies the full segment — even when the
        # early barge-in check confirmed the owner mid-stream. Prefix scores
        # are unreliable (a video prefix once hit 0.547 and its whole
        # transcript sailed through unchecked); a wrongly cancelled reply is
        # recoverable via continuity/coalescing, leaked foreign text is not.
        # PRIMARY decision: full-segment verification. Windowed scores are
        # deliberately NOT allowed to veto a segment — measured on real
        # speech, ~1 s windows of the SAME voice scatter far below the
        # full-segment score (a hard windowed gate rejected everything).
        # Windows only assist below, as a RESCUE for mixed segments.
        # (Started before STT — see verify_task above — so both overlap.)
        res = await verify_task
        speaker_score = round(res.score, 4)
        matched = res.matched
        # Temporal continuity: the owner was just heard (strict match) —
        # judge this segment relative to that score instead of the absolute
        # threshold. Fixes sentence tails split off by the owner-watch
        # scoring a hair under the slider while foreign stays far below.
        if (not matched and res.reason == "mismatch"
                and state.speaker_last_own > 0
                and time.time() - state.speaker_last_own_ts <= SPEAKER_CONT_WINDOW_S):
            # The anchor's FRESHNESS decides WHETHER continuity applies;
            # the bar itself is a fixed relaxation of the threshold. (The
            # earlier `anchor − Δ` formula backfired: the better the last
            # match, the LESS it helped — a 0.63 anchor left the bar at the
            # full threshold and a 0.495 follow-up died by 0.005.)
            eff = max(SPEAKER_CONT_FLOOR,
                      SPEAKER.threshold - SPEAKER_TRIM_REL_DELTA)
            if res.score >= eff:
                matched = True
                LOG.info("speaker: continuity accept seg=%s score=%.3f "
                         "(last own %.3f, bar %.2f)",
                         segment_id, res.score, state.speaker_last_own, eff)
        # Sentence-level short-pass: a short segment while the owner is
        # actively COMPOSING (input pending, no reply being generated) passes
        # regardless of its score — single words neither falsify the
        # transcript nor score reliably (also covers reason "too_short"), and
        # a false reject here lets the debounce fire mid-thought (premature
        # submit). Deliberately NOT while a reply is in flight (a foreign word
        # must not cancel it) and NOT standalone (a lone foreign word must not
        # start its own turn).
        if (not matched and duration_s < SPEAKER_SHORT_SEG_S
                and state.has_pending()
                and not (state.agent_task and not state.agent_task.done())):
            matched = True
            LOG.info("speaker: short segment passes while composing seg=%s "
                     "dur=%.1fs score=%.3f", segment_id, duration_s, res.score)
        if res.matched and res.reason == "match":
            # Only STRICT matches refresh the continuity anchor (a chain of
            # continuity accepts must not drift the reference downwards; a
            # fail-open verify — reason "error"/"no_profile", score 0.0 —
            # must not zero the anchor either).
            state.speaker_last_own = res.score
            state.speaker_last_own_ts = time.time()

        blocks = []
        if CFG.speaker_trim and duration_s >= SPEAKER_TRIM_MIN_S:
            # Equal-length 3 s block scores on a fixed grid — the ONLY scale on
            # which "who is this?" comparisons are valid (the model is heavily
            # length-sensitive; window/prefix scores misled every earlier
            # iteration of this gate).
            blocks = await asyncio.to_thread(SPEAKER.analyze_blocks, pcm_bytes, SAMPLE_RATE)
            if CFG.speaker_debug and blocks:
                LOG.info("spk-debug seg=%s full=%.3f matched=%s blocks=[%s]",
                         segment_id, res.score, matched,
                         " ".join(f"{s0:.1f}-{e0:.1f}:{sc:.2f}" for s0, e0, sc in blocks))

        async def _crop_and_restt(spans, span_total):
            """Crop to the given spans + re-transcribe; '' when unusable.
            The cropped audio goes through the ghost filter too (short
            snippets are prime Whisper-hallucination bait)."""
            nonlocal stt_ms
            cropped = audio_utils.crop_f32_spans(pcm_bytes, SAMPLE_RATE, spans)
            t_stt2 = time.time()
            try:
                t2 = (await STT.transcribe(cropped, SAMPLE_RATE) or "").strip()
            except Exception:
                LOG.exception("speaker trim: re-STT failed seg=%s", segment_id)
                t2 = ""
            stt_ms += int((time.time() - t_stt2) * 1000)
            if t2 and GHOST and GHOST.is_hallucination(
                    t2, no_speech_prob=getattr(STT, "last_no_speech_prob", None),
                    duration_s=span_total):
                t2 = ""
            if t2 and GHOST:
                t2 = GHOST.strip_ghost_sentences(t2)
            return t2

        async def _trim_foreign(best_hint, veto_bar=None):
            """Cut sustained foreign regions (block-relative) + re-transcribe
            the kept audio. Returns True when the text was replaced.
            ``veto_bar`` overrides the second-opinion bar (default: the full
            accept threshold)."""
            nonlocal text, speaker_score, speaker_trimmed, speaker_full_text
            foreign, keep = foreign_regions(
                blocks, duration_s, delta=SPEAKER_BLOCK_DELTA,
                min_region_s=SPEAKER_TRIM_MIN_FOREIGN_S,
                abs_floor=SPEAKER.threshold)
            if not foreign or not keep:
                return False
            # Second opinion before cutting: re-embed each candidate region as
            # ONE contiguous piece (contiguous multi-second audio behaves like
            # a full segment, unlike the noisy grid blocks). A region that
            # still scores like the owner is vetoed — field sessions showed
            # the relative rule cutting the owner's own sentences.
            bar = SPEAKER.threshold if veto_bar is None else veto_bar
            confirmed = []
            for a, b in foreign:
                sc = await asyncio.to_thread(
                    SPEAKER.score_region, pcm_bytes, SAMPLE_RATE, a, b)
                if sc >= bar:
                    LOG.info("speaker: trim veto seg=%s region=%.1f-%.1f "
                             "score=%.3f (owner)", segment_id, a, b, sc)
                else:
                    confirmed.append((a, b))
            if not confirmed:
                return False
            if len(confirmed) < len(foreign):
                foreign = confirmed
                keep = keep_regions(foreign, duration_s)
            keep_total = sum(e - s for s, e in keep)
            if keep_total < SPEAKER.min_dur_s:
                return False
            text2 = await _crop_and_restt(keep, keep_total)
            if not text2 or text2 == text:
                return False
            LOG.info("speaker: block trim seg=%s foreign=%s %r → %r",
                     segment_id,
                     ",".join(f"{a:.1f}-{b:.1f}" for a, b in foreign),
                     text[:60], text2[:60])
            speaker_full_text = text
            text = text2
            speaker_trimmed = True
            speaker_score = round(max(speaker_score or 0.0, best_hint), 4)
            return True

        if matched:
            # Owner confirmed — but a sustained foreign passage may still ride
            # along in the transcript. Blocks scoring far below the segment's
            # best SAME-LENGTH block are someone else; cut only whole
            # sentences (sentence-level policy), keep the text when in doubt.
            # Inside a MATCHED segment the veto uses the low FLOOR, not the
            # accept threshold: the doubt belongs to the owner here.
            if len(blocks) >= 2:
                await _trim_foreign(
                    max(sc for _, _, sc in blocks),
                    veto_bar=min(SPEAKER.threshold, SPEAKER_TRIM_VETO_FLOOR))
        else:
            rescued = False
            # Mixed segment where the foreign part dominates? → rescue the
            # owner's regions. Anchor: the best block must clear the floor
            # (owner present at all); regions are then judged relative to it.
            if len(blocks) >= 2:
                best = max(sc for _, _, sc in blocks)
                if best >= SPEAKER.window_threshold:
                    rescued = await _trim_foreign(best)
            if not rescued and not fail_open:
                await _reject(res.score)
                return
        await _spk_dump(segment_id, pcm_bytes, res.score,
                        "trim" if speaker_trimmed else "ok")
        # Owner confirmed → this segment may barge in. When the up-front cancel
        # was deferred purely for the speaker gate (pure VAD, no wake word), do
        # it now so the owner can interrupt a running reply. (No-op if the early
        # mid-stream check already cancelled it.) If the reply has not started
        # playing yet, the cancelled turn's input is carried over (coalescing).
        if not wake_gating:
            await _coalesce_cancel(state, ws)

    # --- Wake-word gate (prefix). Voice only; typed input is always intended.
    if wake_gating and text:
        now = time.time()
        # Manually closed + still running / guard → ignore the segment entirely.
        # NO wake match (no fuzzy matches either), NO cancel of the running
        # reply. This way nothing can reopen the window during processing.
        if _wake_closed_active(state, now):
            LOG.info("wake: closed → ignored seg=%s text=%r", segment_id, text)
            await ws.send_json({
                "type": "transcript.ignored", "segmentId": segment_id,
                "turnId": state.turn_id, "text": text,
                "reason": "wake_closed", "ts": time.time()})
            _resume_debounce(ws, state)
            return
        # Measure "window open?" at SPEECH START, not at commit time:
        # if someone starts speaking within the window, their (possibly long)
        # sentence may still pass even if the window lapses during speaking
        # or transcription (otherwise a race: input recognized but discarded).
        # speech_start_ts is already clock-skew corrected (server time);
        # fall back to now if no timestamp is available.
        ref_ts = speech_start_ts if speech_start_ts is not None else now
        if ref_ts < state.wake_until:
            # Conversation window was open at speech start → follow-up passes.
            command_text = text
        else:
            matched, remainder = wake.match_wake(
                text, CFG.wake_word, fuzzy=CFG.wake_word_fuzzy,
                anywhere=CFG.wake_word_anywhere, ratio=CFG.wake_word_ratio)
            if not matched:
                LOG.info("wake: ignored seg=%s text=%r", segment_id, text)
                await ws.send_json({
                    "type": "transcript.ignored", "segmentId": segment_id,
                    "turnId": state.turn_id, "text": text,
                    "reason": "no_wake_word", "ts": time.time()})
                _resume_debounce(ws, state)
                return
            # Wake detected → emit the early cue if needed (in case no partial
            # already sent it), strip the wake word.
            await _emit_wake_detected(ws, state, segment_id)
            command_text = remainder if CFG.wake_word_strip else text
            if not (command_text or "").strip():
                # Only the wake word was said → "armed", Antonia waits (idle) for
                # the command → idle timer runs on the client (reason=armed).
                await _open_wake_window(ws, state, now, reason="armed")
                await ws.send_json({
                    "type": "wake.armed", "turnId": state.turn_id,
                    "windowS": CFG.wake_word_window_s, "ts": time.time()})
                return

        # Stop command ("stop", "ok stopp", …) → stop the running reply and
        # close the conversation window (the wake word is needed again).
        if wake.is_stop_command(command_text):
            await _cancel_in_flight(state, ws)
            # "Stop" also discards whatever was composed but not yet
            # dispatched — otherwise it silently rides into the NEXT accepted
            # turn minutes later.
            state.pending_resume = ""
            state.pending_texts.clear()
            state.pending_segment_ids.clear()
            state.pending_text_parts.clear()
            state.pending_image_urls.clear()
            state.wake_until = 0.0
            state.wake_detected_seg = None
            LOG.info("wake: stop seg=%s → window closed", segment_id)
            await ws.send_json({
                "type": "wake.closed", "turnId": state.turn_id,
                "reason": "stop_command", "ts": time.time()})
            return

        # Command to the AI → refresh the window; a reply is coming, so the
        # idle timer does NOT run (reason=command). Then clear the running turn
        # — coalescing (not a plain cancel): a follow-up spoken before the
        # reply became audible must carry the committed input over, exactly
        # like the VAD/speaker-gated paths.
        await _open_wake_window(ws, state, now, reason="command")
        text = command_text
        await _coalesce_cancel(state, ws)

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
                LOG.exception("speaker-id failed for segment %s (ignored)", segment_id)

    transcript_evt = {
        "type": "transcript", "segmentId": segment_id, "turnId": state.turn_id,
        "text": text, "sttMs": stt_ms, "totalMs": int((time.time() - t0) * 1000),
        "ts": time.time(),
    }
    if speaker_info is not None:
        transcript_evt["speaker"] = speaker_info
    if speaker_score is not None:      # voice-lock: owner accepted → live match score
        transcript_evt["speakerScore"] = speaker_score
    if speaker_trimmed:                # voice-lock: foreign spans cut out of the text
        transcript_evt["speakerTrimmed"] = True
        transcript_evt["speakerFullText"] = speaker_full_text  # original mixed transcript
    await ws.send_json(transcript_evt)

    if not text:
        _resume_debounce(ws, state)
        return

    state.pending_texts.append(_speaker_tag(text, speaker_info))
    state.pending_segment_ids.append(segment_id)
    # E2E anchor: receive time of this (last) segment. With coalescing the
    # respective last segment wins → measurement from "user last spoke".
    state.speech_end_ts = t0
    # Debounce anchor: a VAD-committed segment arrives only AFTER the client
    # held ~0.8×debounce of silence (redemption) — credit that silence so the
    # total pause before dispatch stays ≈ debounce_ms instead of ~1.8× it.
    # PTT commits are a deliberate button release → no silence to credit.
    redemption_s = 0.0
    if not ptt:
        vp = vad_params_for_debounce(state.debounce_ms)
        redemption_s = vp["redemptionFrames"] * vp["frameMs"] / 1000.0
    state.debounce_anchor = t0 - redemption_s
    await _send_turn_pending(ws, state)
    # (Re)arm the debounce. Two segments can be in STT concurrently — both
    # would arm a timer and the SAME pending input would dispatch twice.
    # Cancel a ticking timer first; never touch a slot that already runs the
    # turn (the gates above own that decision — see _resume_debounce).
    if state.agent_task and not state.agent_task.done():
        return
    if state.debounce_task and not state.debounce_task.done():
        state.debounce_task.cancel()
    state.debounce_task = asyncio.create_task(_debounce_then_run(ws, state))


# ============================================================================ #
# B2: Streaming STT (interim transcripts while a segment is being streamed in)
# ============================================================================ #
async def _do_partial(ws, state: TurnState, seg: dict, pcm: bytes):
    """Transcribes the buffer accumulated so far and sends a
    transcript.partial — UI only, does not change any turn state."""
    id_at_start = seg.get("id")
    try:
        text = await STT.transcribe(pcm, SAMPLE_RATE)
    except Exception:
        LOG.exception("partial STT failed seg=%s", seg.get("id"))
        text = ""
    finally:
        seg["partial_running"] = False
    # Segment committed/aborted — or re-armed by the owner-watch under a new
    # virtual id (the audio this partial saw is no longer in the buffer)?
    # Then discard the partial.
    if seg.get("done") or seg.get("id") != id_at_start or not text:
        return
    seg["partial_text"] = text
    # Early cue: detect the wake word already in the growing partial (wake mode
    # only and only while the window is closed — with an open window no wake word
    # is needed). This way the user knows IMMEDIATELY that it triggered, instead
    # of only after finishing the utterance.
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
    """Throttle check + start of a partial transcription (B2). Sync — starts a
    task when needed, does not block the frame loop."""
    if not (CFG and CFG.stt_partial and STT is not None):
        return
    if seg.get("partial_running") or seg.get("done"):
        return
    buf_len = len(seg["buf"])
    now = time.time()
    # f32 = 4 bytes/sample.
    new_bytes = buf_len - seg.get("last_partial_len", 0)
    min_new = int(SAMPLE_RATE * 4 * (CFG.stt_partial_min_new_ms / 1000.0))
    if new_bytes < min_new:
        return
    if (now - seg.get("last_partial_ts", 0)) * 1000.0 < CFG.stt_partial_min_interval_ms:
        return
    seg["partial_running"] = True
    seg["last_partial_ts"] = now
    seg["last_partial_len"] = buf_len
    _spawn_tracked(state, _do_partial(ws, state, seg, bytes(seg["buf"])))


# ============================================================================ #
# Voice-lock barge-in: early speaker check on the streaming segment.
#
# With the speaker lock engaged the client does NOT stop playback on VAD speech
# (any voice would trigger that — the whole point of the lock is that foreign
# voices must not interrupt). Instead, while a segment streams in (B1) and a
# reply is in flight, the growing buffer is verified against the owner profile
# as soon as enough audio exists; only a CONFIRMED owner voice cancels the
# running reply. Foreign voices leave it playing.
# ============================================================================ #
SPEAKER_BARGE_MIN_S = 1.5      # audio needed before the first early check —
                               # short prefixes overscore (field: a video prefix
                               # hit 0.547 while full blocks of the SAME video
                               # scored 0.14–0.36)
SPEAKER_BARGE_MAX_CHECKS = 2   # one retry with 2× audio (first window may be echo-mixed)
SPEAKER_BARGE_MARGIN = 0.03    # interrupt needs threshold + margin (destructive)
# Segments at least this long get the windowed mixed-voice analysis (trimming)
# instead of one full-segment verify; shorter ones can't meaningfully mix.
# (1.6 s = two 1.2 s windows at 0.6 s hop — the minimum for a mixed decision.)
SPEAKER_TRIM_MIN_S = 1.6
# Relative trim: absolute window scores shift wildly with recording conditions,
# but the relative ORDER within one segment is stable — windows scoring this far
# below the segment's own best window are someone else. Field-calibrated from
# spk-debug data: owner windows ~0.24–0.33 vs foreign ~0.09–0.21 in the same
# segment → a 0.12 gap separates them; 0.30 (the first guess) never fired.
SPEAKER_TRIM_REL_DELTA = 0.12
# Sentence-level policy: only SUSTAINED foreign speech (a train announcement,
# a kid telling a story) is worth cutting/rejecting — a single word neither
# falsifies the transcript nor deserves the risk: a false cut mangles the
# owner's sentence, a false reject triggers premature submits. Short-audio
# embeddings are erratic anyway (measured: same-voice 1.2 s clips score
# 0.19–0.31 vs 0.91 for the full sentence; tiling the clip inflates OWN and
# FOREIGN scores alike, so it cannot rescue short-clip scoring).
SPEAKER_TRIM_MIN_FOREIGN_S = 2.5   # minimum foreign block worth cutting
SPEAKER_SHORT_SEG_S = 2.5          # shorter segments pass mid-conversation
# Relative bar between EQUAL-LENGTH 3 s blocks of the same segment (the only
# scale on which speaker comparisons are valid — see analyze_blocks).
SPEAKER_BLOCK_DELTA = 0.15
# Second-opinion veto floor for cutting inside a MATCHED segment: the cut
# candidate's contiguous re-score must look genuinely foreign before it is
# cut. Field data (2026-07-06): real foreign passages re-score 0.05–0.15,
# the owner's quiet sentence tails 0.30–0.34 — the accept threshold (0.4)
# sat between the two and let the owner's own sentences be cut. The RESCUE
# path (full verify failed) keeps the strict threshold: there the segment
# is NOT confirmed as the owner, so the doubt goes toward cutting.
SPEAKER_TRIM_VETO_FLOOR = 0.25
# Temporal continuity: after a strict owner match, follow-up segments within
# this window are judged RELATIVE to that score (last_own − Δ, floored) —
# sentence tails after an owner-watch split score slightly lower than the
# head, while foreign voices stay far below.
SPEAKER_CONT_WINDOW_S = 30.0
SPEAKER_CONT_FLOOR = 0.30


def _spk_dump_sync(dump_dir, tag, pcm_bytes, score, decision, keep=200):
    """Write a gated segment / enrollment take as a 16-bit WAV into
    SPEAKER_DUMP_DIR so gate decisions can be replayed offline against the
    verifier with the REAL audio. Filename: <ts>_<tag>_<decision>_<score>.wav;
    only the newest `keep` files survive (disk cap). Blocking — call via
    asyncio.to_thread."""
    d = Path(dump_dir)
    d.mkdir(parents=True, exist_ok=True)
    now = time.time()
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(tag))
    name = f"{ts}-{int(now % 1 * 1000):03d}_{safe}_{decision}_{score:.3f}.wav"
    arr = audio_utils.pcm_bytes_to_float32_array(pcm_bytes)
    (d / name).write_bytes(audio_utils.float32_to_pcm16_wav_bytes(arr, SAMPLE_RATE))
    for p in sorted(d.glob("*.wav"))[:-keep]:
        try:
            p.unlink()
        except OSError:
            pass


async def _spk_dump(tag, pcm_bytes, score, decision):
    if not (CFG and CFG.speaker_dump_dir):
        return
    try:
        await asyncio.to_thread(_spk_dump_sync, CFG.speaker_dump_dir, tag,
                                pcm_bytes, score, decision)
    except Exception:
        LOG.exception("speaker dump failed (%s)", tag)


async def _do_speaker_barge(ws, state: TurnState, seg: dict, pcm: bytes):
    try:
        duration_s = (len(pcm) // 4) / SAMPLE_RATE
        res = await asyncio.to_thread(SPEAKER.verify, pcm, SAMPLE_RATE, duration_s)
        matched, score = res.matched, res.score
        # Interrupting is destructive AND prefix scores are flaky → demand a
        # margin above the plain accept threshold. A false positive here only
        # cancels the reply; the commit gate re-verifies the full segment.
        if matched and score < SPEAKER.threshold + SPEAKER_BARGE_MARGIN:
            matched = False
        if not matched:
            # Mixed prefix (kids talking over/next to the owner) dilutes the
            # full-buffer score. Fall back to windows: a single CLEAR owner
            # window above the margined bar suffices.
            wa = await asyncio.to_thread(SPEAKER.analyze_windows, pcm, SAMPLE_RATE)
            if wa is not None and wa.score >= SPEAKER.threshold + SPEAKER_BARGE_MARGIN:
                matched, score = True, wa.score
    except Exception:
        LOG.exception("speaker barge check failed seg=%s", seg.get("id"))
        matched, score = None, 0.0
    finally:
        seg["spk_running"] = False
    # Segment already committed/aborted → the commit path decides, not us.
    if matched is None or seg.get("done"):
        return
    if matched:
        seg["spk_owner"] = True
        seg["spk_score"] = round(score, 4)
        LOG.info("speaker: owner confirmed mid-stream seg=%s score=%.3f → barge-in",
                 seg.get("id"), score)
        await _coalesce_cancel(state, ws)
    else:
        LOG.debug("speaker: mid-stream check not owner seg=%s score=%.3f (reply keeps playing)",
                  seg.get("id"), score)


def _maybe_spawn_speaker_barge(ws, state: TurnState, seg: dict):
    """Throttle check + start of an early speaker verification (see above).
    Sync — spawns a task when needed, does not block the frame loop.

    Wake mode is deliberately excluded: there the wake gate at commit decides
    what may interrupt (deferred cancel), and an open-window follow-up already
    passes both gates at commit."""
    if SPEAKER is None or not SPEAKER.active():
        return
    if not CFG or getattr(state, "wake_word_enabled", False):
        return
    if seg.get("spk_owner") or seg.get("spk_running") or seg.get("done"):
        return
    if seg.get("spk_checks", 0) >= SPEAKER_BARGE_MAX_CHECKS:
        return
    # Only worth checking early when there is something to interrupt — which
    # includes audio the client is still PLAYING from its buffer long after
    # all chunks were sent (fast TTS: audio_ids empties within ~1-2 s).
    in_flight = ((state.agent_task and not state.agent_task.done())
                 or bool(state.audio_ids) or state.client_playing)
    if not in_flight:
        return
    min_s = max(getattr(SPEAKER, "min_dur_s", 0.0), SPEAKER_BARGE_MIN_S)
    need = int(SAMPLE_RATE * 4 * min_s) * (seg.get("spk_checks", 0) + 1)
    if len(seg["buf"]) < need:
        return
    seg["spk_checks"] = seg.get("spk_checks", 0) + 1
    seg["spk_running"] = True
    _spawn_tracked(state, _do_speaker_barge(ws, state, seg, bytes(seg["buf"])))


# ============================================================================ #
# Owner-watch: server-side end-of-owner detection (segmentation fix).
#
# The browser VAD is speaker-agnostic: it closes a segment on SILENCE, not when
# the OWNER stops. If other voices keep talking (kids), the client segment never
# ends and the owner's utterance would sit in the buffer indefinitely. With the
# speaker lock engaged, the server therefore scores the streaming buffer in
# ~1.2 s windows; once the owner HAS spoken and then hasn't been heard for the
# debounce time, the buffered audio is committed as the owner's utterance and
# the stream is re-armed as a fresh virtual segment — the foreign tail is then
# judged on its own at the real client commit (and rejected by the gate).
# ============================================================================ #
OWNER_WATCH_WIN_S = 1.2    # window fed to the embedding model
OWNER_WATCH_HOP_S = 0.6    # step between scored windows


def _maybe_spawn_owner_watch(ws, state: TurnState, seg: dict, peer):
    """Throttle check + start of one owner-watch window score. Sync — spawns a
    task when a full new window of audio is available."""
    if SPEAKER is None or not SPEAKER.active() or not CFG:
        return
    if seg.get("done") or seg.get("own_running"):
        return
    win = int(SAMPLE_RATE * 4 * OWNER_WATCH_WIN_S)
    hop = int(SAMPLE_RATE * 4 * OWNER_WATCH_HOP_S)
    next_off = seg.get("own_next_off", 0)
    if len(seg["buf"]) < next_off + win:
        return
    seg["own_running"] = True
    pcm = bytes(seg["buf"][next_off:next_off + win])
    seg["own_next_off"] = next_off + hop
    _spawn_tracked(state, _owner_watch_step(ws, state, seg, pcm, next_off + win, peer))


async def _owner_watch_step(ws, state: TurnState, seg: dict, pcm: bytes,
                            end_off: int, peer):
  # The own_running guard spans the WHOLE step including the veto embeds —
  # otherwise a parallel step could double-commit the same buffer.
  try:
    try:
        res = await asyncio.to_thread(SPEAKER.window_is_owner, pcm, SAMPLE_RATE)
    except Exception:
        LOG.exception("owner-watch failed seg=%s", seg.get("id"))
        res = None
    if seg.get("done"):
        return
    if res is True:
        seg["own_seen"] = True
        seg["own_last_end"] = end_off / (SAMPLE_RATE * 4.0)
    # Only cut once the owner HAS spoken in this segment and has since been
    # quiet (foreign voices / noise) for the user's pause tolerance.
    if not seg.get("own_seen"):
        return
    buf_s = len(seg["buf"]) / (SAMPLE_RATE * 4.0)
    quiet_s = buf_s - seg.get("own_last_end", 0.0)
    # Deliberately LAZIER than the debounce: true silence closes the segment
    # client-side anyway (VAD redemption ≈ 0.8×debounce) — the watch only has
    # to cut when FOREIGN sound keeps the mic open. A tight bound here cut the
    # owner mid-sentence whenever one trailing word scored foreign.
    if quiet_s < max(2.0, state.debounce_ms / 1000.0 + 0.8):
        return
    # Block-level VETO before cutting: single window scores routinely dip
    # below the bar mid-sentence (field logs: a cut fired after 2.4 s while
    # the owner was still talking). Confirm with reliable block embeddings —
    # head [0, cut] vs tail [cut, end]; cut only when the tail is clearly NOT
    # the head's speaker (same relative rule as the trim logic).
    cut_s = seg.get("own_last_end", 0.0)
    pcm_all = bytes(seg["buf"])
    head_sc = await asyncio.to_thread(SPEAKER.score_region, pcm_all, SAMPLE_RATE, 0.0, cut_s)
    tail_sc = await asyncio.to_thread(SPEAKER.score_region, pcm_all, SAMPLE_RATE, cut_s, None)
    if seg.get("done"):
        return
    if head_sc < SPEAKER.window_threshold or tail_sc >= head_sc - SPEAKER_TRIM_REL_DELTA:
        # Same speaker trailing on (or head not owner-ish) → no split. Count
        # the tail as owner activity so the veto doesn't re-run every hop.
        seg["own_last_end"] = len(seg["buf"]) / (SAMPLE_RATE * 4.0)
        if CFG and CFG.speaker_debug:
            LOG.info("owner-watch veto seg=%s head=%.3f tail=%.3f (no cut)",
                     seg.get("id"), head_sc, tail_sc)
        return
    buf_s = len(seg["buf"]) / (SAMPLE_RATE * 4.0)   # may have grown during embeds
    pcm_all = bytes(seg["buf"])
    # Commit ONLY the owner's head [0, cut+pad]: the block veto just confirmed
    # the tail is a DIFFERENT speaker, and at ~2-2.6 s it is too short for the
    # commit gate's trim (min foreign region 2.5 s) — committing the whole
    # buffer leaked exactly the foreign words the watch exists to keep out.
    cut_bytes = min(len(pcm_all),
                    int((cut_s + 0.25) * SAMPLE_RATE) * 4)
    pcm_head = pcm_all[:cut_bytes]
    seg_id = seg["id"]
    LOG.info("owner-watch: auto-commit seg=%s (%.1fs of %.1fs audio, owner "
             "quiet %.1fs, head=%.2f tail=%.2f)", seg_id,
             len(pcm_head) / (SAMPLE_RATE * 4.0), buf_s, quiet_s, head_sc, tail_sc)
    speech_start = seg.get("speech_start_ts")
    barge = seg.get("barge_in", False)
    # Re-arm the stream as a FRESH virtual segment seeded with the tail: the
    # client keeps streaming; whatever follows (the kids' tail, or the owner
    # speaking again — possibly already begun inside the tail) is judged on
    # its own at the real commit / the next auto-commit.
    seg["own_cuts"] = seg.get("own_cuts", 0) + 1
    seg["id"] = f"{seg_id}-c{seg['own_cuts']}"
    seg["buf"] = bytearray(pcm_all[cut_bytes:])
    seg["speech_start_ts"] = None
    seg["barge_in"] = False
    seg["last_partial_len"] = 0
    seg["last_partial_ts"] = 0.0
    seg["partial_text"] = ""
    seg["spk_checks"] = 0
    seg["spk_owner"] = False
    seg["spk_score"] = None
    seg["own_next_off"] = 0
    seg["own_seen"] = False
    seg["own_last_end"] = 0.0
    _spawn_tracked(state, _handle_audio_segment(
        ws, state, pcm_head, seg_id, peer,
        speech_start_ts=speech_start, barge_in=barge))
  finally:
    seg["own_running"] = False


# ============================================================================ #
# WebSocket handler
# ============================================================================ #
async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=WS_HEARTBEAT_S, max_msg_size=WS_MAX_MSG_BYTES)
    await ws.prepare(request)
    peer = request.remote
    LOG.info("ws connect: %s", peer)

    state = TurnState()
    state.session_user = _default_user()
    state.speed = CFG.tts_speed if CFG else 1.0
    state.debounce_ms = CFG.debounce_ms if CFG else 1200
    # Start default of the wake mode; the client toggles it via 'settings'.
    state.wake_word_enabled = bool(CFG.wake_word_enabled) if CFG else False
    # Bounded: pairs a `segment.start` with the next binary frame (FIFO, normally
    # depth 1). The cap stops a client that sends starts without frames from
    # growing it without limit.
    pending_segment_meta: deque = deque(maxlen=64)
    # B1: streamed input segment (frames arrive while speaking). None = no active
    # stream segment → binary frames are full segments.
    active_stream: dict | None = None
    # Speaker-lock enrollment: while set, incoming binary frames are the owner's
    # enrollment recording (NOT a segment) and are buffered here until commit.
    enroll_stream: bytearray | None = None
    # Voice-clone recording: same idea as enroll_stream — while set, binary frames
    # are the reference sample for a new cloned voice, buffered until commit.
    clone_stream: bytearray | None = None

    if BRIDGE is not None:
        BRIDGE.register_broadcast(ws)
    WS_CLIENTS[ws] = state
    # A browser just connected → speech is likely soon; warm the STT endpoint
    # now so a cold remote model doesn't stall the first real segment.
    _maybe_warmup_stt()

    clone_caps = await voice_clone.voice_clone_hello()
    await ws.send_json({
        "type": "hello", "stage": STAGE,
        "msg": f"Server ready – agent: {CFG.agent_name}.",
        "agent_name": CFG.agent_name,
        "lang": (CFG.app_language if CFG else "en"),
        "basePath": (CFG.base_path if CFG else ""),
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
        # Opus capability: opusIn = server decodes an opus mic uplink,
        # opusOut = server can encode the TTS downlink (client opts in via
        # settings.audioCodec). Both false when AUDIO_OPUS=0 or libopus is
        # missing — the client then keeps the raw PCM paths.
        "audio": {"opusIn": _opus_active(), "opusOut": _opus_active()},
        "wakeWord": {
            "available": True,             # wake mode is always selectable (UI)
            "enabled": bool(CFG.wake_word_enabled) if CFG else False,  # start default
            "word": CFG.wake_word if CFG else "",
            "windowS": CFG.wake_word_window_s if CFG else 0,
            "mode": _wake_mode(),          # "conversation" | "alexa" (one-shot)
        },
        "speakerLock": {
            # available = model loaded (feature usable). enrolled = a profile
            # exists (gate actually filtering). Both false when the feature is off.
            "available": bool(SPEAKER is not None and SPEAKER.loaded),
            "enrolled": bool(SPEAKER is not None and SPEAKER.has_profile()),
            "count": (SPEAKER._count if SPEAKER is not None else 0),
            "threshold": (SPEAKER.threshold if SPEAKER is not None else 0),
            # Runtime toggle state (the client re-asserts its own on connect).
            "enabled": bool(SPEAKER is not None and getattr(SPEAKER, "enabled", True)),
        },
        # Voice library: available = cloning wired (wrapper behind TTS). Carries
        # the current voice list + active id so the client renders immediately.
        "voiceClone": clone_caps,
        "turn": {
            "debounce_ms": state.debounce_ms,
            "debounce_ms_min": CFG.debounce_ms_min, "debounce_ms_max": CFG.debounce_ms_max,
            "vad": vad_params_for_debounce(state.debounce_ms),
        },
        "hermesMode": "separate",
        "hermesAvailable": False,  # toggle removed; mode fixed to 'separate'
        "ts": time.time(),
    })

    # --- Load conversation history from Hermes backend (cross-device) ------
    # Only when a Hermes session key is actually configured: without it the
    # session id would be probed against a NON-Hermes endpoint (e.g. Fireworks)
    # — a real, authenticated HTTP call per connect for nothing.
    if CFG and CONV is not None and CFG.hermes_session_key_separate:
        try:
            history = await fetch_history(
                base_url=CFG.llm_base_url,
                api_key=CFG.llm_api_key,
                session_id=state.hermes_session_id_separate,
                max_messages=CFG.llm_history_turns * 2,
            )
            if history:
                CONV.seed_history(state.session_user, history)
                # Send displayable history to the client (user/assistant only).
                await ws.send_json({
                    "type": "history",
                    "messages": history,
                    "sessionId": state.hermes_session_id_separate[:16],
                    "ts": time.time(),
                })
        except Exception as exc:
            LOG.warning("history load failed for %s: %s", peer, exc)

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
                    # Client finished playing its buffered reply audio → there
                    # is nothing left to barge into.
                    state.client_playing = False
                    # Wake word: reply fully spoken → (re)open the conversation
                    # window; only now does the idle timer run.
                    # Exception: the user closed the channel themselves in the
                    # meantime → do not reopen (reset the flag once).
                    if CFG and state.wake_word_enabled:
                        if state.wake_suppress_reopen:
                            # Manually closed during the reply: do not reopen.
                            # Short trailing guard, so the echo right after the
                            # reply doesn't reopen the window after all.
                            state.wake_suppress_reopen = False
                            state.wake_closed_until = max(
                                state.wake_closed_until, time.time() + WAKE_CLOSE_GUARD_S)
                        elif _wake_oneshot():
                            # Alexa mode: close the window immediately after the
                            # reply, a new command needs the wake word again.
                            state.wake_until = 0.0
                            state.wake_detected_seg = None
                            await ws.send_json({
                                "type": "wake.closed", "turnId": state.turn_id,
                                "reason": "oneshot", "ts": time.time()})
                        else:
                            await _open_wake_window(ws, state, reason="done")
                    # Voice lock, temporal continuity: restart the follow-up
                    # window at the END of the reply playback (same "done"
                    # pattern as the wake window above). Anchored at the
                    # user's segment the window was always expired by the
                    # time they could answer — LLM + TTS + playback routinely
                    # exceed SPEAKER_CONT_WINDOW_S. Only the timestamp moves;
                    # the SCORE anchor still requires a strict match.
                    if (SPEAKER is not None and SPEAKER.active()
                            and state.speaker_last_own > 0):
                        state.speaker_last_own_ts = time.time()
                    ident = get_speaker_identifier()
                    if ident is not None:
                        try:
                            ident.mark_playback_finished()
                        except Exception:
                            LOG.exception("mark_playback_finished failed (ignored)")
                elif t == "barge_in":
                    reason = (data.get("reason") or "speech")
                    # Voice lock: an unverified voice must not interrupt. VAD-
                    # triggered barge-ins are ignored — the interruption happens
                    # server-side once the streamed segment's speaker is confirmed
                    # as the owner (_do_speaker_barge). Defense in depth: current
                    # clients don't even send these while the lock is engaged.
                    # Deliberate user actions (stop button, PTT press) still pass.
                    if (SPEAKER is not None and SPEAKER.active()
                            and reason not in ("manual", "ptt-press")):
                        LOG.info("barge-in ignored (speaker lock) from %s (reason=%s)",
                                 peer, reason)
                    # Wake mode + manually closed/guard: the speech is NOT directed
                    # at Antonia → do NOT cancel the running reply.
                    elif CFG and state.wake_word_enabled and _wake_closed_active(state):
                        LOG.info("barge-in ignored (wake closed) from %s (reason=%s)", peer, reason)
                    else:
                        in_flight = ((state.agent_task and not state.agent_task.done())
                                     or bool(state.audio_ids) or state.client_playing)
                        if in_flight:
                            LOG.info("barge-in from %s (reason=%s)", peer, reason)
                            await _cancel_in_flight(state, ws)
                        # Whoever interrupts a RUNNING reply (server-side in-flight
                        # OR still playing client-side) continues the conversation
                        # → keep the window open, so the following (interrupting)
                        # input isn't discarded as "no wake word". NOT for mere
                        # noise without a running reply (otherwise no gate ever again).
                        if CFG and state.wake_word_enabled and (in_flight or bool(data.get("playing"))):
                            await _open_wake_window(ws, state, reason="command")
                elif t == "wake.close":
                    # Close the voice channel (conversation window) manually, WITHOUT
                    # cancelling running processing (that's what 'barge_in' does).
                    # After this the wake word is needed again.
                    if CFG and state.wake_word_enabled:
                        in_flight = (state.agent_task and not state.agent_task.done()) or bool(state.audio_ids)
                        state.wake_until = 0.0
                        state.wake_detected_seg = None
                        # Short guard: trailing partials/echo must not reopen the
                        # window right now.
                        state.wake_closed_until = time.time() + WAKE_CLOSE_GUARD_S
                        # If a reply is still running (server-side in_flight OR the
                        # client is still playing the tail), a playback.done will
                        # follow that would otherwise reopen the window → suppress it.
                        state.wake_suppress_reopen = bool(in_flight or data.get("playing"))
                        LOG.info("wake: manually closed (%s, in_flight=%s)", peer, in_flight)
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
                        "id": data.get("segmentId") or _short_id(),
                        "speech_start_ts": speech_start_ts,
                        "barge_in": bool(data.get("bargeIn")),
                        "ptt": bool(data.get("ptt")),
                    })
                elif t == "segment.stream.start":
                    # While the owner is recording an enrollment or voice-clone
                    # take, VAD segments are ignored entirely — their frames
                    # would race with the recording frames for the binary channel.
                    if enroll_stream is not None or clone_stream is not None:
                        continue
                    # B1: start of a streamed segment. Following binary frames
                    # (raw f32le) are appended until segment.stream.commit arrives.
                    _sst = data.get("speechStartTs")
                    _cnow = data.get("clientNow")
                    speech_start_ts = None
                    if isinstance(_sst, (int, float)) and isinstance(_cnow, (int, float)):
                        speech_start_ts = time.time() - (float(_cnow) - float(_sst)) / 1000.0
                    # Uplink codec: codec:"opus" → binary frames are framed opus
                    # packets, decoded on arrival to 16 kHz f32 so every
                    # downstream consumer (partials, speaker gates, owner-watch,
                    # commit STT) keeps seeing plain PCM. Defense in depth: a
                    # client should only request opus when hello advertised it;
                    # if the codec is unusable anyway, the segment is dropped
                    # (frames discarded, commit becomes a no-op) and the client
                    # is told via transcript.error so its pending-STT UI clears.
                    _codec = str(data.get("codec") or "pcm").lower()
                    _opus_dec = None
                    if _codec == "opus":
                        if _opus_active():
                            try:
                                _opus_dec = opus_codec.OpusDecoder(SAMPLE_RATE)
                            except Exception:
                                LOG.exception("opus decoder init failed")
                        if _opus_dec is None:
                            LOG.warning("segment %s requested opus but codec is "
                                        "unavailable — segment dropped",
                                        data.get("segmentId"))
                            await ws.send_json({
                                "type": "transcript.error",
                                "segmentId": data.get("segmentId"),
                                "error": "opus codec unavailable on server",
                                "ts": time.time()})
                    active_stream = {
                        "id": data.get("segmentId") or _short_id(),
                        "buf": bytearray(),
                        "codec": _codec, "opus_dec": _opus_dec,
                        "speech_start_ts": speech_start_ts,
                        "barge_in": bool(data.get("bargeIn")),
                        "ptt": bool(data.get("ptt")),
                        # B2: partial throttle state.
                        "partial_running": False, "last_partial_ts": 0.0,
                        "last_partial_len": 0, "partial_text": "", "done": False,
                        # Voice-lock early barge-in state (_maybe_spawn_speaker_barge).
                        "spk_checks": 0, "spk_running": False,
                        "spk_owner": False, "spk_score": None,
                        # Owner-watch state (_maybe_spawn_owner_watch).
                        "own_next_off": 0, "own_running": False,
                        "own_seen": False, "own_last_end": 0.0, "own_cuts": 0,
                    }
                elif t == "segment.stream.commit":
                    if active_stream is not None:
                        _seg = active_stream
                        _seg["done"] = True       # suppress late partials
                        active_stream = None
                        pcm = bytes(_seg["buf"])
                        if pcm:
                            _spawn_tracked(state, _handle_audio_segment(
                                ws, state, pcm, _seg["id"], peer,
                                speech_start_ts=_seg.get("speech_start_ts"),
                                barge_in=_seg.get("barge_in", False),
                                ptt=_seg.get("ptt", False)))
                elif t == "segment.stream.abort":
                    if active_stream is not None:
                        active_stream["done"] = True
                    active_stream = None
                elif t == "speaker.enroll.start":
                    # Begin buffering the owner's enrollment recording. Drop any
                    # half-streamed segment so its frames don't leak into it,
                    # and any queued segment.start metas — they would pair with
                    # the wrong binary frame after enrollment (FIFO desync).
                    if active_stream is not None:
                        active_stream["done"] = True
                        active_stream = None
                    pending_segment_meta.clear()
                    enroll_stream = bytearray() if (SPEAKER and SPEAKER.loaded) else None
                    if SPEAKER is None or not SPEAKER.loaded:
                        await ws.send_json({"type": "speaker.enroll.ack", "ok": False,
                                            "error": "unavailable", "ts": time.time()})
                elif t == "speaker.enroll.commit":
                    buf = bytes(enroll_stream) if enroll_stream is not None else b""
                    enroll_stream = None
                    if not (SPEAKER and SPEAKER.loaded):
                        await ws.send_json({"type": "speaker.enroll.ack", "ok": False,
                                            "error": "unavailable", "ts": time.time()})
                    elif len(buf) < int(SAMPLE_RATE * 4 * SPEAKER.min_dur_s):
                        await ws.send_json({"type": "speaker.enroll.ack", "ok": False,
                                            "error": "too_short", "ts": time.time()})
                    else:
                        try:
                            status = await asyncio.to_thread(SPEAKER.enroll, buf, SAMPLE_RATE)
                            LOG.info("speaker: enrolled take (count=%d, sampleScore=%s)",
                                     status.get("count"), status.get("sampleScore"))
                            await _spk_dump(f"enroll{status.get('count', 0)}", buf,
                                            float(status.get("sampleScore") or 0.0),
                                            "enroll")
                            await ws.send_json({"type": "speaker.enroll.ack", "ok": True,
                                                **status, "ts": time.time()})
                        except Exception as exc:
                            LOG.exception("speaker enroll failed")
                            await ws.send_json({"type": "speaker.enroll.ack", "ok": False,
                                                "error": str(exc), "ts": time.time()})
                elif t == "speaker.enroll.abort":
                    enroll_stream = None
                elif t == "speaker.enroll.clear":
                    if SPEAKER is not None:
                        SPEAKER.clear_profile()
                    await ws.send_json({
                        "type": "speaker.status",
                        "available": bool(SPEAKER is not None and SPEAKER.loaded),
                        "enrolled": bool(SPEAKER is not None and SPEAKER.has_profile()),
                        "count": (SPEAKER._count if SPEAKER is not None else 0),
                        "ts": time.time()})
                elif t == "voice.clone.start":
                    # Begin buffering a voice-clone reference recording. Same
                    # channel discipline as enrollment: drop any half-streamed
                    # segment + queued metas so their frames can't leak in.
                    if not voice_clone.clone_active():
                        await ws.send_json({"type": "voice.clone.ack", "ok": False,
                                            "error": "unavailable", "ts": time.time()})
                    else:
                        if active_stream is not None:
                            active_stream["done"] = True
                            active_stream = None
                        pending_segment_meta.clear()
                        clone_stream = bytearray()
                elif t == "voice.clone.commit":
                    buf = bytes(clone_stream) if clone_stream is not None else b""
                    clone_stream = None
                    name = (data.get("name") or "").strip()
                    if not voice_clone.clone_active():
                        await ws.send_json({"type": "voice.clone.ack", "ok": False,
                                            "error": "unavailable", "ts": time.time()})
                    else:
                        ack = await voice_clone.clone_commit(buf, name)
                        await ws.send_json({"type": "voice.clone.ack", **ack,
                                            "ts": time.time()})
                        if ack.get("ok"):
                            await voice_clone.emit_voice_state(ws)
                elif t == "voice.clone.abort":
                    clone_stream = None
                elif t == "voice.list":
                    if voice_clone.clone_active():
                        await voice_clone.emit_voice_state(ws)
                elif t == "voice.select":
                    if voice_clone.clone_active():
                        VOICES.set_active(str(data.get("id") or "").strip())
                        await voice_clone.emit_voice_state(ws)
                elif t == "voice.rename":
                    if voice_clone.clone_active():
                        try:
                            await VOICES.rename(str(data.get("id") or ""),
                                                str(data.get("name") or "").strip())
                            await voice_clone.emit_voice_state(ws)
                        except Exception as exc:  # noqa: BLE001
                            await ws.send_json({"type": "voice.error", "op": "rename",
                                                "error": str(exc), "ts": time.time()})
                elif t == "voice.delete":
                    if voice_clone.clone_active():
                        vid = str(data.get("id") or "")
                        try:
                            await VOICES.delete(vid)
                            # If the deleted voice was active, fall back to default.
                            if VOICES.get_active() == vid:
                                VOICES.set_active(voices_mod.DEFAULT_VOICE_ID)
                            await voice_clone.emit_voice_state(ws)
                        except Exception as exc:  # noqa: BLE001
                            await ws.send_json({"type": "voice.error", "op": "delete",
                                                "error": str(exc), "ts": time.time()})
                elif t == "voice.preview":
                    if voice_clone.clone_active():
                        vid = str(data.get("id") or voices_mod.DEFAULT_VOICE_ID)
                        ptext = (data.get("text") or voice_clone.preview_sentence()).strip()
                        try:
                            pcm, sr = await TTS.synth(ptext, speed=state.speed, voice=vid)
                            wav = await asyncio.to_thread(
                                audio_utils.pcm16_to_wav_bytes, pcm, sr)
                            await ws.send_json({
                                "type": "voice.preview.audio", "id": vid, "mime": "audio/wav",
                                "audioB64": base64.b64encode(wav).decode("ascii"),
                                "ts": time.time()})
                        except Exception as exc:  # noqa: BLE001
                            LOG.exception("voice preview failed")
                            await ws.send_json({"type": "voice.error", "op": "preview",
                                                "error": str(exc), "ts": time.time()})
                elif t == "text.message":
                    text_val = (data.get("text") or "").strip()
                    img_urls = data.get("imageUrls") or []
                    if not isinstance(img_urls, list):
                        img_urls = []
                    img_urls = [str(u) for u in img_urls if u]
                    if not text_val and not img_urls:
                        continue
                    if state.agent_task and not state.agent_task.done():
                        await _coalesce_cancel(state, ws)
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
                    state.pending_resume = ""
                    state.pending_texts.clear()
                    state.pending_segment_ids.clear()
                    # Typed text / attached images composed before the reset
                    # must not ride into the fresh session either.
                    state.pending_text_parts.clear()
                    state.pending_image_urls.clear()
                    state.speech_end_ts = 0.0
                    pending_segment_meta.clear()
                    if active_stream is not None:
                        active_stream["done"] = True
                    active_stream = None
                    state.turn_id = _short_id()
                    _ident = get_speaker_identifier()
                    if _ident is not None:
                        try:
                            _ident.reset()
                        except Exception:
                            LOG.exception("speaker identifier reset failed (ignored)")
                    new_user = f"{_default_user()}-{_short_id()}"
                    state.session_user = new_user
                    if CONV is not None:
                        CONV.reset(new_user)
                    # Rotate the Hermes session ID so X-Hermes-Session-Id
                    # points to a fresh conversation thread (new voice session).
                    # Persisted server-side: without that, every reconnect /
                    # page reload would re-derive the stable pre-reset ID and
                    # silently continue the OLD Hermes session.
                    state.hermes_session_id_separate = rotate_hermes_session_id()
                    LOG.info("session reset for %s: user=%s", peer, new_user)
                    await ws.send_json({
                        "type": "session.reset.ack", "sessionUser": new_user,
                        "sessionKey": _session_key_for_user(new_user),
                        "hermesMode": "separate",
                        "sharedWithTelegram": False, "ts": time.time(),
                    })
                    # The session is shared across devices → the reset applies
                    # everywhere at once: cancel other connections' in-flight
                    # turns, move them onto the fresh CONV user key + session
                    # ID and tell their UIs to clear.
                    for peer_ws, peer_state in list(WS_CLIENTS.items()):
                        if peer_ws is ws:
                            continue
                        try:
                            await _cancel_in_flight(peer_state, peer_ws)
                        except Exception:
                            LOG.exception("peer cancel on session reset failed (ignored)")
                        peer_state.session_user = new_user
                        peer_state.hermes_session_id_separate = state.hermes_session_id_separate
                    await _broadcast_peers(ws, {"type": "session.reset.remote",
                                                "ts": time.time()})
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
                        # Leaving wake mode → close an open conversation window.
                        if was_on and not state.wake_word_enabled:
                            state.wake_until = 0.0
                    if "speakerThreshold" in data and SPEAKER is not None:
                        # Voice-lock strictness, tunable live from the UI (process-
                        # wide on the shared verifier; fine for a single-owner setup).
                        try:
                            SPEAKER.threshold = max(0.2, min(0.95, float(data["speakerThreshold"])))
                        except (TypeError, ValueError):
                            pass
                    if "speakerLockEnabled" in data and SPEAKER is not None:
                        # Temporary on/off of the whole voice-lock gate (profile kept).
                        SPEAKER.enabled = bool(data["speakerLockEnabled"])
                    if "audioCodec" in data:
                        # Downlink codec request. Only honor "opus" when the
                        # codec is actually usable — otherwise fall back to raw
                        # PCM and say so in the ack (the client adapts).
                        want = str(data.get("audioCodec") or "").lower()
                        state.audio_codec = ("opus" if want == "opus"
                                             and _opus_active() else "pcm")
                    # hermesMode from client is ignored (fixed to 'separate').
                    vad_params = vad_params_for_debounce(state.debounce_ms)
                    LOG.info("settings updated: speed=%.2f debounce=%dms wake=%s codec=%s",
                             state.speed, state.debounce_ms, state.wake_word_enabled,
                             state.audio_codec)
                    await ws.send_json({
                        "type": "settings.ack", "speed": state.speed,
                        "debounceMs": state.debounce_ms,
                        "wakeWordEnabled": state.wake_word_enabled,
                        "audioCodec": state.audio_codec,
                        "hermesMode": "separate",
                        "speakerThreshold": (SPEAKER.threshold if SPEAKER is not None else None),
                        "speakerLockEnabled": (SPEAKER.enabled if SPEAKER is not None else None),
                        "vad": vad_params, "ts": time.time(),
                    })
                else:
                    LOG.debug("ws text: %r", data)
            elif msg.type == WSMsgType.BINARY:
                if enroll_stream is not None:
                    # Owner enrollment recording — buffer, don't treat as a segment.
                    enroll_stream.extend(msg.data)
                    continue
                if clone_stream is not None:
                    # Voice-clone reference recording — buffer until commit.
                    clone_stream.extend(msg.data)
                    continue
                if active_stream is not None:
                    # B1: frame of a streamed segment — append. The final
                    # transcription happens at commit; B2 pushes in partials
                    # (throttled) in between; the voice-lock early barge-in
                    # checks the speaker as soon as enough audio streamed in.
                    if active_stream.get("opus_dec") is not None:
                        # Opus uplink: decode packets ON ARRIVAL to 16 kHz f32
                        # so the buffer stays plain PCM for every consumer.
                        # Decode errors drop the packet, never the WS loop.
                        try:
                            _pkts = audio_utils.parse_opus_uplink_packets(msg.data)
                        except ValueError as exc:
                            LOG.warning("malformed opus uplink frame dropped "
                                        "(%d bytes): %s", len(msg.data), exc)
                            _pkts = []
                        for _pkt in _pkts:
                            try:
                                active_stream["buf"].extend(
                                    active_stream["opus_dec"].decode_packet(_pkt))
                            except Exception as exc:
                                LOG.warning("opus packet decode failed "
                                            "(%d bytes): %s — packet dropped",
                                            len(_pkt), exc)
                    elif active_stream.get("codec") == "opus":
                        # Opus requested but no decoder (unsupported race):
                        # frames are opus packets — unusable, drop them.
                        continue
                    else:
                        active_stream["buf"].extend(msg.data)
                    _maybe_spawn_partial(ws, state, active_stream)
                    _maybe_spawn_speaker_barge(ws, state, active_stream)
                    _maybe_spawn_owner_watch(ws, state, active_stream, peer)
                    continue
                if pending_segment_meta:
                    _meta = pending_segment_meta.popleft()
                else:
                    _meta = {"id": _short_id(), "speech_start_ts": None, "barge_in": False}
                _spawn_tracked(state, _handle_audio_segment(
                    ws, state, msg.data, _meta["id"], peer,
                    speech_start_ts=_meta.get("speech_start_ts"),
                    barge_in=_meta.get("barge_in", False),
                    ptt=_meta.get("ptt", False)))
            elif msg.type == WSMsgType.ERROR:
                LOG.warning("ws error from %s: %s", peer, ws.exception())
                break
    finally:
        if BRIDGE is not None:
            BRIDGE.unregister_broadcast(ws)
        WS_CLIENTS.pop(ws, None)
        _cancel_connection_tasks(state)
        LOG.info("ws close: %s", peer)
        if not ws.closed:
            await ws.close()
    return ws

# ============================================================================ #
# App boot — build_app / init_backends / main / run live in plauder.app and are
# re-exported here so `plauder.server.run` (entrypoint shims) and the tests'
# `srv.build_app` keep working. Imported at the bottom to avoid an import cycle.
# ============================================================================ #
from .app import build_app, init_backends, main, run  # noqa: E402,F401

if __name__ == "__main__":
    run()
