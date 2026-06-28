#!/usr/bin/env python3
"""Voice-Chat Server — HTTP/WebSocket layer.

Transport + turn orchestration only. STT/TTS/LLM sit behind pluggable
backends (see plauder.backends), chosen via .env. Text processing
(sanitizer, hallucination filter, merging) and turn state are separate modules.

Pipeline:
  Browser (16 kHz float32 PCM via VAD/push-to-talk)
    └─ WebSocket → TurnState (debounce + coalescing)
                 → STT.transcribe → hallucination filter
                 → ConversationManager.chat (LLM + history)
                 → sanitizer → TTS.synth → WAV → browser
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from pathlib import Path

from aiohttp import WSMsgType, web

from . import audio as audio_utils
from . import sanitizer
from . import wake
from .backends import LLMBackend, STTBackend, TTSBackend, UpstreamTimeoutError
from .config import SAMPLE_RATE, Config
from .images import _resolve_image_urls
from .session import ConversationManager
from .telegram_bridge import TelegramBridge
from .turn_state import TurnState, vad_params_for_debounce

LOG = logging.getLogger("voice-chat")

HERE = Path(__file__).resolve().parent.parent
STATIC_DIR = HERE / "static"
INDEX_HTML = STATIC_DIR / "index.html"

WS_MAX_MSG_BYTES = 16 * 1024 * 1024   # max size of a single WebSocket frame (image data URLs)
WS_HEARTBEAT_S = 20.0                 # WebSocket ping interval
AGENT_RETRY_DELAY_S = 0.5             # pause before the single silent retry on an idle timeout
SEGMENT_ID_LEN = 8                    # length of the short hex ids used for segments/turns

STAGE = 6  # protocol version (client-compatible)


def _short_id() -> str:
    """Short random hex id for segments / turns / ephemeral users."""
    return uuid.uuid4().hex[:SEGMENT_ID_LEN]

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

# House-Mode speaker-ID (lazy, optional — the speaker_id module is not part of
# this repo; stays disabled when it is missing).
_SPEAKER_IDENTIFIER = None
_SPEAKER_INIT_FAILED = False


def configure(cfg: Config, *, stt=None, tts=None, conv=None, bridge=None, ghost=None):
    """Sets the runtime state. Tests can inject mock backends here."""
    global CFG, STT, TTS, CONV, BRIDGE, GHOST
    CFG = cfg
    STT = stt
    TTS = tts
    CONV = conv
    BRIDGE = bridge
    GHOST = ghost if ghost is not None else sanitizer.HallucinationFilter.from_config(cfg)


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


# ============================================================================ #
# HTTP routes
# ============================================================================ #
async def index(_request):
    if not INDEX_HTML.exists():
        return web.Response(status=500, text=f"index.html missing: {INDEX_HTML}")
    # Inject the configured UI language (APP_LANGUAGE) so the page renders in the
    # right locale immediately, without a flash of the fallback language.
    lang = (CFG.app_language if CFG else "en")
    html = INDEX_HTML.read_text(encoding="utf-8").replace("__APP_LANG__", lang)
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
        start_evt["debounceMs"] = state.debounce_ms  # pause share of e2e
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
    """TTS streaming with fallback: uses ``synth_stream`` if present, otherwise
    the classic ``synth`` (one chunk). Keeps the orchestrator backend-agnostic."""
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
    """Streaming turn (A1+A2).

    LLM tokens are read live; as soon as a sentence is complete, it goes into
    the TTS queue. A parallel TTS worker synthesizes sentence by sentence and
    sends the audio as PCM chunks (VCT2) progressively to the client — sentence 1
    is played back while sentence 2 is still being generated/synthesized.
    """
    pron = CFG.pronunciations_file if CFG else None
    max_chars = CFG.tts_max_chars_per_chunk if CFG else 220
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
                        # Latency breakdown up to the FIRST playback (≠ total time):
                        if anchor:
                            start_evt["e2eMs"] = int((now - anchor) * 1000)
                            start_evt["debounceMs"] = state.debounce_ms  # pause share of e2e
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
        """Sends new reply text to the client — but only once it's clear that
        it's not a pure NO_REPLY (otherwise 'NO_REPLY' would briefly flash up)."""
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
            return  # still looks like NO_REPLY → hold back the sentences
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
        return
    except Exception as exc:
        await _drain_tts()
        state.audio_ids.discard(audio_id)
        LOG.exception("turn=%s agent stream failed", turn_id)
        await ws.send_json({"type": "reply.error", "turnId": turn_id, "replyId": reply_id,
                            "error": str(exc), "ts": time.time()})
        return

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


def _resume_debounce(ws, state: TurnState) -> None:
    """(Re)arm the debounce timer when coalesced input is still pending.

    Used by the segment paths that drop the current segment (ghost-filtered,
    wake-gated, empty transcript) but must not strand earlier pending input.

    NOTE: deliberately does NOT cancel an existing debounce_task. That slot is
    reused by `_debounce_then_run` to hold the *running* turn once the timer
    elapses, so cancelling it here would abort an in-flight reply — which the
    wake "ignored during processing" tests rely on NOT happening. A true
    debounce-reset that is safe against this would need to separate the timer
    slot from the running-turn slot (see B2 in the review notes).
    """
    if state.has_pending():
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

    # With the wake word active, a running turn is NOT cancelled immediately —
    # only once the segment passes the gate (otherwise foreign chatter would
    # interrupt Antonia mid-sentence even though it isn't directed at her).
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

    # Hallucination filter
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
            state.wake_until = 0.0
            state.wake_detected_seg = None
            LOG.info("wake: stop seg=%s → window closed", segment_id)
            await ws.send_json({
                "type": "wake.closed", "turnId": state.turn_id,
                "reason": "stop_command", "ts": time.time()})
            return

        # Command to the AI → refresh the window; a reply is coming, so the
        # idle timer does NOT run (reason=command). Then clear the running turn.
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
                LOG.exception("speaker-id failed for segment %s (ignored)", segment_id)

    transcript_evt = {
        "type": "transcript", "segmentId": segment_id, "turnId": state.turn_id,
        "text": text, "sttMs": stt_ms, "totalMs": int((time.time() - t0) * 1000),
        "ts": time.time(),
    }
    if speaker_info is not None:
        transcript_evt["speaker"] = speaker_info
    await ws.send_json(transcript_evt)

    if not text:
        _resume_debounce(ws, state)
        return

    state.pending_texts.append(_speaker_tag(text, speaker_info))
    state.pending_segment_ids.append(segment_id)
    # E2E anchor: receive time of this (last) segment. With coalescing the
    # respective last segment wins → measurement from "user last spoke".
    state.speech_end_ts = t0
    await _send_turn_pending(ws, state)
    state.debounce_task = asyncio.create_task(_debounce_then_run(ws, state))


# ============================================================================ #
# B2: Streaming STT (interim transcripts while a segment is being streamed in)
# ============================================================================ #
async def _do_partial(ws, state: TurnState, seg: dict, pcm: bytes):
    """Transcribes the buffer accumulated so far and sends a
    transcript.partial — UI only, does not change any turn state."""
    try:
        text = await STT.transcribe(pcm, SAMPLE_RATE)
    except Exception:
        LOG.exception("partial STT failed seg=%s", seg.get("id"))
        text = ""
    finally:
        seg["partial_running"] = False
    # Segment committed/aborted in the meantime? Then discard the partial.
    if seg.get("done") or not text:
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

    if BRIDGE is not None:
        BRIDGE.register_broadcast(ws)

    await ws.send_json({
        "type": "hello", "stage": STAGE,
        "msg": f"Server ready – agent: {CFG.agent_name}.",
        "agent_name": CFG.agent_name,
        "lang": (CFG.app_language if CFG else "en"),
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
            "available": True,             # wake mode is always selectable (UI)
            "enabled": bool(CFG.wake_word_enabled) if CFG else False,  # start default
            "word": CFG.wake_word if CFG else "",
            "windowS": CFG.wake_word_window_s if CFG else 0,
            "mode": _wake_mode(),          # "conversation" | "alexa" (one-shot)
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
                    ident = get_speaker_identifier()
                    if ident is not None:
                        try:
                            ident.mark_playback_finished()
                        except Exception:
                            LOG.exception("mark_playback_finished failed (ignored)")
                elif t == "barge_in":
                    reason = (data.get("reason") or "speech")
                    # Wake mode + manually closed/guard: the speech is NOT directed
                    # at Antonia → do NOT cancel the running reply.
                    if CFG and state.wake_word_enabled and _wake_closed_active(state):
                        LOG.info("barge-in ignored (wake closed) from %s (reason=%s)", peer, reason)
                    else:
                        in_flight = (state.agent_task and not state.agent_task.done()) or bool(state.audio_ids)
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
                    })
                elif t == "segment.stream.start":
                    # B1: start of a streamed segment. Following binary frames
                    # (raw f32le) are appended until segment.stream.commit arrives.
                    _sst = data.get("speechStartTs")
                    _cnow = data.get("clientNow")
                    speech_start_ts = None
                    if isinstance(_sst, (int, float)) and isinstance(_cnow, (int, float)):
                        speech_start_ts = time.time() - (float(_cnow) - float(_sst)) / 1000.0
                    active_stream = {
                        "id": data.get("segmentId") or _short_id(),
                        "buf": bytearray(),
                        "speech_start_ts": speech_start_ts,
                        "barge_in": bool(data.get("bargeIn")),
                        # B2: partial throttle state.
                        "partial_running": False, "last_partial_ts": 0.0,
                        "last_partial_len": 0, "partial_text": "", "done": False,
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
                        # Leaving wake mode → close an open conversation window.
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
                    # B1: frame of a streamed segment — append. The final
                    # transcription happens at commit; B2 pushes in partials
                    # (throttled) in between.
                    active_stream["buf"].extend(msg.data)
                    _maybe_spawn_partial(ws, state, active_stream)
                    continue
                if pending_segment_meta:
                    _meta = pending_segment_meta.popleft()
                else:
                    _meta = {"id": _short_id(), "speech_start_ts": None, "barge_in": False}
                _spawn_tracked(state, _handle_audio_segment(
                    ws, state, msg.data, _meta["id"], peer,
                    speech_start_ts=_meta.get("speech_start_ts"),
                    barge_in=_meta.get("barge_in", False)))
            elif msg.type == WSMsgType.ERROR:
                LOG.warning("ws error from %s: %s", peer, ws.exception())
                break
    finally:
        if BRIDGE is not None:
            BRIDGE.unregister_broadcast(ws)
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
