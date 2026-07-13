"""Speaker lock — gating policy around the SpeakerVerifier.

Everything that DECIDES based on speaker scores lives here; the scoring
itself (embeddings, thresholds, block/window analysis) is
``plauder.speaker_verify``. Three cooperating mechanisms:

- **Commit gate** (``apply_commit_gate``): hard lock at segment commit — only
  the enrolled owner voice is transcribed; foreign voices are dropped, and
  sustained foreign passages riding along in an owner segment are cut out of
  the transcript (block-relative trim + contiguous re-score veto + re-STT).
- **Early barge-in** (``maybe_spawn_speaker_barge``): with the lock engaged
  the client does NOT stop playback on VAD speech (any voice would trigger
  that — the whole point of the lock is that foreign voices must not
  interrupt). Instead, while a segment streams in (B1) and a reply is in
  flight, the growing buffer is verified against the owner profile as soon
  as enough audio exists; only a CONFIRMED owner voice cancels the running
  reply. Foreign voices leave it playing.
- **Owner-watch** (``maybe_spawn_owner_watch``): server-side end-of-owner
  detection (segmentation fix). The browser VAD is speaker-agnostic: it
  closes a segment on SILENCE, not when the OWNER stops. If other voices
  keep talking (kids), the client segment never ends and the owner's
  utterance would sit in the buffer indefinitely. The server therefore
  scores the streaming buffer in ~1.2 s windows; once the owner HAS spoken
  and then hasn't been heard for the debounce time, the buffered audio is
  committed as the owner's utterance and the stream is re-armed as a fresh
  virtual segment — the foreign tail is then judged on its own at the real
  client commit (and rejected by the gate).

Runtime state (CFG/SPEAKER/STT/GHOST, set by ``configure``) lives in
``plauder.server``; this module reads it at call time via ``server.<name>``
to avoid an import cycle — same pattern as ``plauder.app``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

from . import audio as audio_utils
from . import server
from .config import SAMPLE_RATE
from .speaker_verify import foreign_regions, keep_regions
from .turn_state import TurnState

LOG = logging.getLogger("voice-chat")

# Laughter / nonverbal vocalizations are acoustically NOT the owner's normal
# speech, so the speaker embedding scores them like a foreign voice and the
# block-trim would cut them out (struck through in the UI). But laughter is
# genuine owner input the LLM needs as conversational context — it must never
# be trimmed. This matches the whole utterance OR the bit the trim removed:
# Whisper renders laughter as repeated ha/he/hi/ho syllables ("Hahaha", "Ha
# ha ha", "Hehehe") or as a bracketed/asterisked tag ("*Gelächter*",
# "[laughter]", "*Lachen*"). Punctuation/whitespace is ignored.
_LAUGH_RE = re.compile(
    r"^[\s\.\,\!\?…\-—–\"'»«„“”]*"
    r"(?:"
    r"(?:ha|he|hi|ho|hah|heh|hih|haha|hehe|haw|mhm|hm+)"
    r"(?:[\s\-\.\,]*(?:ha|he|hi|ho|hah|heh|hih|haha|hehe|haw|hm+))*"
    r"|[\*\[\(]\s*(?:gelächter|gelaechter|lachen|lacht|laughter|laughs|laughing|chuckle|giggl\w*)\s*[\*\]\)]"
    r")"
    r"[\s\.\,\!\?…\-—–\"'»«„“”]*$",
    re.IGNORECASE,
)


def is_laughter_only(text: str) -> bool:
    """True when ``text`` is purely laughter / a nonverbal-laughter tag (no
    real words). Used to protect laughter from the speaker block-trim: the
    embedding scores laughter like a foreign voice, but it is genuine owner
    input the LLM needs.
    """
    if not text or not text.strip():
        return False
    return bool(_LAUGH_RE.match(text.strip()))

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

OWNER_WATCH_WIN_S = 1.2    # window fed to the embedding model
OWNER_WATCH_HOP_S = 0.6    # step between scored windows


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


async def spk_dump(tag, pcm_bytes, score, decision):
    if not (server.CFG and server.CFG.speaker_dump_dir):
        return
    try:
        await asyncio.to_thread(_spk_dump_sync, server.CFG.speaker_dump_dir, tag,
                                pcm_bytes, score, decision)
    except Exception:
        LOG.exception("speaker dump failed (%s)", tag)


# --------------------------------------------------------------------------- #
# Commit gate (full-segment decision at segment commit)
# --------------------------------------------------------------------------- #
async def apply_commit_gate(ws, state: TurnState, *, segment_id, pcm_bytes,
                            text: str, duration_s: float, ptt: bool,
                            verify_task, wake_gating: bool) -> dict | None:
    """Voice gate at segment commit. Hard lock: only the enrolled owner voice
    is transcribed; any other voice is dropped here, before the wake gate /
    LLM. Voice only (typed input bypasses this path entirely).

    Returns ``None`` when the segment was rejected (``transcript.ignored`` has
    been sent, debounce resumed — the caller just drops the segment), else a
    dict: ``text`` (possibly trimmed), ``score``, ``trimmed``, ``full_text``
    (pre-trim transcript, None unless trimmed) and ``stt_ms`` (extra STT time
    spent on re-transcribing trimmed audio)."""
    CFG, SPEAKER, STT, GHOST = server.CFG, server.SPEAKER, server.STT, server.GHOST
    speaker_score = None
    speaker_trimmed = False
    speaker_full_text = None
    stt_ms = 0

    async def _reject(score):
        LOG.info("speaker: rejected seg=%s score=%.3f text=%r",
                 segment_id, score, text[:80])
        await spk_dump(segment_id, pcm_bytes, score, "rej")
        await ws.send_json({
            "type": "transcript.ignored", "segmentId": segment_id,
            "turnId": state.turn_id, "text": text,
            "reason": "speaker_mismatch", "speakerScore": round(score, 4),
            "ts": time.time()})
        server._resume_debounce(ws, state)

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
    # (Started before STT — see verify_task in the caller — so both overlap.)
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
    # Owner-laughter pass: a segment that is PURELY laughter scores like a
    # foreign voice (laughter embeds unlike normal speech), so it would be
    # rejected outright. But laughter right after the owner was heard is the
    # owner laughing — genuine input the LLM needs. Accept it when the text is
    # laughter-only AND the owner was recently confirmed (continuity window),
    # so a stranger laughing out of the blue still doesn't pass.
    if (not matched and is_laughter_only(text)
            and state.speaker_last_own > 0
            and time.time() - state.speaker_last_own_ts <= SPEAKER_CONT_WINDOW_S):
        matched = True
        LOG.info("speaker: owner-laughter passes seg=%s dur=%.1fs score=%.3f "
                 "text=%r", segment_id, duration_s, res.score, text[:40])
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
        # Laughter guard: if the trim removed laughter that was in the original
        # transcript, the "foreign" block was almost certainly the owner
        # laughing (laughter embeds unlike normal speech and looks foreign to
        # the gate). Laughter is genuine owner input the LLM needs — keep the
        # full transcript, don't cut. Detect via the laughter tokens present
        # in the original but gone after the trim.
        removed_laughter = [
            tok for tok in re.findall(r"[^\s.,!?…]+|[\*\[\(][^\*\]\)]+[\*\]\)]", text)
            if is_laughter_only(tok) and tok not in text2
        ]
        if removed_laughter:
            LOG.info("speaker: trim SKIPPED seg=%s — would cut laughter %r "
                     "(kept full transcript)", segment_id, removed_laughter)
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
            return None
    await spk_dump(segment_id, pcm_bytes, res.score,
                   "trim" if speaker_trimmed else "ok")
    # Owner confirmed → this segment may barge in. When the up-front cancel
    # was deferred purely for the speaker gate (pure VAD, no wake word), do
    # it now so the owner can interrupt a running reply. (No-op if the early
    # mid-stream check already cancelled it.) If the reply has not started
    # playing yet, the cancelled turn's input is carried over (coalescing).
    if not wake_gating:
        await server._coalesce_cancel(state, ws)
    return {"text": text, "score": speaker_score, "trimmed": speaker_trimmed,
            "full_text": speaker_full_text, "stt_ms": stt_ms}


# --------------------------------------------------------------------------- #
# Voice-lock barge-in: early speaker check on the streaming segment
# --------------------------------------------------------------------------- #
async def _do_speaker_barge(ws, state: TurnState, seg: dict, pcm: bytes):
    SPEAKER = server.SPEAKER
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
        await server._coalesce_cancel(state, ws)
    else:
        LOG.debug("speaker: mid-stream check not owner seg=%s score=%.3f (reply keeps playing)",
                  seg.get("id"), score)


def maybe_spawn_speaker_barge(ws, state: TurnState, seg: dict):
    """Throttle check + start of an early speaker verification (see above).
    Sync — spawns a task when needed, does not block the frame loop.

    Wake mode is deliberately excluded: there the wake gate at commit decides
    what may interrupt (deferred cancel), and an open-window follow-up already
    passes both gates at commit."""
    SPEAKER = server.SPEAKER
    if SPEAKER is None or not SPEAKER.active():
        return
    if not server.CFG or getattr(state, "wake_word_enabled", False):
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
    server._spawn_tracked(state, _do_speaker_barge(ws, state, seg, bytes(seg["buf"])))


# --------------------------------------------------------------------------- #
# Owner-watch: server-side end-of-owner detection (segmentation fix)
# --------------------------------------------------------------------------- #
def maybe_spawn_owner_watch(ws, state: TurnState, seg: dict, peer):
    """Throttle check + start of one owner-watch window score. Sync — spawns a
    task when a full new window of audio is available."""
    SPEAKER = server.SPEAKER
    if SPEAKER is None or not SPEAKER.active() or not server.CFG:
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
    server._spawn_tracked(state, _owner_watch_step(ws, state, seg, pcm, next_off + win, peer))


async def _owner_watch_step(ws, state: TurnState, seg: dict, pcm: bytes,
                            end_off: int, peer):
  # The own_running guard spans the WHOLE step including the veto embeds —
  # otherwise a parallel step could double-commit the same buffer.
  SPEAKER, CFG = server.SPEAKER, server.CFG
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
    server._spawn_tracked(state, server._handle_audio_segment(
        ws, state, pcm_head, seg_id, peer,
        speech_start_ts=speech_start, barge_in=barge))
  finally:
    seg["own_running"] = False
