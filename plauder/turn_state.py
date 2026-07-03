"""Per-connection turn state + VAD parameters.

Debounce + coalescing: voice segments and text sends share a single debounce
window. New inputs during the window are collected and, after it expires, sent
off in ONE LLM call.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field


def _stable_session_id() -> str:
    """Derive a deterministic session ID from HERMES_SESSION_KEY_SEPARATE.

    Uses SHA-256 of the key so the voice session ID is stable across reconnects
    (same key → same ID → Hermes loads the previous turn history). Falls back to
    a random UUID if the env var is not set.
    """
    key = os.environ.get("HERMES_SESSION_KEY_SEPARATE", "")
    if not key:
        return uuid.uuid4().hex
    return hashlib.sha256(key.encode()).hexdigest()

# VAD frames ≈ 32 ms per frame at 16 kHz/512 samples (silero-vad-web).
VAD_FRAME_MS = 32
VAD_REDEMPTION_MIN = 8       # ~256 ms
VAD_REDEMPTION_MAX = 160     # ~5.1 s


def vad_params_for_debounce(debounce_ms: int) -> dict:
    """VAD parameters depending on the desired debounce. For long thinking
    pauses the VAD must hold the silence longer before it fires "speech end".
    """
    frames = int(round((debounce_ms * 0.8) / VAD_FRAME_MS))
    frames = max(VAD_REDEMPTION_MIN, min(VAD_REDEMPTION_MAX, frames))
    return {
        "redemptionFrames": frames,
        "minSpeechFrames": 3,
        "preSpeechPadFrames": 8,
        "frameMs": VAD_FRAME_MS,
    }


@dataclass
class TurnState:
    """Holds the current turn of a connection (voice + text + images)."""
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    pending_texts: list = field(default_factory=list)
    pending_segment_ids: list = field(default_factory=list)
    pending_text_parts: list = field(default_factory=list)
    pending_image_urls: list = field(default_factory=list)
    debounce_task: object = None
    agent_task: object = None
    audio_ids: set = field(default_factory=set)
    speed: float = 1.0
    debounce_ms: int = 1200
    # Legacy: parallel text tasks. Nothing appends to this anymore, but the
    # server still iterates it in `_cancel_in_flight` — so it is kept on purpose.
    text_tasks: list = field(default_factory=list)
    # Detached segment-/partial-handler tasks, tracked so the connection can
    # cancel them on close (otherwise they may send on a closed WebSocket).
    inflight_tasks: set = field(default_factory=set)
    # User suffix that determines the LLM session key per connection.
    session_user: str | None = None
    # Wake word is a per-connection input mode (set by the client). When it is
    # off, the connection runs like pure VAD/PTT without a gate. Initial value =
    # CFG.wake_word_enabled (start default), then toggleable via 'settings'.
    wake_word_enabled: bool = False
    # Wake word: up to this point in time (time.time()) the conversation window
    # is open → segments without a wake word are let through (follow-up questions).
    wake_until: float = 0.0
    # Segment ID for which a `wake.detected` (acoustic early feedback) was already
    # sent — prevents a double chime from partial + final segment.
    wake_detected_seg: str | None = None
    # Did the user manually close the conversation window during an in-flight
    # reply ('wake.close')? Then do NOT use the otherwise following playback.done
    # to re-open the window.
    wake_suppress_reopen: bool = False
    # After a manual close, a brief guard (time.time() threshold): until then NO
    # automatic re-opening and no wake.detected — otherwise a trailing partial /
    # echo / background noise would re-open the window immediately.
    wake_closed_until: float = 0.0
    # End-to-end latency anchor: time.time() at which the last segment
    # contributing to the turn arrived at the server ("user is done speaking").
    # The time until the first playback is measured against this point.
    speech_end_ts: float = 0.0
    # Combined input text of the turn currently being processed (set by
    # _run_turn for its lifetime). If the owner keeps speaking BEFORE any audio
    # of the reply played, the cancelled turn's input is re-queued into the
    # next turn (coalescing) instead of being dropped.
    inflight_combined: str | None = None
    # Image URLs of the in-flight turn — coalescing must restore them together
    # with the text, or a re-queued "what's in this photo?" loses its photo.
    inflight_images: list = field(default_factory=list)
    # True once the current turn actually EMITTED audio (audio.start sent).
    # audio_ids is set eagerly at stream start (for the audio.stop backup), so
    # it cannot distinguish "reply audible" from "still thinking".
    audio_started: bool = False
    # True while the CLIENT is (presumably) still playing reply audio: set at
    # audio.start, cleared on playback.done / audio.stop. Crucial for barge-in:
    # a fast TTS ships all chunks within ~1-2 s (audio_ids emptied at
    # audio.end), while the client keeps PLAYING for many seconds — during
    # that time the server would otherwise see "nothing to interrupt".
    client_playing: bool = False
    # Voice lock, temporal continuity: full-verify score + time of the last
    # segment that STRICTLY matched the owner. Segments shortly after get a
    # relative bar (last_own − Δ) instead of the absolute threshold — the same
    # voice trailing on scores slightly lower (sentence tails after an
    # owner-watch split, longer utterances), while foreign voices stay far
    # below. Preserved across turns (deliberately not cleared by reset()).
    speaker_last_own: float = 0.0
    speaker_last_own_ts: float = 0.0
    # Hermes session ID for the voice session (deterministic from the session
    # key so it survives reconnects; rotated on explicit /new).
    hermes_session_id_separate: str = field(default_factory=_stable_session_id)

    def has_pending(self) -> bool:
        return bool(self.pending_texts or self.pending_text_parts or self.pending_image_urls)

    def reset(self) -> None:
        """Begin a new turn (after a successful agent call).

        Not a full reset: only turn_id, the pending lists and speech_end_ts are
        RESET. Deliberately PRESERVED are audio_ids, all wake_* fields, speed,
        debounce_ms and session_user (they apply across turns).
        """
        self.turn_id = uuid.uuid4().hex[:8]
        self.pending_texts.clear()
        self.pending_segment_ids.clear()
        self.pending_text_parts.clear()
        self.pending_image_urls.clear()
        self.speech_end_ts = 0.0
