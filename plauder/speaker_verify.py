"""Speaker lock — a self-contained speaker-VERIFICATION gate.

Goal: the voice chat should only listen to ONE enrolled owner voice. Any other
voice (children/guests/TV in the background) is dropped BEFORE it reaches the
LLM. This is speaker *verification* (1 enrolled profile, cosine similarity to a
reference embedding), NOT diarization or the optional House-Mode multi-speaker
*identification* (see server.get_speaker_identifier — a different, external
module). It is language-independent, so it complements Whisper instead of
replacing it.

Design mirrors the wake-word gate (plauder/wake.py): a cheap gate on top of the
audio we already have. The embedding model is loaded lazily via ``sherpa-onnx``
(which bundles the exact kaldi-fbank a CAM++/WeSpeaker model expects — no torch).
Everything degrades gracefully: if the dependency, the model file, or the
enrolled profile is missing, the gate is simply disabled and every segment
passes (fail-open, so a misconfiguration never bricks the mic).

Enrollment happens from the UI: the client records a few seconds of the owner's
voice; ``enroll()`` folds each take into a running-mean profile stored as JSON at
``speaker_profile_path``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np

from . import audio as audio_utils

LOG = logging.getLogger("voice-chat")

PROFILE_VERSION = 1


@dataclass
class VerifyResult:
    matched: bool
    score: float          # cosine similarity to the enrolled profile (-1..1)
    reason: str           # "match" | "mismatch" | "too_short" | "no_profile" | "disabled" | "error"


@dataclass
class WindowAnalysis:
    """Windowed speaker analysis of a segment (mixed-voice trimming).

    ``spans``       merged (start_s, end_s) time spans where the OWNER speaks
                    (already padded so word edges survive cropping).
    ``owner_ratio`` owner windows / voiced windows (0..1). 1.0 = all owner.
    ``voiced``      number of non-silent windows analyzed (0 = nothing to say).
    ``score``       best owner-window similarity (UI feedback / logging).
    ``windows``     ALL voiced windows as (start_s, end_s, score) — for
                    RELATIVE rules (absolute window scores shift wildly with
                    conditions; relative order within one segment is stabler).
    """
    spans: list
    owner_ratio: float
    voiced: int
    score: float
    windows: list | None = None


def foreign_regions(blocks, total_s: float, *, delta: float = 0.15,
                    min_region_s: float = 2.5) -> tuple:
    """Derive (foreign_regions, keep_regions) from equal-length block scores.

    A time cell counts as foreign only when EVERY block covering it scores
    below (best block − delta) — conservative: one owner-ish covering block
    protects the cell. Foreign regions shorter than ``min_region_s`` are
    dropped (sentence-level policy: single words/short dips are never cut).
    """
    if not blocks:
        return [], [(0.0, total_s)]
    best = max(sc for _, _, sc in blocks)
    bar = best - delta
    edges = sorted({0.0, total_s} | {round(b[0], 3) for b in blocks}
                   | {round(b[1], 3) for b in blocks})
    regions: list = []
    for a, b in zip(edges, edges[1:]):
        if b - a <= 1e-6:
            continue
        cov = [sc for s, e, sc in blocks if s < b - 1e-6 and e > a + 1e-6]
        is_foreign = bool(cov) and all(sc < bar for sc in cov)
        if is_foreign and regions and abs(regions[-1][1] - a) < 1e-6:
            regions[-1][1] = b
        elif is_foreign:
            regions.append([a, b])
    regions = [(a, b) for a, b in regions if b - a >= min_region_s]
    keep: list = []
    prev = 0.0
    for a, b in regions:
        if a - prev > 0.05:
            keep.append((prev, a))
        prev = b
    if total_s - prev > 0.05:
        keep.append((prev, total_s))
    return regions, keep


def spans_from_windows(windows, bar: float, total_s: float, *,
                       pad_s: float = 0.15, merge_gap_s: float = 0.4) -> list:
    """Merge windows scoring ≥ ``bar`` into padded, croppable time spans."""
    spans: list = []
    for s, e, sc in windows or []:
        if sc < bar:
            continue
        s = max(0.0, s - pad_s)
        e = min(total_s, e + pad_s)
        if spans and s - spans[-1][1] <= merge_gap_s:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return [(s, e) for s, e in spans]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return vec.astype(np.float32)
    return (vec / n).astype(np.float32)


class SpeakerVerifier:
    """Loads a sherpa-onnx speaker-embedding model and gates segments against an
    enrolled owner profile. Instances are cheap to construct; the model is only
    built in ``load()``."""

    def __init__(self, *, model_path: str, profile_path: str, threshold: float = 0.5,
                 min_dur_s: float = 0.6, provider: str = "cpu", num_threads: int = 1,
                 sample_rate: int = 16000):
        self.model_path = model_path
        self.profile_path = profile_path
        self.threshold = threshold
        self.min_dur_s = min_dur_s
        self.provider = provider
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        # Runtime kill-switch, toggled live from the UI (process-wide on the
        # shared verifier; fine for a single-owner setup, like ``threshold``).
        # When False the gate goes fully fail-open (``active()`` returns False)
        # WITHOUT forgetting the enrolled profile — a temporary "let everyone in".
        self.enabled = True
        self._extractor = None
        # Enrolled profile: running SUM of embeddings + count (so extra takes can
        # be folded in with a correct mean). The normalized mean is the reference.
        self._sum: np.ndarray | None = None
        self._count: int = 0
        self._dim: int | None = None
        self._load_profile()

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg) -> "SpeakerVerifier | None":
        if not getattr(cfg, "speaker_lock_enabled", False):
            return None
        return cls(
            model_path=cfg.speaker_model_path,
            profile_path=cfg.speaker_profile_path,
            threshold=cfg.speaker_threshold,
            min_dur_s=cfg.speaker_min_dur_s,
            provider=cfg.speaker_provider,
            num_threads=cfg.speaker_num_threads,
        )

    def load(self) -> None:
        """Build the embedding extractor. Import sherpa-onnx lazily (optional dep).
        Raises BackendError if the dependency or model file is missing."""
        from .backends.base import BackendError
        if not self.model_path or not os.path.exists(self.model_path):
            raise BackendError(
                f"SPEAKER_LOCK_ENABLED=1 but SPEAKER_MODEL_PATH is missing/not found "
                f"({self.model_path!r}). Download a CAM++/WeSpeaker ONNX model, see .env.example.")
        try:
            import sherpa_onnx  # lazy optional dep
        except ImportError as exc:
            raise BackendError(
                "SPEAKER_LOCK_ENABLED=1 but sherpa-onnx is not installed. "
                "`pip install sherpa-onnx` (no torch needed)." ) from exc
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=self.model_path,
            num_threads=self.num_threads,
            provider=self.provider,
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._dim = int(self._extractor.dim)
        LOG.info("🔒 Speaker lock loaded (dim=%d, provider=%s, enrolled=%s, thr=%.2f)",
                 self._dim, self.provider, self.has_profile(), self.threshold)

    @property
    def loaded(self) -> bool:
        return self._extractor is not None

    def has_profile(self) -> bool:
        return self._sum is not None and self._count > 0

    @property
    def window_threshold(self) -> float:
        """Acceptance bar for SHORT windows (~1.2 s). Far below the full-segment
        ``threshold``: field calibration (spk-debug) showed real-world window
        scores compress hard — the owner's own windows peak around 0.2–0.35
        while their full segments score 0.4–0.6. This floor only anchors
        "the owner is present at all"; the actual span selection is RELATIVE
        to the segment's best window (see the server's trim logic)."""
        return max(0.18, self.threshold - 0.22)

    def score_region(self, pcm, sample_rate: int | None = None,
                     start_s: float = 0.0, end_s: float | None = None) -> float:
        """Cosine of ONE contiguous region vs the profile. Block-level second
        opinion for the trim logic: block embeddings of a few seconds behave
        like full segments (reliable), unlike the noisy ~1 s windows."""
        if not self.active():
            return 0.0
        sr = sample_rate or self.sample_rate
        samples = self._samples(pcm)
        a = max(0, int(start_s * sr))
        b = samples.shape[0] if end_s is None else min(samples.shape[0], int(end_s * sr))
        if b - a < int(0.4 * sr):
            return 0.0
        try:
            emb = self.embed(samples[a:b], sr)
        except Exception:
            LOG.exception("speaker: region embed failed")
            return 0.0
        return float(np.dot(_l2_normalize(self._sum), emb))

    def active(self) -> bool:
        """Gate is live only when the model loaded AND a profile exists AND the
        runtime toggle is on. Without a profile the gate stays open (fail-open)
        so the user can still enroll; ``enabled=False`` opens it on purpose."""
        return self.loaded and self.has_profile() and self.enabled

    # -- embedding --------------------------------------------------------- #
    def _samples(self, pcm) -> np.ndarray:
        if isinstance(pcm, (bytes, bytearray, memoryview)):
            arr = audio_utils.pcm_bytes_to_float32_array(bytes(pcm))
        else:
            arr = np.asarray(pcm, dtype=np.float32)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def embed(self, pcm, sample_rate: int | None = None) -> np.ndarray:
        """Compute an L2-normalized embedding for a mono float32 @16k buffer."""
        if self._extractor is None:
            raise RuntimeError("SpeakerVerifier.load() has not run")
        samples = self._samples(pcm)
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate or self.sample_rate, waveform=samples)
        stream.input_finished()
        vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return _l2_normalize(vec)

    # -- verification gate ------------------------------------------------- #
    def verify(self, pcm, sample_rate: int | None = None,
               duration_s: float | None = None) -> VerifyResult:
        """Decide whether a segment is the enrolled owner. Fail-open on any
        problem (returns matched=True) so a misconfiguration never blocks input."""
        if not self.loaded:
            return VerifyResult(True, 0.0, "disabled")
        if not self.has_profile():
            return VerifyResult(True, 0.0, "no_profile")
        if duration_s is not None and duration_s < self.min_dur_s:
            # Too little audio for a reliable embedding. Hard lock → reject.
            return VerifyResult(False, 0.0, "too_short")
        try:
            emb = self.embed(pcm, sample_rate)
        except Exception:
            LOG.exception("speaker verify: embedding failed (letting segment pass)")
            return VerifyResult(True, 0.0, "error")
        ref = _l2_normalize(self._sum)
        score = float(np.dot(ref, emb))
        return VerifyResult(score >= self.threshold, score,
                            "match" if score >= self.threshold else "mismatch")

    # -- single-window check (owner-watch: live end-of-owner detection) ----- #
    def window_is_owner(self, pcm, sample_rate: int | None = None, *,
                        silence_rms: float = 0.005) -> bool | None:
        """Classify ONE short audio window: True = owner, False = foreign voice,
        None = silence/not decidable. Used by the server's owner-watch to detect
        that the owner stopped talking while others keep the VAD open."""
        if not self.active():
            return None
        samples = self._samples(pcm)
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        if rms < silence_rms:
            return None
        try:
            emb = self.embed(samples, sample_rate)
        except Exception:
            LOG.exception("speaker: window embed failed (treated as silence)")
            return None
        return float(np.dot(_l2_normalize(self._sum), emb)) >= self.window_threshold

    # -- equal-length block scoring (foreign-sentence detection) ------------ #
    def analyze_blocks(self, pcm, sample_rate: int | None = None, *,
                       block_s: float = 3.0, hop_s: float = 1.5,
                       silence_rms: float = 0.005) -> list:
        """Score EQUAL-LENGTH ~3 s blocks on a fixed grid against the profile.

        The model is heavily length-sensitive (same voice: 1.2 s → ~0.3,
        full sentence → ~0.9), so scores are only comparable BETWEEN BLOCKS OF
        THE SAME LENGTH. Foreign-speech detection therefore compares each block
        against the best block of the same segment — never against thresholds
        calibrated on other lengths. Returns [(start_s, end_s, score), …] for
        voiced blocks (near-silent blocks are skipped)."""
        if not self.active():
            return []
        sr = sample_rate or self.sample_rate
        samples = self._samples(pcm)
        n = samples.shape[0]
        blk = max(1, int(sr * block_s))
        hop = max(1, int(sr * hop_s))
        if n < blk:
            blk = n
        ref = _l2_normalize(self._sum)
        starts = list(range(0, max(1, n - blk + 1), hop))
        if starts and starts[-1] + blk < n:
            starts.append(n - blk)
        out = []
        for st in starts:
            w = samples[st:st + blk]
            rms = float(np.sqrt(np.mean(w * w))) if w.size else 0.0
            if rms < silence_rms:
                continue
            emb = self.embed(w, sr)
            out.append((st / sr, (st + blk) / sr, float(np.dot(ref, emb))))
        return out

    # -- windowed analysis (mixed segments: owner + other voices in turn) --- #
    def analyze_windows(self, pcm, sample_rate: int | None = None, *,
                        win_s: float = 1.2, hop_s: float = 0.6,
                        pad_s: float = 0.15, merge_gap_s: float = 0.4,
                        silence_rms: float = 0.005) -> "WindowAnalysis | None":
        """Label ~1 s windows of a segment as owner/foreign and merge the owner
        windows into croppable time spans.

        Handles the SEQUENTIAL mix case (the owner speaks, then someone else
        keeps talking into the same segment): the caller crops the audio to
        ``spans`` and re-transcribes only the owner's part. Near-silent windows
        are skipped (their embeddings are garbage and would randomly split
        spans); small gaps between owner windows are bridged by ``merge_gap_s``.
        Returns None when the verifier is not active (no model / no profile).
        """
        if not self.active():
            return None
        sr = sample_rate or self.sample_rate
        samples = self._samples(pcm)
        n = samples.shape[0]
        win = max(1, int(sr * win_s))
        hop = max(1, int(sr * hop_s))
        if n < win:
            win = n
        ref = _l2_normalize(self._sum)
        starts = list(range(0, max(1, n - win + 1), hop))
        if starts and starts[-1] + win < n:      # cover the tail
            starts.append(n - win)
        voiced = 0
        all_windows: list = []
        scores: list = []
        for st in starts:
            w = samples[st:st + win]
            rms = float(np.sqrt(np.mean(w * w))) if w.size else 0.0
            if rms < silence_rms:
                continue
            voiced += 1
            emb = self.embed(w, sr)
            score = float(np.dot(ref, emb))
            all_windows.append((st / sr, (st + win) / sr, score))
            if score >= self.window_threshold:
                scores.append(score)
        return WindowAnalysis(
            spans=spans_from_windows(all_windows, self.window_threshold, n / sr,
                                     pad_s=pad_s, merge_gap_s=merge_gap_s),
            owner_ratio=(len(scores) / voiced) if voiced else 0.0,
            voiced=voiced,
            score=(max(scores) if scores else 0.0),
            windows=all_windows,
        )

    # -- enrollment -------------------------------------------------------- #
    def enroll(self, pcm, sample_rate: int | None = None) -> dict:
        """Fold one recorded take into the running-mean owner profile and persist.
        Returns a status dict (also usable as the WS ack payload)."""
        if self._extractor is None:
            raise RuntimeError("SpeakerVerifier.load() has not run")
        emb = self.embed(pcm, sample_rate)          # already L2-normalized
        # Self-similarity against the profile so far — feedback on take quality.
        prev_score = None
        if self.has_profile():
            prev_score = float(np.dot(_l2_normalize(self._sum), emb))
        if self._sum is None:
            self._sum = emb.copy()
            self._dim = emb.shape[0]
        else:
            self._sum = self._sum + emb
        self._count += 1
        self._save_profile()
        return {**self.status(), "sampleScore": prev_score}

    def clear_profile(self) -> None:
        self._sum = None
        self._count = 0
        try:
            if self.profile_path and os.path.exists(self.profile_path):
                os.remove(self.profile_path)
        except OSError:
            LOG.warning("speaker: could not remove profile %s", self.profile_path)

    def status(self) -> dict:
        return {
            "available": self.loaded,
            "enrolled": self.has_profile(),
            "count": self._count,
            "threshold": self.threshold,
            "dim": self._dim,
        }

    # -- persistence ------------------------------------------------------- #
    def _load_profile(self) -> None:
        if not self.profile_path or not os.path.exists(self.profile_path):
            return
        try:
            with open(self.profile_path, encoding="utf-8") as fh:
                data = json.load(fh)
            summ = np.asarray(data.get("sum") or [], dtype=np.float32)
            count = int(data.get("count") or 0)
            # Embeddings are model-specific: a profile enrolled with another
            # model is useless (different space, often different dim). Ignore
            # it — the gate then fails open until the owner re-enrolls.
            prof_model = data.get("model")
            if prof_model and self.model_path and prof_model != os.path.basename(self.model_path):
                LOG.warning("speaker: profile %s was enrolled with model %s, "
                            "current model is %s — ignoring profile (re-enroll needed)",
                            self.profile_path, prof_model, os.path.basename(self.model_path))
                return
            if summ.size and count > 0:
                self._sum = summ
                self._count = count
                self._dim = summ.shape[0]
        except Exception:
            LOG.exception("speaker: failed to load profile %s (ignored)", self.profile_path)

    def _save_profile(self) -> None:
        if not self.profile_path or self._sum is None:
            return
        payload = {
            "version": PROFILE_VERSION,
            "model": os.path.basename(self.model_path),
            "dim": int(self._dim or self._sum.shape[0]),
            "count": self._count,
            "sum": [float(x) for x in self._sum.tolist()],
            "updated": time.time(),
        }
        try:
            # Atomic write so a crash mid-save can't corrupt the profile.
            d = os.path.dirname(os.path.abspath(self.profile_path)) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self.profile_path)
        except Exception:
            LOG.exception("speaker: failed to save profile %s", self.profile_path)
