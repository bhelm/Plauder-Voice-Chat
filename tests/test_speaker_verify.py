"""Speaker-verification gate — unit tests with a fake embedding extractor.

The real embeddings come from a sherpa-onnx model; here we inject a tiny fake
extractor that maps the sign of the audio's mean to one of two orthogonal
vectors, so "owner" (positive) and "impostor" (negative) are cleanly separable.
"""
import numpy as np

from plauder.speaker_verify import SpeakerVerifier


class _FakeStream:
    def __init__(self):
        self.s = None

    def accept_waveform(self, sample_rate, waveform):
        self.s = np.asarray(waveform, dtype=np.float32)

    def input_finished(self):
        pass


class _FakeExtractor:
    dim = 4

    def create_stream(self):
        return _FakeStream()

    def compute(self, stream):
        m = float(np.mean(stream.s)) if stream.s is not None and stream.s.size else 0.0
        return [1.0, 0.0, 0.0, 0.0] if m >= 0 else [0.0, 1.0, 0.0, 0.0]


def _mk(tmp_path, *, loaded=True, threshold=0.5, min_dur_s=0.5):
    sv = SpeakerVerifier(model_path="fake.onnx", profile_path=str(tmp_path / "profile.json"),
                         threshold=threshold, min_dur_s=min_dur_s)
    if loaded:
        sv._extractor = _FakeExtractor()
        sv._dim = 4
    return sv


_OWNER = np.full(1600, 0.5, dtype=np.float32)
_IMPOSTOR = np.full(1600, -0.5, dtype=np.float32)


def test_enroll_then_owner_matches_impostor_rejected(tmp_path):
    sv = _mk(tmp_path)
    assert not sv.has_profile()
    sv.enroll(_OWNER)
    assert sv.has_profile() and sv.active()

    owner = sv.verify(_OWNER, duration_s=1.0)
    assert owner.matched and owner.reason == "match" and owner.score > 0.9

    impostor = sv.verify(_IMPOSTOR, duration_s=1.0)
    assert not impostor.matched and impostor.reason == "mismatch"


def test_too_short_segment_is_rejected(tmp_path):
    sv = _mk(tmp_path, min_dur_s=0.5)
    sv.enroll(_OWNER)
    res = sv.verify(_OWNER, duration_s=0.1)   # below min_dur_s
    assert not res.matched and res.reason == "too_short"


def test_fail_open_without_profile_or_model(tmp_path):
    # Loaded but not enrolled → gate open (so the user can still enroll/speak).
    sv = _mk(tmp_path)
    assert not sv.active()
    r = sv.verify(_OWNER, duration_s=1.0)
    assert r.matched and r.reason == "no_profile"

    # Not loaded at all → disabled, everything passes.
    sv2 = _mk(tmp_path, loaded=False)
    r2 = sv2.verify(_OWNER, duration_s=1.0)
    assert r2.matched and r2.reason == "disabled"


def test_profile_persists_across_instances(tmp_path):
    sv = _mk(tmp_path)
    sv.enroll(_OWNER)
    assert sv._count == 1

    # A fresh instance loads the profile from disk in __init__.
    sv2 = _mk(tmp_path)
    assert sv2.has_profile() and sv2._count == 1
    assert sv2.verify(_OWNER, duration_s=1.0).matched
    assert not sv2.verify(_IMPOSTOR, duration_s=1.0).matched


def test_clear_profile_removes_file(tmp_path):
    sv = _mk(tmp_path)
    sv.enroll(_OWNER)
    assert sv.has_profile()
    sv.clear_profile()
    assert not sv.has_profile()
    assert not (tmp_path / "profile.json").exists()
    # After clearing, the gate is open again (fail-open).
    assert sv.verify(_IMPOSTOR, duration_s=1.0).matched


def test_multiple_takes_average(tmp_path):
    sv = _mk(tmp_path)
    st1 = sv.enroll(_OWNER)
    assert st1["count"] == 1 and st1["sampleScore"] is None
    st2 = sv.enroll(_OWNER)
    assert st2["count"] == 2 and st2["sampleScore"] is not None


# --- windowed analysis (mixed-voice trimming) --------------------------------
SR = 16000


def test_analyze_windows_mixed_segment(tmp_path):
    """Owner speaks 0–2 s, impostor 2–4 s → one owner span over the first ~2 s."""
    sv = _mk(tmp_path)
    sv.enroll(_OWNER)
    seg = np.concatenate([np.full(SR * 2, 0.5, np.float32),
                          np.full(SR * 2, -0.5, np.float32)])
    wa = sv.analyze_windows(seg, SR)
    assert wa is not None and wa.voiced >= 4
    assert 0.3 < wa.owner_ratio < 0.8
    assert len(wa.spans) == 1
    s, e = wa.spans[0]
    assert s == 0.0 and 1.5 < e < 2.9   # covers the owner part, not the tail


def test_analyze_windows_skips_silence(tmp_path):
    """Silent windows are neither owner nor foreign — they must not break spans
    or dilute the ratio."""
    sv = _mk(tmp_path)
    sv.enroll(_OWNER)
    seg = np.concatenate([np.full(SR * 2, 0.5, np.float32),
                          np.zeros(SR * 2, np.float32),
                          np.full(SR * 2, 0.5, np.float32)])
    wa = sv.analyze_windows(seg, SR)
    assert wa.owner_ratio == 1.0        # every VOICED window is the owner


def test_analyze_windows_inactive_returns_none(tmp_path):
    sv = _mk(tmp_path)                  # loaded but not enrolled → not active
    assert sv.analyze_windows(_OWNER, SR) is None


def test_analyze_blocks_equal_length_grid(tmp_path):
    """Owner 0–4 s, impostor 4–8 s → equal-length 3 s blocks separate them."""
    sv = _mk(tmp_path)
    sv.enroll(_OWNER)
    seg = np.concatenate([np.full(SR * 4, 0.5, np.float32),
                          np.full(SR * 4, -0.5, np.float32)])
    blocks = sv.analyze_blocks(seg, SR)
    assert blocks and all(abs((e - s) - 3.0) < 0.01 for s, e, _ in blocks)
    head = [sc for s, e, sc in blocks if e <= 4.0]
    tail = [sc for s, e, sc in blocks if s >= 4.0]
    assert min(head) > max(tail)      # owner blocks clearly above impostor blocks


def test_foreign_regions_sentence_policy():
    from plauder.speaker_verify import foreign_regions
    # Foreign tail ≥ 2.5 s → cut; keep = the owner's head.
    blocks = [(0.0, 3.0, 0.50), (1.5, 4.5, 0.20), (3.0, 6.0, 0.15)]
    foreign, keep = foreign_regions(blocks, 6.0, delta=0.15, min_region_s=2.5)
    assert foreign == [(3.0, 6.0)]
    assert keep == [(0.0, 3.0)]
    # Short dip (< min_region_s) → nothing cut.
    blocks = [(0.0, 3.0, 0.50), (2.0, 5.0, 0.20), (3.5, 6.5, 0.55)]
    foreign, keep = foreign_regions(blocks, 6.5, delta=0.15, min_region_s=2.5)
    assert foreign == [] and keep == [(0.0, 6.5)]
    # All blocks close together → same speaker, nothing cut.
    blocks = [(0.0, 3.0, 0.45), (1.5, 4.5, 0.40)]
    foreign, keep = foreign_regions(blocks, 4.5, delta=0.15, min_region_s=2.5)
    assert foreign == []


def test_crop_f32_spans():
    from plauder import audio as audio_utils
    seg = np.concatenate([np.full(SR, 0.5, np.float32),
                          np.full(SR, -0.5, np.float32)])
    out = audio_utils.crop_f32_spans(seg.tobytes(), SR, [(0.0, 1.0)])
    arr = np.frombuffer(out, dtype=np.float32)
    assert arr.shape[0] == SR and float(arr.mean()) > 0.4
    assert audio_utils.crop_f32_spans(seg.tobytes(), SR, []) == b""
