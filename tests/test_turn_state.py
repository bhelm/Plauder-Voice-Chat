"""TurnState + VAD parameters (debounce/coalescing)."""
from plauder.turn_state import (
    VAD_REDEMPTION_MAX,
    VAD_REDEMPTION_MIN,
    TurnState,
    vad_params_for_debounce,
)


def test_fresh_state_has_no_pending():
    s = TurnState()
    assert not s.has_pending()


def test_pending_voice_text_images():
    s = TurnState()
    s.pending_texts.append("hallo")
    assert s.has_pending()
    s2 = TurnState()
    s2.pending_text_parts.append("tippe")
    assert s2.has_pending()
    s3 = TurnState()
    s3.pending_image_urls.append("/uploads/x.png")
    assert s3.has_pending()


def test_reset_clears_pending_and_rotates_turn_id():
    s = TurnState()
    old = s.turn_id
    s.pending_texts.append("a")
    s.pending_segment_ids.append("seg")
    s.pending_text_parts.append("b")
    s.pending_image_urls.append("u")
    s.reset()
    assert not s.has_pending()
    assert s.turn_id != old


def test_vad_params_scale_with_debounce():
    low = vad_params_for_debounce(300)
    high = vad_params_for_debounce(5000)
    assert low["redemptionFrames"] >= VAD_REDEMPTION_MIN
    assert high["redemptionFrames"] <= VAD_REDEMPTION_MAX
    assert high["redemptionFrames"] > low["redemptionFrames"]


def test_vad_params_clamped():
    assert vad_params_for_debounce(10)["redemptionFrames"] == VAD_REDEMPTION_MIN
    assert vad_params_for_debounce(999999)["redemptionFrames"] == VAD_REDEMPTION_MAX
    assert vad_params_for_debounce(1200)["minSpeechFrames"] == 3
