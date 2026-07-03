"""Sanitizer: strip emoji/Markdown, nonverbal tags, NO_REPLY, ghost filter, merge."""
from plauder import sanitizer


def test_strips_emoji():
    assert "😀" not in sanitizer.sanitize_for_tts("Hallo 😀 Welt")


def test_strips_markdown_bold_italic():
    assert sanitizer.sanitize_for_tts("**fett** und _kursiv_") == "fett und kursiv"


def test_strips_code_fence():
    out = sanitizer.sanitize_for_tts("Text ```python\nprint(1)\n``` Ende")
    assert "print(1)" not in out
    assert "code block omitted" in out


def test_link_becomes_label():
    assert sanitizer.sanitize_for_tts("[Klick hier](http://x.y)") == "Klick hier"


def test_nonverbal_alias_maps_to_tag():
    assert "[laughter]" in sanitizer.sanitize_for_tts("Haha [lacht] super")


def test_unknown_bracket_tag_removed():
    out = sanitizer.sanitize_for_tts("Hallo [irgendwas] Welt")
    assert "irgendwas" not in out


def test_is_no_reply_variants():
    assert sanitizer.is_no_reply("NO_REPLY")
    assert sanitizer.is_no_reply("  *NO_REPLY*  ")
    assert sanitizer.is_no_reply("")
    assert sanitizer.is_no_reply("No response from OpenClaw.")
    assert not sanitizer.is_no_reply("Das ist eine echte Antwort.")
    assert sanitizer.is_no_reply(None) is False


def test_merge_transcripts_inserts_period():
    out = sanitizer.merge_transcripts(["hallo", "wie gehts"])
    assert out == "hallo. Wie gehts"


def test_merge_transcripts_keeps_sentence_end():
    out = sanitizer.merge_transcripts(["Hallo!", "Wie gehts?"])
    assert out == "Hallo! Wie gehts?"


def test_merge_transcripts_empty():
    assert sanitizer.merge_transcripts([]) == ""
    assert sanitizer.merge_transcripts(["", "  "]) == ""


def test_hallucination_filter_denylist_with_high_no_speech():
    hf = sanitizer.HallucinationFilter(enabled=True, no_speech_prob_threshold=0.6)
    assert hf.is_hallucination("Thank you.", no_speech_prob=0.9)
    # Denylist hit but low no_speech_prob -> do NOT filter (conservative).
    assert not hf.is_hallucination("Thank you.", no_speech_prob=0.1)
    # Not on denylist -> never filter.
    assert not hf.is_hallucination("Das ist echt wichtig", no_speech_prob=0.99)


def test_hallucination_filter_credit_roll_always():
    # Credit-roll outros are filtered UNCONDITIONALLY (no no_speech_prob needed —
    # cloud STT reports none) and as a substring so trailing years/channels hit.
    hf = sanitizer.HallucinationFilter(enabled=True)
    assert hf.is_hallucination("Untertitel des ZDF, 2020", no_speech_prob=None)
    assert hf.is_hallucination("Untertitelung des ZDF für Funk, 2017")
    assert hf.is_hallucination("Thanks for watching!", no_speech_prob=0.0)
    assert hf.is_hallucination("Untertitel der Amara.org-Community")
    # A genuine request that merely mentions subtitles must NOT be filtered.
    assert not hf.is_hallucination("Mach mal die Untertitel an", no_speech_prob=None)


def test_hallucination_filter_extra_substring_rule():
    hf = sanitizer.HallucinationFilter(enabled=True, extra_phrases="brought to you by*")
    assert hf.is_hallucination("Brought to you by ACME Corp", no_speech_prob=None)


def test_hallucination_filter_disabled():
    hf = sanitizer.HallucinationFilter(enabled=False)
    assert not hf.is_hallucination("Thank you.", no_speech_prob=0.99)


def test_hallucination_filter_duration_signal():
    hf = sanitizer.HallucinationFilter(enabled=True, use_duration=True, max_dur_s=1.5)
    assert hf.is_hallucination("danke", no_speech_prob=None, duration_s=0.5)


def test_hallucination_filter_from_config():
    class _Cfg:
        stt_hallucination_filter = True
        stt_ghost_no_speech_prob = 0.6
        stt_ghost_use_duration = False
        stt_ghost_max_dur_s = 1.5
        stt_ghost_extra_phrases = "mein extra geist"
    hf = sanitizer.HallucinationFilter.from_config(_Cfg())
    assert "mein extra geist" in hf.phrases
