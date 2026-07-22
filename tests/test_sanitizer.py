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
    # Genuine speech CONTAINING a credit-roll phrase mid-sentence passes
    # (prefix rule): only utterances that ARE the credit roll are ghosts.
    assert not hf.is_hallucination("Kannst du Untertitel von dem Video erzeugen")
    assert not hf.is_hallucination("Er sagte please subscribe im Video")


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


def test_strip_ghost_sentences_embedded():
    """Whisper appends ghost phrases as standalone sentences at mid-segment
    pauses — prune them without touching the genuine rest."""
    hf = sanitizer.HallucinationFilter(enabled=True)
    assert hf.strip_ghost_sentences(
        "Okay, jetzt habe ich mal reingeredet. Vielen Dank. Okay, und jetzt rede ich weiter."
    ) == "Okay, jetzt habe ich mal reingeredet. Okay, und jetzt rede ich weiter."
    # Credit-roll variants with trailing year match as sentence prefix.
    assert hf.strip_ghost_sentences(
        "Das war mein Satz. Untertitelung des ZDF, 2020"
    ) == "Das war mein Satz."
    assert hf.strip_ghost_sentences(
        "Bis später dann. Tschüss. Also wie gesagt, morgen klappt."
    ) == "Bis später dann. Also wie gesagt, morgen klappt."
    # A genuine sentence CONTAINING a phrase survives (exact match only).
    assert hf.strip_ghost_sentences(
        "Ich wollte dir vielen Dank sagen. Das war nett."
    ) == "Ich wollte dir vielen Dank sagen. Das war nett."
    # Single-sentence texts are left to the whole-utterance check.
    assert hf.strip_ghost_sentences("Vielen Dank.") == "Vielen Dank."
    # All-ghost multi-sentence → unchanged here (whole-utterance path decides).
    assert hf.strip_ghost_sentences("Vielen Dank. Tschüss.") == "Vielen Dank. Tschüss."
    # Disabled filter → no-op.
    hf_off = sanitizer.HallucinationFilter(enabled=False)
    assert hf_off.strip_ghost_sentences("A. Vielen Dank. B.") == "A. Vielen Dank. B."


# --- laugh-unit splitting (audio-coupled laugh animation) --------------------
def test_split_laugh_units_isolates_tag_plus_spelled_laugh():
    from plauder.server import _split_laugh_units
    assert _split_laugh_units("[laughter] Hahaha! Und weiter.") == [
        ("[laughter] Hahaha!", True), ("Und weiter.", False)]


def test_split_laugh_units_midsentence():
    from plauder.server import _split_laugh_units
    assert _split_laugh_units("Das war lustig [laughter] wirklich.") == [
        ("Das war lustig", False), ("[laughter]", True), ("wirklich.", False)]


def test_split_laugh_units_plain_sentence_unchanged():
    from plauder.server import _split_laugh_units
    assert _split_laugh_units("Ein ganz normaler Satz.") == [
        ("Ein ganz normaler Satz.", False)]


def test_split_laugh_units_bare_soft_tag():
    from plauder.server import _split_laugh_units
    assert _split_laugh_units("*lacht* na gut.") == [
        ("*lacht*", True), ("na gut.", False)]


# --- emote-tag detection (audio-cued avatar emotes) --------------------------
def test_detect_emote_kinds_silent_and_omnivoice_tags():
    from plauder.server import _detect_emote_kinds
    # silent avatar tag + OmniVoice tag, in text order
    assert _detect_emote_kinds("Na gut [sigh], dann eben [wave] tschüss.") == [
        "sigh", "wave"]


def test_detect_emote_kinds_order_of_appearance():
    from plauder.server import _detect_emote_kinds
    assert _detect_emote_kinds("[blush] also [confirmation-en] ja.") == [
        "blush", "nod"]


def test_detect_emote_kinds_german_soft_tags_and_variants():
    from plauder.server import _detect_emote_kinds
    assert _detect_emote_kinds("*staunt* [waves-hand] [yawn]") == [
        "surprise", "wave", "sleepy"]


def test_detect_emote_kinds_plain_text_empty():
    from plauder.server import _detect_emote_kinds
    assert _detect_emote_kinds("Ein ganz normaler Satz ohne alles.") == []


def test_detect_emote_kinds_ignores_laughter():
    # laughter has its own audio-coupled path (_split_laugh_units) and must
    # NOT additionally produce a start-only mark
    from plauder.server import _detect_emote_kinds
    assert _detect_emote_kinds("[laughter] Hahaha!") == []
