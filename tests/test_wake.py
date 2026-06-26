"""Wake-Word-Matching (Prefix-Gate, fuzzy)."""
import pytest

from plauder.wake import match_wake, is_stop_command


def test_exact_match_at_start_strips_word():
    ok, rest = match_wake("Antonia wie spät ist es?", "antonia")
    assert ok is True
    assert rest == "wie spät ist es?"


def test_match_with_comma_preserves_remainder_case():
    ok, rest = match_wake("Antonia, wie spät ist es?", "antonia")
    assert ok is True
    assert rest == "wie spät ist es?"


def test_leading_filler_allowed():
    ok, rest = match_wake("Hey Antonia, mach das Licht an.", "antonia")
    assert ok is True
    assert rest == "mach das Licht an."


def test_no_wake_word_is_rejected():
    ok, rest = match_wake("Wie spät ist es eigentlich?", "antonia")
    assert ok is False
    assert rest == "Wie spät ist es eigentlich?"


def test_fuzzy_mishearings_match():
    for variant in ("Antonja", "Anthonia", "Antonien"):
        ok, _ = match_wake(f"{variant} hallo", "antonia")
        assert ok is True, f"{variant!r} sollte matchen"


def test_split_wake_word_two_tokens():
    ok, rest = match_wake("An Tonia, hilf mir.", "antonia")
    assert ok is True
    assert rest == "hilf mir."


def test_word_only_in_middle_rejected_by_default():
    # Standard: nur am Anfang (nach Füllwörtern). „ist" ist kein Füllwort.
    ok, _ = match_wake("Sag mal Antonia etwas", "antonia")
    assert ok is False


def test_anywhere_mode_matches_in_middle():
    ok, rest = match_wake("Sag mal Antonia etwas", "antonia", anywhere=True)
    assert ok is True
    assert rest == "etwas"


def test_fuzzy_off_requires_exact():
    ok, _ = match_wake("Antonja hallo", "antonia", fuzzy=False)
    assert ok is False


def test_unrelated_word_not_false_positive():
    ok, _ = match_wake("Telefon klingelt", "antonia")
    assert ok is False
    ok2, _ = match_wake("Banane essen", "antonia")
    assert ok2 is False


def test_only_wake_word_yields_empty_remainder():
    ok, rest = match_wake("Antonia.", "antonia")
    assert ok is True
    assert rest == ""


# --- Stop-Kommando (beendet das Konversationsfenster) ---

@pytest.mark.parametrize("text", [
    "stop", "Stopp", "stopp.", "Halt", "Ende", "Schluss",
    "ok stop", "okay stopp", "stop danke", "stopp jetzt", "so, ende",
    "hey stopp bitte",
])
def test_stop_command_positive(text):
    assert is_stop_command(text) is True


@pytest.mark.parametrize("text", [
    "wie spät ist es?",
    "soll ich den Bus stoppen?",
    "stopp den Timer bitte",
    "ende der durchsage kommt gleich",
    "raus",          # darf NICHT fuzzy auf „aus/ende" matchen
    "wende",
    "",
])
def test_stop_command_negative(text):
    assert is_stop_command(text) is False


def test_stop_command_fuzzy_mishearing():
    # knapper Whisper-Verhörer, gleicher Anfang → noch als Stop erkannt
    assert is_stop_command("stoppp") is True
    # ohne fuzzy nicht
    assert is_stop_command("stoppp", fuzzy=False) is False
