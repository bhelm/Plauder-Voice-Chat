"""Wake-word matching (prefix gate, fuzzy)."""
import pytest

from plauder.wake import match_wake, is_stop_command, strip_end_marker


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
    # Default: only at the start (after filler words). "ist" is not a filler word.
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


# --- Stop command (ends the conversation window) ---

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
    "raus",          # must NOT fuzzy-match "aus/ende"
    "wende",
    "",
])
def test_stop_command_negative(text):
    assert is_stop_command(text) is False


def test_stop_command_fuzzy_mishearing():
    # slight Whisper mishearing, same start → still recognized as stop
    assert is_stop_command("stoppp") is True
    # not without fuzzy
    assert is_stop_command("stoppp", fuzzy=False) is False


# --- Trailing end marker ("… Ende"/"… End" closes the window) ---

@pytest.mark.parametrize("text,rest", [
    ("mach das Licht aus, Ende", "mach das Licht aus"),
    ("wie spät ist es? End", "wie spät ist es?"),
    ("danke ENDE!", "danke"),
])
def test_end_marker_stripped_from_tail(text, rest):
    assert strip_end_marker(text) == (rest, True)


@pytest.mark.parametrize("text", ["Ende", "End.", "ENDE"])
def test_end_marker_alone_leaves_empty_rest(text):
    got_rest, ended = strip_end_marker(text)
    assert ended is True and got_rest == ""


@pytest.mark.parametrize("text", [
    "das Ende vom Film war gut",     # marker mid-sentence → no trigger
    "wie spät ist es?",
    "wende",                          # no fuzzy on the single trailing word
    "am Wochenende",
    "",
])
def test_end_marker_negative(text):
    assert strip_end_marker(text) == (text, False)


def test_shorter_name_does_not_match_wake_word():
    """Regression: "Anton" (a real name, 2 chars short) scores 0.83 against
    "antonia" and used to trigger the assistant. Mishearings of roughly the
    same length (Antonja, Anthonia) still match."""
    from plauder.wake import match_wake
    assert match_wake("Anton komm her", "antonia")[0] is False
    assert match_wake("Antonja, wie spät ist es?", "antonia")[0] is True
    assert match_wake("Anthonia was geht", "antonia")[0] is True
