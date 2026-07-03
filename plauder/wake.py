"""Wake-word detection via the STT transcript (prefix gate).

No dedicated model: uses the transcript that exists anyway. A segment only
counts as directed at the AI if the text begins with the wake word (filler words
like "Hey"/"Ok" before it are allowed). Everything else is discarded.

Robust against Whisper mishearings: the comparison is normalized and fuzzy
("Antonja", "Anthonia", "an Tonia" …). Pure text processing, no deps.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Strip everything except letters/digits/whitespace (incl. punctuation).
_PUNCT_RE = re.compile(r"[^0-9a-zäöüßáàâéèêíìîóòôúùûñ\s-]", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# Typical filler words before the wake word ("Hey Antonia", "Ok Antonia").
_LEAD_FILLERS = {
    "hey", "hallo", "hi", "ok", "okay", "also", "ja", "und", "äh", "ähm",
    "he", "du", "mal", "so",
}

# Stop commands that end the conversation window ("stop", "ok stopp", …).
_STOP_WORDS = {"stop", "stopp", "stoppe", "stoppen", "stoppt", "halt", "ende",
               "schluss", "fertig"}
# Filler words still allowed AFTER the stop word ("stop danke", "stop jetzt").
_TRAIL_FILLERS = {"danke", "bitte", "jetzt", "mal", "so", "ok", "okay", "ist", "gut"}


def _norm(s: str) -> str:
    """Lowercase, punctuation → whitespace, whitespace normalized."""
    s = (s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _norm_tok(t: str) -> str:
    """A single (whitespace-separated) original token → plain comparison
    string without punctuation/whitespace."""
    return _norm(t).replace(" ", "")


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _token_matches(tok: str, wake: str, fuzzy: bool, ratio: float) -> bool:
    if not tok:
        return False
    if tok == wake:
        return True
    if not fuzzy:
        return False
    # Fuzzy matching tolerates MISHEARINGS of the wake word (Antonja,
    # Anthonia) — not shorter NAMES that happen to share the stem: "Anton"
    # scores 0.83 against "antonia" and would trigger the assistant whenever
    # the household mentions an Anton. A heard token missing 2+ characters is
    # a different word, not a mishearing.
    if len(tok) < len(wake) - 1:
        return False
    if len(tok) >= 4 and len(wake) >= 4 and tok[:4] == wake[:4] and _similar(tok, wake) >= ratio * 0.92:
        return True
    return _similar(tok, wake) >= ratio


def match_wake(text: str, wake: str, *, fuzzy: bool = True, anywhere: bool = False,
               max_lead_words: int = 2, ratio: float = 0.78) -> tuple[bool, str]:
    """Checks whether ``text`` contains the wake word.

    Default (``anywhere=False``): the wake word must be at the start, optionally
    after up to ``max_lead_words`` filler words. Returns ``(matched, remainder)``
    — ``remainder`` is the original text WITHOUT the wake word (and leading
    filler words), for passing on to the LLM.
    """
    wake_n = _norm(wake).replace(" ", "")
    if not wake_n:
        return False, text
    orig = (text or "").split()
    if not orig:
        return False, text
    ntoks = [_norm_tok(t) for t in orig]
    n = len(orig)

    def remainder_from(idx_after: int) -> str:
        rest = " ".join(orig[idx_after:]).strip()
        return rest.lstrip(",.:;!?–—- ").strip()

    def lead_ok(i: int) -> bool:
        return anywhere or i == 0 or all(ntoks[k] in _LEAD_FILLERS for k in range(i))

    search_end = n if anywhere else min(n, 1 + max_lead_words)
    for i in range(search_end):
        if _token_matches(ntoks[i], wake_n, fuzzy, ratio) and lead_ok(i):
            return True, remainder_from(i + 1)
        # Pull together a split wake word ("an tonia").
        if i + 1 < n:
            joined = ntoks[i] + ntoks[i + 1]
            if _token_matches(joined, wake_n, fuzzy, ratio) and lead_ok(i):
                return True, remainder_from(i + 2)
    return False, text


def is_stop_command(text: str, *, fuzzy: bool = True, ratio: float = 0.86) -> bool:
    """True if ``text`` is (only) a stop command — e.g. "stop", "ok stopp",
    "stop danke".

    Leading/trailing filler words are stripped off; after that EXACTLY ONE stop
    word must remain. This way longer sentences that merely contain the word
    ("soll ich den Bus stoppen?") deliberately do NOT trigger. Fuzzy catches
    narrow Whisper mishearings without falsely matching short words like
    "raus"/"wende" (same initial letter + minimum length required)."""
    toks = _norm(text).split()
    while toks and toks[0] in _LEAD_FILLERS:
        toks.pop(0)
    while toks and toks[-1] in _TRAIL_FILLERS:
        toks.pop()
    if len(toks) != 1:
        return False
    t = toks[0]
    if t in _STOP_WORDS:
        return True
    if fuzzy and len(t) >= 4:
        return any(t[0] == w[0] and _similar(t, w) >= ratio
                   for w in _STOP_WORDS if len(w) >= 4)
    return False
