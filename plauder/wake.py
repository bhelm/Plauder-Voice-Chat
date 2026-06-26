"""Wake-Word-Erkennung über das STT-Transkript (Prefix-Gate).

Kein eigenes Modell: nutzt das ohnehin vorhandene Transkript. Ein Segment zählt
nur dann als an die KI gerichtet, wenn der Text mit dem Wake-Word beginnt
(Füllwörter wie „Hey"/„Ok" davor sind erlaubt). Alles andere wird verworfen.

Robust gegen Whisper-Verhörer: Vergleich erfolgt normalisiert und fuzzy
(„Antonja", „Anthonia", „an Tonia" …). Reine Textverarbeitung, keine Deps.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Alles außer Buchstaben/Ziffern/Whitespace raus (inkl. Satzzeichen).
_PUNCT_RE = re.compile(r"[^0-9a-zäöüßáàâéèêíìîóòôúùûñ\s-]", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# Typische Füllwörter vor dem Wake-Word ("Hey Antonia", "Ok Antonia").
_LEAD_FILLERS = {
    "hey", "hallo", "hi", "ok", "okay", "also", "ja", "und", "äh", "ähm",
    "he", "du", "mal", "so",
}

# Stop-Kommandos, die das Konversationsfenster beenden ("stop", "ok stopp", …).
_STOP_WORDS = {"stop", "stopp", "stoppe", "stoppen", "stoppt", "halt", "ende",
               "schluss", "fertig"}
# Füllwörter, die NACH dem Stop-Wort noch erlaubt sind ("stop danke", "stop jetzt").
_TRAIL_FILLERS = {"danke", "bitte", "jetzt", "mal", "so", "ok", "okay", "ist", "gut"}


def _norm(s: str) -> str:
    """Kleinschreibung, Satzzeichen → Whitespace, Whitespace normalisiert."""
    s = (s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _norm_tok(t: str) -> str:
    """Ein einzelnes (Whitespace-getrenntes) Original-Token → reiner Vergleichs-
    string ohne Satzzeichen/Whitespace."""
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
    if len(tok) >= 4 and len(wake) >= 4 and tok[:4] == wake[:4] and _similar(tok, wake) >= ratio * 0.92:
        return True
    return _similar(tok, wake) >= ratio


def match_wake(text: str, wake: str, *, fuzzy: bool = True, anywhere: bool = False,
               max_lead_words: int = 2, ratio: float = 0.78) -> tuple[bool, str]:
    """Prüft, ob ``text`` das Wake-Word enthält.

    Standard (``anywhere=False``): das Wake-Word muss am Anfang stehen, optional
    nach bis zu ``max_lead_words`` Füllwörtern. Gibt ``(matched, remainder)``
    zurück — ``remainder`` ist der Originaltext OHNE Wake-Word (und führende
    Füllwörter), zum Weiterreichen an das LLM.
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
        # Zweigeteiltes Wake-Word ("an tonia") zusammenziehen.
        if i + 1 < n:
            joined = ntoks[i] + ntoks[i + 1]
            if _token_matches(joined, wake_n, fuzzy, ratio) and lead_ok(i):
                return True, remainder_from(i + 2)
    return False, text


def is_stop_command(text: str, *, fuzzy: bool = True, ratio: float = 0.86) -> bool:
    """True, wenn ``text`` (nur) ein Stop-Kommando ist — z.B. „stop", „ok stopp",
    „stop danke".

    Führende/abschließende Füllwörter werden abgezogen; danach muss GENAU EIN
    Stop-Wort übrig bleiben. So lösen längere Sätze, die das Wort nur enthalten
    („soll ich den Bus stoppen?"), bewusst NICHT aus. Fuzzy fängt knappe
    Whisper-Verhörer, ohne Kurzwörter wie „raus"/„wende" fälschlich zu treffen
    (gleicher Anfangsbuchstabe + Mindestlänge gefordert)."""
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
