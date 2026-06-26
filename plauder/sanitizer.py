"""Text-Sanitizer: Emoji/Markdown raus, nonverbale Tags normalisieren,
Aussprache-Lexikon, NO_REPLY-Erkennung, Whisper-Halluzinations-Filter und
Transkript-Merging.

Reine Textverarbeitung — keine Netzwerk-/Backend-Abhängigkeiten.
"""

from __future__ import annotations

import json
import os
import re

# --------------------------------------------------------------------------- #
# Emoji / Markdown
# --------------------------------------------------------------------------- #
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "‍"   # ZWJ
    "️"   # VS16
    "]+",
    flags=re.UNICODE,
)
_MD_BOLD_ITALIC = re.compile(r"(\*\*|__|\*|_)(.+?)\1")
_MD_INLINE_CODE = re.compile(r"`+([^`]+)`+")
_MD_CODE_FENCE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# --------------------------------------------------------------------------- #
# Non-verbale Tags (OmniVoice-Whitelist + deutsche Aliase)
# --------------------------------------------------------------------------- #
_NONVERBAL_TAGS = {
    "laughter", "sigh", "confirmation-en", "question-en", "question-ah",
    "question-oh", "question-ei", "question-yi", "surprise-ah", "surprise-oh",
    "surprise-wa", "surprise-yo", "dissatisfaction-hnn",
}
_NONVERBAL_ALIASES = {
    "lacht": "laughter", "lachen": "laughter", "lach": "laughter",
    "haha": "laughter", "kichert": "laughter", "kichern": "laughter",
    "seufzt": "sigh", "seufz": "sigh", "seufzer": "sigh", "seufzen": "sigh",
    "überrascht": "surprise-oh", "staunt": "surprise-wa", "wow": "surprise-wa",
    "zustimmung": "confirmation-en", "stimmt zu": "confirmation-en",
    "unzufrieden": "dissatisfaction-hnn", "brummt": "dissatisfaction-hnn",
}
_BRACKET_TAG_RE = re.compile(r"\[\s*([^\[\]]{1,40}?)\s*\]")
_SOFT_TAG_RE = re.compile(r"[\*\(]\s*([a-zA-ZäöüÄÖÜß ]{2,20}?)\s*[\*\)]")


def normalize_nonverbal_tags(text: str) -> str:
    def _square(m):
        inner = m.group(1).strip().lower()
        if inner in _NONVERBAL_TAGS:
            return f"[{inner}]"
        if inner in _NONVERBAL_ALIASES:
            return f"[{_NONVERBAL_ALIASES[inner]}]"
        return ""  # unbekannter Pseudo-Tag → nicht vorlesen

    text = _BRACKET_TAG_RE.sub(_square, text)

    def _soft(m):
        inner = m.group(1).strip().lower()
        if inner in _NONVERBAL_ALIASES:
            return f"[{_NONVERBAL_ALIASES[inner]}]"
        if inner in _NONVERBAL_TAGS:
            return f"[{inner}]"
        return m.group(0)

    return _SOFT_TAG_RE.sub(_soft, text)


# --------------------------------------------------------------------------- #
# Aussprache-Lexikon
# --------------------------------------------------------------------------- #
_pron_cache: dict = {"path": None, "mtime": None, "rules": []}


def load_pronunciations(path: str | None):
    """Lädt das Aussprache-Lexikon (JSON {wort: ersatz}). Cacht nach mtime."""
    if not path:
        return []
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return []
    if _pron_cache["path"] != path or _pron_cache["mtime"] != mtime:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            rules = [
                (re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v)
                for k, v in data.items()
                if isinstance(v, str) and not k.startswith("_")
            ]
            _pron_cache.update(path=path, mtime=mtime, rules=rules)
        except Exception:
            _pron_cache.update(path=path, mtime=mtime, rules=[])
    return _pron_cache["rules"]


def sanitize_for_tts(text: str, *, pronunciations_file: str | None = None) -> str:
    """Macht LLM-Text TTS-tauglich: Code/Links/Markdown/Emojis raus, nonverbale
    Tags normalisiert, Aussprache-Korrekturen, Whitespace normalisiert.
    """
    if not text:
        return text
    text = _MD_CODE_FENCE.sub(" (Code-Block weggelassen) ", text)
    text = _MD_INLINE_CODE.sub(lambda m: m.group(1), text)
    text = _MD_LINK.sub(lambda m: m.group(1), text)
    text = re.sub(r"[\U0001F602\U0001F923]+", " [laughter] ", text)
    text = normalize_nonverbal_tags(text)
    text = _MD_BOLD_ITALIC.sub(lambda m: m.group(2), text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    for _rx, _repl in load_pronunciations(pronunciations_file):
        text = _rx.sub(_repl, text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# NO_REPLY-Erkennung
# --------------------------------------------------------------------------- #
NO_REPLY_TOKEN = "NO_REPLY"
_NO_REPLY_RE = re.compile(
    r"^\s*[\*_`\"'\(\[]*\s*NO_REPLY\s*[\*_`\"'\)\]]*\s*[.!]?\s*$", re.IGNORECASE)
_OPENCLAW_FALLBACK_RE = re.compile(
    r"^\s*(no response from openclaw\.?|no response generated\.?"
    r"( please try again\.?)?|no response\.?)\s*$",
    re.IGNORECASE,
)


def is_no_reply(text: str) -> bool:
    """True, wenn der LLM-Output als „keine Antwort“ zu werten ist."""
    if text is None:
        return False
    t = text.strip()
    if not t:
        return True
    if _NO_REPLY_RE.match(t):
        return True
    if _OPENCLAW_FALLBACK_RE.match(t):
        return True
    return False


# --------------------------------------------------------------------------- #
# Whisper-Halluzinations-Filter ("Thank you"-Geister)
# --------------------------------------------------------------------------- #
def normalize_ghost(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"^[\s\.\,\!\?…\-—–\"'»«„“”]+", "", t)
    t = re.sub(r"[\s\.\,\!\?…\-—–\"'»«„“”]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


_GHOST_PHRASES_BASE = {
    "thank you", "thank you.", "thanks", "thank you very much",
    "thanks for watching", "thanks for watching!",
    "thank you for watching", "thank you for watching!",
    "please subscribe", "like and subscribe",
    "see you next time", "see you in the next video",
    "bye", "bye bye", "goodbye", "you",
    "vielen dank", "danke", "dankeschön", "danke schön",
    "vielen dank fürs zuschauen", "vielen dank fürs zuschauen!",
    "danke fürs zuschauen", "bis zum nächsten mal",
    "untertitel von", "untertitel im auftrag des zdf",
    "untertitelung des zdf", "untertitelung des zdf, 2020",
    "untertitel der amara.org-community", "amara.org",
    "tschüss", "auf wiedersehen",
}


class HallucinationFilter:
    """Verwirft Whisper-Geister-Floskeln. Konservativ: Denylist-Treffer UND
    (no_speech_prob hoch [ODER optional kurze Audio-Dauer]).
    Aus Config gebaut, damit es testbar bleibt.
    """

    def __init__(self, *, enabled: bool = True, no_speech_prob_threshold: float = 0.6,
                 use_duration: bool = False, max_dur_s: float = 1.5,
                 extra_phrases: str = ""):
        self.enabled = enabled
        self.no_speech_prob_threshold = no_speech_prob_threshold
        self.use_duration = use_duration
        self.max_dur_s = max_dur_s
        phrases = {normalize_ghost(p) for p in _GHOST_PHRASES_BASE if p.strip()}
        if extra_phrases:
            for p in extra_phrases.split("|"):
                n = normalize_ghost(p)
                if n:
                    phrases.add(n)
        self.phrases = phrases

    @classmethod
    def from_config(cls, cfg) -> "HallucinationFilter":
        return cls(
            enabled=cfg.stt_hallucination_filter,
            no_speech_prob_threshold=cfg.stt_ghost_no_speech_prob,
            use_duration=cfg.stt_ghost_use_duration,
            max_dur_s=cfg.stt_ghost_max_dur_s,
            extra_phrases=cfg.stt_ghost_extra_phrases,
        )

    def is_hallucination(self, text: str, *, no_speech_prob=None, duration_s=None) -> bool:
        if not self.enabled:
            return False
        norm = normalize_ghost(text)
        if not norm or norm not in self.phrases:
            return False
        nsp = no_speech_prob if isinstance(no_speech_prob, (int, float)) else None
        high_no_speech = (nsp is not None) and (nsp > self.no_speech_prob_threshold)
        short_audio = (
            self.use_duration
            and isinstance(duration_s, (int, float))
            and duration_s < self.max_dur_s
        )
        return bool(high_no_speech or short_audio)


# --------------------------------------------------------------------------- #
# Transkript-Merging
# --------------------------------------------------------------------------- #
def merge_transcripts(parts: list[str]) -> str:
    """Klebt Transkript-Teile zusammen; fügt einen Punkt ein, wenn das
    Vorgänger-Stück nicht mit Satzzeichen endete.
    """
    pieces = [p.strip() for p in parts if p and p.strip()]
    if not pieces:
        return ""
    out = pieces[0]
    for nxt in pieces[1:]:
        if out and out[-1] in ".?!…":
            out = out + " " + nxt
        elif out and out[-1] in ",:;":
            out = out + " " + nxt
        else:
            out = out + ". " + (nxt[0].upper() + nxt[1:] if nxt else nxt)
    return _MULTI_SPACE.sub(" ", out).strip()
