"""Text sanitizer: strip emoji/markdown, normalize nonverbal tags,
pronunciation lexicon, NO_REPLY detection, Whisper hallucination filter and
transcript merging.

Pure text processing — no network/backend dependencies.
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
    "\U0001F300-\U0001FAFF"   # already covers 1F6xx/1F7xx/1F9xx/1FAxx
    "\U00002600-\U000027BF"
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
# Nonverbal tags (OmniVoice whitelist + German aliases)
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
        return ""  # unknown pseudo-tag → do not read aloud

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
# Pronunciation lexicon
# --------------------------------------------------------------------------- #
_pron_cache: dict = {"path": None, "mtime": None, "rules": []}


def load_pronunciations(path: str | None):
    """Loads the pronunciation lexicon (JSON {word: replacement}). Cached by mtime."""
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
    """Makes LLM text TTS-suitable: strip code/links/markdown/emoji, normalize
    nonverbal tags, pronunciation corrections, normalize whitespace.
    """
    if not text:
        return text
    text = _MD_CODE_FENCE.sub(" (code block omitted) ", text)
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
# NO_REPLY detection
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
    """True if the LLM output is to be treated as 'no reply'."""
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


def is_no_reply_prefix(text: str) -> bool:
    """True while the streamed text SO FAR could still become a pure NO_REPLY.

    The streaming path must hold back reply.delta/TTS until this returns False
    — checking only ``is_no_reply`` lets a partial token leak ("NO_" is not a
    complete NO_REPLY, gets emitted, then the reply turns silent)."""
    t = (text or "").lstrip().lstrip("*_`\"'([").lstrip()
    if len(t) <= len(NO_REPLY_TOKEN):
        return NO_REPLY_TOKEN.startswith(t.upper())
    return is_no_reply(text)


# --------------------------------------------------------------------------- #
# Whisper hallucination filter ("Thank you" ghosts)
# --------------------------------------------------------------------------- #
def normalize_ghost(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"^[\s\.\,\!\?…\-—–\"'»«„“”]+", "", t)
    t = re.sub(r"[\s\.\,\!\?…\-—–\"'»«„“”]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


_GHOST_PHRASES_BASE = {
    "thank you", "thank you.", "thanks", "thank you very much",
    "see you next time", "see you in the next video",
    "bye", "bye bye", "goodbye", "you",
    "vielen dank", "danke", "dankeschön", "danke schön",
    "danke fürs zuschauen", "bis zum nächsten mal",
    "tschüss", "auf wiedersehen",
}

# Credit-roll / channel-outro hallucinations. Whisper emits these on silence or
# low-level background noise (TV, YouTube playing nearby). They NEVER occur as
# genuine speech directed at a voice assistant, so they are filtered
# UNCONDITIONALLY (no no_speech_prob / duration corroboration needed — important
# because cloud STT backends do not report no_speech_prob at all). Matched as a
# PREFIX of the normalized transcript so trailing years / channel names still
# hit ("Untertitel des ZDF, 2020") without swallowing genuine speech that only
# contains the phrase mid-sentence.
_GHOST_ALWAYS_SUBSTR_BASE = (
    "untertitel des zdf", "untertitelung des zdf", "untertitel von",
    "untertitel im auftrag", "untertitel der amara", "amara.org",
    "thanks for watching", "thank you for watching",
    "please subscribe", "like and subscribe", "subtitles by",
    "vielen dank fürs zuschauen", "danke fürs zuschauen und",
)


class HallucinationFilter:
    """Discards Whisper ghost phrases. Two tiers:

    * Credit-roll / outro phrases (``_GHOST_ALWAYS_SUBSTR_BASE``) are filtered
      UNCONDITIONALLY — matched as a prefix so trailing years/channel names
      still hit. As whole utterances these never occur as genuine speech, and
      cloud STT backends report no ``no_speech_prob``, so gating them would
      disable the filter.
    * Ambiguous short phrases (``_GHOST_PHRASES_BASE``: "danke", "bye", …) stay
      conservative: exact denylist hit AND (no_speech_prob high OR short audio).

    Built from Config so it stays testable.
    """

    def __init__(self, *, enabled: bool = True, no_speech_prob_threshold: float = 0.6,
                 use_duration: bool = False, max_dur_s: float = 1.5,
                 extra_phrases: str = ""):
        self.enabled = enabled
        self.no_speech_prob_threshold = no_speech_prob_threshold
        self.use_duration = use_duration
        self.max_dur_s = max_dur_s
        phrases = {normalize_ghost(p) for p in _GHOST_PHRASES_BASE if p.strip()}
        always = [normalize_ghost(p) for p in _GHOST_ALWAYS_SUBSTR_BASE if p.strip()]
        if extra_phrases:
            # An extra phrase ending in "*" is an unconditional substring rule;
            # otherwise it joins the conservative exact-match denylist.
            for p in extra_phrases.split("|"):
                if p.strip().endswith("*"):
                    n = normalize_ghost(p.strip()[:-1])
                    if n:
                        always.append(n)
                else:
                    n = normalize_ghost(p)
                    if n:
                        phrases.add(n)
        self.phrases = phrases
        self.always_substr = [a for a in always if a]

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
        if not norm:
            return False
        # Tier 1: credit-roll phrases → always ghost, no corroboration needed.
        # PREFIX match: a hallucinated credit roll IS the whole utterance
        # ("Untertitel von Stephanie Geiges", "Untertitel des ZDF, 2020"), so
        # trailing names/years still hit — but genuine speech that merely
        # CONTAINS the phrase ("Kannst du Untertitel von dem Video erzeugen")
        # is not swallowed.
        if any(norm.startswith(sub) for sub in self.always_substr):
            return True
        # Tier 2: ambiguous short phrases → need a corroborating signal.
        if norm not in self.phrases:
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
# Transcript merging
# --------------------------------------------------------------------------- #
def merge_transcripts(parts: list[str]) -> str:
    """Glues transcript pieces together; inserts a period when the preceding
    piece did not end with a punctuation mark.
    """
    pieces = [p.strip() for p in parts if p and p.strip()]
    if not pieces:
        return ""
    out = pieces[0]
    for nxt in pieces[1:]:
        if out and out[-1] in ".?!…,:;":
            out = out + " " + nxt
        else:
            out = out + ". " + (nxt[0].upper() + nxt[1:] if nxt else nxt)
    return _MULTI_SPACE.sub(" ", out).strip()
