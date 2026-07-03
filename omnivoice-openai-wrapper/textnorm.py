#!/usr/bin/env python
"""German text normalisation for TTS: expand digits/dates/currency/etc. to
spoken words, because OmniVoice mis-reads bare digit strings.
Applied to the input text before synthesis. Best-effort; grammatical case
of ordinals is approximated."""
import re
from num2words import num2words

_MONTHS = {1:"Januar",2:"Februar",3:"März",4:"April",5:"Mai",6:"Juni",
           7:"Juli",8:"August",9:"September",10:"Oktober",11:"November",12:"Dezember"}
# preceding words that trigger dative/accusative ordinal ending ("am 15." -> "fünfzehnten")
# "der" excluded: usually nominative in dates ("der 3. Juli" -> "der dritte")
_DAT = {"am","dem","den","vom","zum","beim","im","seit","ab","bis","zur","des"}
_DIGIT = {"0":"null","1":"eins","2":"zwei","3":"drei","4":"vier",
          "5":"fünf","6":"sechs","7":"sieben","8":"acht","9":"neun"}


def _card(n):
    return num2words(int(n), lang="de")


def _ordinal(n, dative=False):
    w = num2words(int(n), to="ordinal", lang="de")  # e.g. "fünfzehnte"
    return w + "n" if dative else w


def _spell_digits(s):
    return " ".join(_DIGIT[c] for c in s if c in _DIGIT)


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    if not 0 <= h <= 23 or not 0 <= mi <= 59:
        return m.group(0)
    out = f"{_card(h)} Uhr"
    if mi:
        out += f" {_card(mi)}"
    return out  # trailing "Uhr" in source is consumed by the regex


def _date_num(m):
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    if y < 100:
        y += 2000
    return f"{_ordinal(d, dative=True)} {_MONTHS[mo]} {_card(y)}"


def _date_month(m):
    prev = (m.group(1) or "").strip().lower()
    d = int(m.group(2))
    month = m.group(3)
    dative = prev in _DAT
    lead = (m.group(1) or "")
    return f"{lead}{_ordinal(d, dative=dative)} {month}"


def _currency(m):
    euros = m.group(1).replace(".", "")
    cents = m.group(2)
    out = f"{_card(euros)} Euro"
    if cents:
        c = int(cents.ljust(2, "0")[:2])
        if c:
            out += f" {_card(c)}"
    return out


def _percent(m):
    num = m.group(1)
    if "," in num:
        i, dec = num.split(",")
        val = f"{_card(i)} Komma {_spell_digits(dec)}"
    else:
        val = _card(num)
    return f"{val} Prozent"


def _decimal(m):
    return f"{_card(m.group(1))} Komma {_spell_digits(m.group(2))}"


_ORD_NOM = {"der", "die", "das"}  # -> nominative "-te"


def _ordinal_trig(m):
    trig = m.group(1)
    dative = trig.lower() not in _ORD_NOM  # am/den/im/... -> "-ten"
    return f"{trig} {_ordinal(m.group(2), dative=dative)}"


def _int_grouped(m):
    return _card(m.group(0).replace(".", ""))


def _int_plain(m):
    return _card(m.group(0))


# order matters
_RULES = [
    (re.compile(r"\b(\d{1,2}):(\d{2})\b(?:\s*Uhr)?"), _time),
    (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b"), _date_num),
    (re.compile(r"(\b\w+\s+)?(\d{1,2})\.\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\b"), _date_month),
    (re.compile(r"(?:€|EUR)\s*(\d{1,3}(?:\.\d{3})*|\d+)(?:,(\d{1,2}))?", re.I), _currency),
    (re.compile(r"(\d{1,3}(?:\.\d{3})*|\d+)(?:,(\d{1,2}))?\s*(?:€|Euro|EUR)\b", re.I), _currency),
    (re.compile(r"(\d+(?:,\d+)?)\s*%"), _percent),
    (re.compile(r"\b(\d+),(\d+)\b"), _decimal),
    # ordinals ONLY after an explicit trigger word (never at a bare sentence end)
    (re.compile(r"\b(am|dem|den|der|die|das|vom|zum|im|beim|zur|des|seit|bis|ab)\s+(\d{1,2})\.(?!\d)", re.I), _ordinal_trig),
    (re.compile(r"\b\d{1,3}(?:\.\d{3})+\b"), _int_grouped),
    (re.compile(r"\d+"), _int_plain),
]


def normalize_de(text: str) -> str:
    if not text:
        return text
    for rx, fn in _RULES:
        text = rx.sub(fn, text)
    return text


if __name__ == "__main__":
    tests = [
        "Ihre Bestellung mit der Nummer 4237 wurde am 15. März zugestellt und kostet 89,90 Euro.",
        "Heute ist Freitag, der 3. Juli 2026.",
        "Der Termin ist am 15.03.2026 um 14:30 Uhr.",
        "Das Paket wiegt 3,5 kg und ist zu 50% recycelt.",
        "Es kostet €1.299,00 und wurde 1.000.000 mal verkauft.",
        "Ruf mich um 9:05 an, Zimmer 237.",
        "Die Temperatur beträgt 21,7 Grad.",
        "Wir treffen uns in Zimmer 12. Danach essen wir.",
        "Das steht in Kapitel 5. Lies es bitte.",
        "Er wurde am 21. Januar geboren und belegte den 3. Platz.",
    ]
    for t in tests:
        print("IN :", t)
        print("OUT:", normalize_de(t))
        print()
