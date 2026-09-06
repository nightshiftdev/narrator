"""Text normalisation and sentence segmentation.

A surprising share of the "robot" feeling comes from here, not from the voice:
a model that says "eighteen fourty-eight" as "one thousand eight hundred and
forty-eight", or reads "Dr." as "dee arr", breaks the spell instantly.
"""
from __future__ import annotations

import re

from .document import Block, Document, Sentence

# --------------------------------------------------------------------- numbers

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = [(1_000_000_000_000, "trillion"), (1_000_000_000, "billion"),
           (1_000_000, "million"), (1_000, "thousand"), (100, "hundred")]


def spell_int(n: int) -> str:
    if n < 0:
        return "minus " + spell_int(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + ("-" + _ONES[r] if r else "")
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            out = f"{spell_int(head)} {name}"
            if rest:
                joiner = " and " if rest < 100 else " "
                out += joiner + spell_int(rest)
            return out
    return str(n)


def spell_year(n: int) -> str:
    """1969 -> 'nineteen sixty-nine', 2007 -> 'two thousand seven'."""
    if 1100 <= n <= 1999 or 2010 <= n <= 2099:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return f"{spell_int(hi)} hundred"
        if lo < 10:
            return f"{spell_int(hi)} oh {spell_int(lo)}"
        return f"{spell_int(hi)} {spell_int(lo)}"
    return spell_int(n)


def spell_ordinal(n: int) -> str:
    words = spell_int(n)
    irregular = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                 "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
    head, _, tail = words.rpartition(" ")
    last = tail.rpartition("-")[2] or tail
    if last in irregular:
        rep = irregular[last]
    elif last.endswith("y"):
        rep = last[:-1] + "ieth"
    else:
        rep = last + "th"
    return words[: len(words) - len(last)] + rep


def spell_decimal(s: str) -> str:
    whole, _, frac = s.partition(".")
    out = spell_int(int(whole or 0))
    if frac:
        out += " point " + " ".join(_ONES[int(d)] for d in frac)
    return out


# ----------------------------------------------------------------- expansions

ABBREV = {
    "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss", "Dr.": "Doctor",
    "Prof.": "Professor", "St.": "Saint", "Mt.": "Mount", "Jr.": "Junior",
    "Sr.": "Senior", "Rev.": "Reverend", "Gen.": "General", "Capt.": "Captain",
    "Sgt.": "Sergeant", "Lt.": "Lieutenant", "Hon.": "Honourable",
}

INLINE = {
    "e.g.": "for example", "i.e.": "that is", "etc.": "and so on",
    "vs.": "versus", "cf.": "compare", "approx.": "approximately",
    "&": " and ", "%": " percent", "×": " times ", "÷": " divided by ",
    "→": " leads to ", "≈": " roughly ", "≤": " at most ", "≥": " at least ",
    "±": " plus or minus ",
    "™": "", "®": "", "©": "copyright ", "…": "...", "–": "—",
}

CURRENCY = {"$": ("dollars", "dollar"), "£": ("pounds", "pound"), "€": ("euros", "euro")}
MAGNITUDE = {"k": "thousand", "m": "million", "b": "billion", "bn": "billion",
             "t": "trillion", "tn": "trillion"}



# Initialisms are deliberately left alone. espeak already reads "AI" as
# /ˌeɪˈaɪ/, "FBI" as /ˌɛfbˌiːˈaɪ/ and "NASA" as /nˈæsɐ/, and it reads an
# all-capital name exactly as it reads the same name in title case. Every
# attempt to help made it worse: spelling with periods put a full stop
# between the letters, and spelling letter *names* ("ay eye") is phonemised
# as "eye eye", because "ay" is /aɪ/ rather than /eɪ/.


def normalise(text: str) -> str:
    s = text

    # strip bare URLs and emails — reading them aloud is unbearable
    s = re.sub(r"https?://\S+", "a link", s)
    s = re.sub(r"\bwww\.\S+", "a link", s)
    s = re.sub(r"\b[\w.+-]+@[\w-]+\.\w+\b", "an email address", s)

    # academic citation noise: (Smith et al., 2019), [12]
    s = re.sub(r"\((?:[A-Z][\w'-]+(?:\s+(?:et al\.?|and|&)\s*)?[\w'-]*,?\s*)+\d{4}[a-z]?\)", "", s)
    s = re.sub(r"\[\d+(?:\s*[,–-]\s*\d+)*\]", "", s)

    for k, v in ABBREV.items():
        s = s.replace(k, v)
    # currency:  $4.2M  -> four point two million dollars
    def _money(m: re.Match) -> str:
        sym, num, mag = m.group(1), m.group(2).replace(",", ""), (m.group(3) or "").lower()
        plural, singular = CURRENCY[sym]
        spoken = spell_decimal(num) if "." in num else spell_int(int(num))
        if mag:
            return f"{spoken} {MAGNITUDE[mag]} {plural}"
        return f"{spoken} {singular if num in ('1', '1.0') else plural}"

    s = re.sub(r"([$£€])\s?(\d[\d,]*(?:\.\d+)?)(?:\s*(bn|tn|[kmbt]))?\b", _money, s, flags=re.I)

    for k, v in INLINE.items():
        s = re.sub(re.escape(k), v, s, flags=re.I if k.isalpha() or "." in k else 0)

    # temperatures before the generic degree sign
    s = re.sub(r"\s*°\s*C\b", " degrees Celsius", s)
    s = re.sub(r"\s*°\s*F\b", " degrees Fahrenheit", s)
    s = re.sub(r"\s*°", " degrees", s)

    # ranges and scores:  1939-45, 3-2
    s = re.sub(r"\b(\d{4})\s*[–-]\s*(\d{2,4})\b",
               lambda m: f"{spell_year(int(m.group(1)))} to {spell_year(int(m.group(2)))}"
               if len(m.group(2)) == 4 else f"{spell_year(int(m.group(1)))} to {spell_int(int(m.group(2)))}", s)

    # times
    s = re.sub(r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?",
               lambda m: f"{spell_int(int(m.group(1)))} "
                         + ("" if m.group(2) == "00" else
                            ("oh " + spell_int(int(m.group(2))) if int(m.group(2)) < 10
                             else spell_int(int(m.group(2)))))
                         + (" " + m.group(3).replace(".", "").upper()[0] + "." + m.group(3).replace(".", "").upper()[1] + "." if m.group(3) else ""),
               s, flags=re.I)

    # ordinals: 21st, 3rd
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: spell_ordinal(int(m.group(1))), s)

    # percentages already handled by INLINE '%'; now plain numbers
    def _num(m: re.Match) -> str:
        raw = m.group(0)
        clean = raw.replace(",", "")
        if "." in clean:
            return spell_decimal(clean)
        n = int(clean)
        # a bare 4-digit number in prose is nearly always a year
        if 1000 <= n <= 2999 and "," not in raw:
            return spell_year(n)
        return spell_int(n)

    s = re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", _num, s)

    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    # letter-spelling an initialism leaves "A. P. I.." at a sentence end
    s = re.sub(r"(?<!\.)\.\.(?!\.)", ".", s)
    return s.strip()


# ------------------------------------------------------------------ sentences

# Abbreviations that never end a sentence.
_ABBREV_GUARD = re.compile(r"\b(?:[A-Z]|[Ee]\.g|[Ii]\.e|vs|etc|approx|al)\.$")
# These end a sentence unless a number follows: "No. 5", "Fig. 2", "Ch. 4".
_NUMBERED_GUARD = re.compile(r"\b(?:Fig|No|Vol|Ch|pp|Sec|Art)\.$")
# A boundary keeps any closing quote or bracket with the sentence it ends.
_BOUNDARY = re.compile(r'(?<=[.!?])(?<!\.\.\.)["\u201d\u2019\')\]]*\s+')


def split_sentences(text: str) -> list[str]:
    """Sentence split that doesn't trip on abbreviations, decimals or dialogue."""
    parts: list[str] = []
    last = 0
    for m in _BOUNDARY.finditer(text):
        parts.append(text[last:m.start()] + m.group(0).strip())
        last = m.end()
    parts.append(text[last:])

    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        merge = False
        if out:
            prev = out[-1]
            if _ABBREV_GUARD.search(prev) or len(part) < 3:
                merge = True
            elif _NUMBERED_GUARD.search(prev) and not part[:1].isupper():
                merge = True
            elif part[:1].islower():
                # a sentence does not begin in lower case; this is a continuation
                merge = True
        if merge:
            out[-1] = out[-1] + " " + part
        else:
            out.append(part)

    # Very long sentences are hard for TTS to shape; break at clause boundaries.
    final: list[str] = []
    for s in out:
        if len(s) <= 300:
            final.append(s)
            continue
        chunks = re.split(r"(?<=[,;:])\s+(?=(?:and|but|or|which|who|that|while|because|although|however)\b)", s)
        buf = ""
        for c in chunks:
            if len(buf) + len(c) < 260:
                buf = (buf + " " + c).strip()
            else:
                if buf:
                    final.append(buf)
                buf = c
        if buf:
            final.append(buf)
    return final


_MONTHS = ("January|February|March|April|May|June|July|August|September|"
           "October|November|December")
# "MARLOW April 1997 The rain had not stopped…" — a POV or scene header that
# the PDF laid out on its own line and the text layer ran into the prose.
_SCENE_HEADER = re.compile(
    rf"^((?:[A-Z][A-Z'\-]+)(?:\s+[A-Z][A-Z'\-]+){{0,3}})"      # NAME / NAME NAME
    rf"(\s+(?:{_MONTHS})\s+\d{{4}}|\s+\d{{4}})?"             # optional date
    rf"\s+(?=[A-Z][a-z])"                                       # prose follows
)


def split_scene_header(text: str) -> tuple[str, str]:
    """Peel a run-together scene header off the front of a block."""
    m = _SCENE_HEADER.match(text)
    if not m:
        return "", text
    header = (m.group(1) + (m.group(2) or "")).strip()
    body = text[m.end():]
    # Only a header if what follows is substantial prose.
    if len(body.split()) < 6:
        return "", text
    return header, body


# A typographic scene break. It is not read; it is held.
_DIVIDER = re.compile(r"^\s*(?:[*#~·•\-—_]\s*){3,}\s*")


def strip_divider(text: str) -> tuple[bool, str]:
    m = _DIVIDER.match(text)
    if not m or not text[m.end():].strip():
        return False, text
    return True, text[m.end():]


# ------------------------------------------------------------------ dialogue

_QUOTES = '"\u201c\u201d'
# he said, she asked, Ruth whispered, said Marlow — the subordinate half
_TAG = re.compile(
    r"^\s*(?:[A-Z][\w'-]*\s+)?"
    r"(?:said|says|asked|asks|replied|answered|whispered|shouted|muttered|"
    r"added|continued|repeated|offered|admitted|snapped|called|told|breathed|"
    r"echoed|murmured|insisted|observed|noted|began|finished)\b"
    r"[^.!?]*[.!?]?\s*$", re.I)


def _looks_like_tag(fragment: str) -> bool:
    f = fragment.strip().strip(",")
    if not f or len(f.split()) > 6:
        return False
    return bool(_TAG.match(f))


def split_dialogue(text: str, carry_open: bool) -> tuple[list[tuple[str, str]], bool]:
    """Split a sentence into (role, text) parts and report the quote state.

    A straight double quote is both the opening and the closing mark, so
    quotes are tracked as a toggle rather than as a matched pair, and the
    open state is carried between sentences - a speech turn often runs
    across several.

    Only a sentence that opens a quote (or continues one) counts as speech;
    a quoted phrase inside a clause is a quotation within narration, not a
    line someone says, and keeps the narrator's voice.
    """
    stripped = text.lstrip()
    opens_here = bool(stripped) and stripped[0] in _QUOTES
    if not carry_open and not opens_here:
        return [("narration", text)], False

    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    open_now = carry_open
    role = "speech" if carry_open else "narration"

    for ch in text:
        if ch in _QUOTES:
            if buf:
                parts.append((role, "".join(buf).strip()))
            buf = []
            open_now = not open_now
            role = "speech" if open_now else "after"
            continue
        buf.append(ch)
    if buf:
        parts.append((role, "".join(buf).strip()))

    out: list[tuple[str, str]] = []
    for r, t in parts:
        if not t:
            continue
        if r == "after":
            r = "tag" if _looks_like_tag(t) else "narration"
        out.append((r, t))
    return (out or [("narration", text)]), open_now


def segment(doc: Document, *, read_code: bool = False) -> list[Sentence]:
    """Document -> flat list of Sentences, normalised and ready to direct."""
    sentences: list[Sentence] = []
    for b in doc.blocks:
        if b.kind == "code":
            if not read_code:
                continue
            body = "Code block omitted."
        else:
            body = normalise(b.text)
        if not body:
            continue

        # A scene divider belongs to the silence before it, not to the voice.
        had_break, body = strip_divider(body)
        if had_break and sentences:
            sentences[-1].scene_break = True
        if not body.strip():
            continue

        if b.is_heading:
            sentences.append(Sentence(body, b.index, b.kind))
        elif (header := split_scene_header(b.text))[0]:
            head, rest = header
            sentences.append(Sentence(normalise(head), b.index, "heading"))
            for x in split_sentences(normalise(rest)):
                inner_break, x = strip_divider(x)
                if inner_break and sentences:
                    sentences[-1].scene_break = True
                if x.strip():
                    sentences.append(Sentence(x, b.index, b.kind))
        else:
            open_quote = False
            for s in split_sentences(body):
                # a divider can also land mid-block, when the PDF text layer
                # ran the break line into the surrounding paragraph
                inner_break, s = strip_divider(s)
                if inner_break and sentences:
                    sentences[-1].scene_break = True
                if not s.strip():
                    continue
                parts, open_quote = split_dialogue(s, open_quote)
                for role, part in parts:
                    sentences.append(Sentence(part, b.index, b.kind, role=role))

    for i, s in enumerate(sentences):
        s.index = i
    return sentences
