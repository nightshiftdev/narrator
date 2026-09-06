"""Casting: who says each line, and in whose voice.

Two problems, solved separately.

*Attribution* is the hard one. Novels mostly do not write "she said" — this
project's test chapter has 44 spoken lines and 2 attribution tags — so the
speaker has to be inferred from context and from the alternation of turns.
That is an LLM job, and a local 14B model does it well when it is given the
character roster up front and asked about one window at a time.

*Casting* is then bookkeeping: map each character to a voice, keep it stable
for the whole book, and never reuse the narrator's voice for someone else.

The governing rule is that a wrong voice is worse than one voice. Anything
the model is not sure about falls back to the narrator, because under-casting
is invisible and mis-casting is jarring.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field

from .director import OLLAMA_URL, DirectorConfig, _api, _claude_cli
from .document import Sentence

NARRATOR = "NARRATOR"

# Distinct, long-form-friendly voices, ordered so the first picks are the most
# different from one another.
MALE_POOL = ["am_michael", "am_onyx", "bm_daniel", "am_liam", "bm_fable",
             "am_adam", "am_echo", "am_puck"]
FEMALE_POOL = ["af_heart", "af_nicole", "bf_emma", "af_aoede", "bf_lily",
               "af_sky", "af_kore", "af_river"]

# A pronoun is not a character. The model will happily return "She" if the
# sample it saw never named her.
PRONOUNS = {"HE", "SHE", "IT", "THEY", "I", "YOU", "WE", "ME", "HIM", "HER",
            "THEM", "US", "SOMEONE", "ANYONE", "NOBODY", "EVERYONE", "MAN",
            "WOMAN", "BOY", "GIRL", "THE NARRATOR", "NARRATOR", "UNKNOWN"}

ROSTER_SYSTEM = """You are casting an audiobook.

From the passage, list the characters who speak aloud, and the point-of-view
narrators (a POV narrator is whoever says "I"; scene headers in capitals often
name them).

Return ONLY JSON:
{"characters":[{"name":"<as written>","gender":"male|female|unknown",
                "age":"adult|child","is_narrator":true|false}]}

Only list people who actually speak or narrate. Do not invent characters.
Give each character's NAME as written in the text ("Sara", "Mira"). Never
return a pronoun such as "She" or "The woman" — if a speaker is never named,
omit them entirely rather than naming them by pronoun."""

ATTRIB_SYSTEM = """You are casting an audiobook.

You receive numbered lines from a novel in order. Lines marked (speech) are
spoken aloud, with the quote marks already removed. Decide who utters each
one.

Return ONLY JSON: {"speakers":[{"i":<line number>,"who":"<name>"}]}

Rules:
- Include an entry for EVERY (speech) line you are given, including one-word
  lines like "Okay." or "No." — those are turns in the exchange and matter
  most for getting the alternation right.
- Use a name from the roster, or NARRATOR when the point-of-view character is
  speaking aloud.
- Two consecutive speech lines are usually different speakers taking turns,
  but not always: one person can say several sentences in a row.
- Use UNKNOWN when you genuinely cannot tell. That is better than guessing;
  an unknown line is read by the narrator, which is never jarring."""


@dataclass
class Character:
    name: str
    gender: str = "unknown"
    age: str = "adult"
    is_narrator: bool = False
    voice: str = ""


@dataclass
class Cast:
    characters: dict[str, Character] = field(default_factory=dict)
    narrator_voice: str = ""

    def voice_for(self, sentence: Sentence) -> str:
        """The voice this line should be read in, or "" for the narrator."""
        who = sentence.speaker
        if not who or who in (NARRATOR, "UNKNOWN"):
            return ""
        c = self.characters.get(who.upper())
        return c.voice if c and c.voice else ""

    def summary(self) -> list[str]:
        out = []
        for c in sorted(self.characters.values(), key=lambda c: c.name):
            role = "narrator" if c.is_narrator else c.gender
            out.append(f"{c.name} ({role}) -> {c.voice or 'narrator voice'}")
        return out


# --------------------------------------------------------------------- model


def _ask(system: str, user: str, cfg: DirectorConfig, schema: dict) -> list[dict]:
    if cfg.provider == "local":
        payload = {"model": cfg.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "stream": False, "format": schema,
                   "options": {"temperature": 0.2, "num_ctx": 8192}}
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = json.load(r).get("message", {}).get("content")
        except Exception:
            return []
    else:
        raw = (_api(system, user, cfg.model)
               or _claude_cli(system, user, cfg.model))
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    for key in ("characters", "speakers"):
        if isinstance(data.get(key), list):
            return [x for x in data[key] if isinstance(x, dict)]
    return []


ROSTER_SCHEMA = {
    "type": "object",
    "properties": {"characters": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"},
                       "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                       "age": {"type": "string", "enum": ["adult", "child"]},
                       "is_narrator": {"type": "boolean"}},
        "required": ["name", "gender", "age", "is_narrator"]}}},
    "required": ["characters"],
}

ATTRIB_SCHEMA = {
    "type": "object",
    "properties": {"speakers": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "who": {"type": "string"}},
        "required": ["i", "who"]}}},
    "required": ["speakers"],
}


def build_roster(sentences: list[Sentence], title: str,
                 cfg: DirectorConfig) -> dict[str, Character]:
    """One pass over a sample of the book to learn who is in it."""
    # Characters are named in the narration around their dialogue, not in the
    # dialogue itself, so the sample has to carry that narration with it —
    # otherwise the model can only call someone "She".
    keep: set[int] = set(range(min(40, len(sentences))))
    for i, s in enumerate(sentences):
        if s.role in ("speech", "tag"):
            keep.update(range(max(0, i - 3), min(len(sentences), i + 2)))
        if len(keep) > 130:
            break
    ordered = [sentences[i] for i in sorted(keep)][:140]
    lines = "\n".join(f"[{s.index}] ({s.role}) {s.text}" for s in ordered)
    rows = _ask(ROSTER_SYSTEM, f"Novel: {title}\n\n{lines}", cfg, ROSTER_SCHEMA)

    out: dict[str, Character] = {}
    for r in rows:
        name = str(r.get("name", "")).strip().strip('"\'')
        if not name or len(name) > 40 or name.upper() in PRONOUNS:
            continue
        if not re.match(r"^[A-Z][\w'\-. ]*$", name):
            continue
        out[name.upper()] = Character(
            name=name,
            gender=str(r.get("gender", "unknown")).lower(),
            age=str(r.get("age", "adult")).lower(),
            is_narrator=bool(r.get("is_narrator")),
        )
    return out


def attribute(sentences: list[Sentence], roster: dict[str, Character],
              title: str, cfg: DirectorConfig, on_progress=None) -> None:
    """Fill in Sentence.speaker for spoken lines, in place."""
    speech_idx = [i for i, s in enumerate(sentences) if s.role == "speech"]
    if not speech_idx:
        return
    names = ", ".join(c.name for c in roster.values()) or "(unknown)"

    window, context = 26, 6
    done = 0
    for start in range(0, len(sentences), window):
        batch = sentences[start:start + window]
        if not any(s.role == "speech" for s in batch):
            done += len(batch)
            if on_progress:
                on_progress(done, len(sentences))
            continue
        lo = max(0, start - context)
        shown = sentences[lo:start + window + context]
        lines = "\n".join(f"[{s.index}] ({s.role}) {s.text}" for s in shown)
        want = [s.index for s in batch if s.role == "speech"]
        user = (f"Novel: {title}\nRoster: {names}\n\n{lines}\n\n"
                f"Attribute every one of these speech lines: {want}")
        for row in _ask(ATTRIB_SYSTEM, user, cfg, ATTRIB_SCHEMA):
            try:
                idx = int(row.get("i", -1))
            except (TypeError, ValueError):
                continue
            who = str(row.get("who", "")).strip()
            if idx in want and who:
                sentences[idx].speaker = who
        done += len(batch)
        if on_progress:
            on_progress(done, len(sentences))


def assign_voices(roster: dict[str, Character], narrator_voice: str,
                  available: set[str]) -> Cast:
    """Give every speaking character a stable voice, never the narrator's."""
    male = [v for v in MALE_POOL if v in available and v != narrator_voice]
    female = [v for v in FEMALE_POOL if v in available and v != narrator_voice]
    other = [v for v in (MALE_POOL + FEMALE_POOL)
             if v in available and v != narrator_voice]

    for c in sorted(roster.values(), key=lambda c: c.name):
        if c.is_narrator:
            continue                      # narrators read in the POV voice
        pool = male if c.gender == "male" else female if c.gender == "female" else other
        if pool:
            c.voice = pool.pop(0)
            for p in (male, female, other):
                if c.voice in p:
                    p.remove(c.voice)
    return Cast(characters=roster, narrator_voice=narrator_voice)


def track_pov(sentences: list[Sentence], roster: dict[str, Character]) -> None:
    """Carry the point-of-view name forward from scene headers."""
    narrators = {c.name.upper() for c in roster.values() if c.is_narrator}
    current = ""
    for s in sentences:
        if s.kind in ("heading", "chapter", "title"):
            head = re.match(r"^([A-Za-z][\w'\-]*)", s.text.strip())
            if head and head.group(1).upper() in narrators:
                current = head.group(1).upper()
        s.pov = current
