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

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field

from .director import OLLAMA_URL, DirectorConfig, _api, _claude_cli
from .document import Sentence

NARRATOR = "NARRATOR"

# Attribution is as expensive as direction and just as reusable, so it is
# cached the same way: keyed by the batch text, the roster and the model.
ATTRIB_CACHE = Path.home() / ".cache" / "narrator" / "attribution"
ATTRIB_VERSION = "1"

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
- The narration around a line is the strongest clue, and it usually refers to
  the speaker by pronoun rather than by name. If the lines just before and
  after a spoken line describe "she" doing something, that line is almost
  certainly hers, even when the previous speaker was someone else:

      "They let me go today."      <- the narrator
      She was quiet.
      I could hear the stove.
      "Okay."                      <- HERS, not the narrator's
      She turned back to it.

- A first-person narrator describing someone else ("She was quiet", "She
  stopped", "She turned back") is not speaking in those lines. Do not assign
  a spoken line to the narrator merely because the narrator spoke last.
- A name inside a spoken line is who is being ADDRESSED, not who is talking.
  If someone says "Jack." they are calling to Jack, so the speaker is
  somebody else.
- Use UNKNOWN when you genuinely cannot tell. That is better than guessing;
  an unknown line is read by the narrator, which is never jarring."""


@dataclass
class Character:
    name: str
    gender: str = "unknown"
    age: str = "adult"
    is_narrator: bool = False
    voice: str = ""
    flat: bool = False


@dataclass
class Cast:
    characters: dict[str, Character] = field(default_factory=dict)
    narrator_voice: str = ""
    # Speakers whose lines are never coloured. A machine narrator that gets
    # the same emotional shaping as a person stops reading as a machine, so
    # this switches the style offsets off entirely for them.
    flat: set[str] = field(default_factory=set)

    def is_flat(self, sentence: Sentence) -> bool:
        who = self.resolve(sentence.speaker or "")
        if who and who in self.flat:
            return True
        # narration inside a flat narrator's section is flat too
        return bool(sentence.pov) and sentence.pov.upper() in self.flat

    def resolve(self, who: str) -> str:
        """Map a speaker name to a cast key, tolerating how names vary.

        A model will call the same person Yael, Yael Gur and YAEL GUR within
        one book. Left alone each becomes a separate character with its own
        voice, so one person would change voice mid-scene.
        """
        key = (who or "").strip().upper()
        if not key or key in self.characters:
            return key
        parts = key.split()
        for cand in self.characters:
            cparts = cand.split()
            if parts[0] == cparts[0] or (len(parts) > 1 and parts[-1] == cparts[-1]):
                return cand
        return key

    def voice_for(self, sentence: Sentence) -> str:
        """The voice this line should be read in, or "" for the narrator."""
        who = sentence.speaker
        if not who or who.upper() in (NARRATOR, "UNKNOWN"):
            return ""
        c = self.characters.get(self.resolve(who))
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
    # dialogue itself, so the sample has to carry that narration with it --
    # otherwise the model can only call someone "She".
    #
    # The sample must also span the whole book. Sampling only the opening
    # misses anyone introduced later, and a character who is not on the
    # roster cannot be attributed at all: their lines get handed to whoever
    # is on it.
    keep: set[int] = set(range(min(30, len(sentences))))
    speech_at = [i for i, s in enumerate(sentences) if s.role in ("speech", "tag")]
    if speech_at:
        budget = 150
        stride = max(1, len(speech_at) // max(1, budget // 4))
        for i in speech_at[::stride]:
            keep.update(range(max(0, i - 3), min(len(sentences), i + 2)))
    ordered = [sentences[i] for i in sorted(keep)][:220]
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

    ATTRIB_CACHE.mkdir(parents=True, exist_ok=True)
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

        h = hashlib.sha256()
        h.update(ATTRIB_VERSION.encode())
        h.update(cfg.model.encode())
        h.update(names.encode())
        h.update(user.encode())
        cache_file = ATTRIB_CACHE / f"{h.hexdigest()[:24]}.json"
        rows: list[dict] = []
        if cache_file.exists():
            try:
                rows = json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                rows = []
        if not rows:
            rows = _ask(ATTRIB_SYSTEM, user, cfg, ATTRIB_SCHEMA)
            if rows:
                try:
                    cache_file.write_text(json.dumps(rows))
                except OSError:
                    pass

        for row in rows:
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
                  available: set[str], pov_order: list[str] | None = None) -> Cast:
    """Give every speaker a stable voice, and every POV narrator their own.

    A book that changes narrator needs a different voice per point of view,
    or the switch is inaudible and the device is lost. The first narrator
    encountered keeps the voice the reader asked for; the rest are cast from
    the pools by gender.
    """
    taken = {narrator_voice}

    def take(gender: str) -> str:
        pools = ([MALE_POOL] if gender == "male"
                 else [FEMALE_POOL] if gender == "female"
                 else [MALE_POOL, FEMALE_POOL])
        for pool in pools:
            for v in pool:
                if v in available and v not in taken:
                    taken.add(v)
                    return v
        return ""

    # Narrators first, in the order they appear, so the primary POV keeps the
    # requested voice.
    order = pov_order or []
    narrators = [c for c in roster.values() if c.is_narrator]
    narrators.sort(key=lambda c: order.index(c.name.upper())
                   if c.name.upper() in order else len(order))
    for i, c in enumerate(narrators):
        c.voice = narrator_voice if i == 0 else take(c.gender)

    for c in sorted(roster.values(), key=lambda c: c.name):
        if not c.is_narrator:
            c.voice = take(c.gender)
    return Cast(characters=roster, narrator_voice=narrator_voice)


def track_pov(sentences: list[Sentence], roster: dict[str, Character]) -> None:
    """Carry the point-of-view name forward from scene headers.

    The headers themselves are the authority, not the roster: a book that
    marks its sections MIRA / SENTINEL / JACK is telling us who narrates, and
    that signal is far more reliable than asking a model to work it out. Any
    all-capital name at the head of a section is taken as a POV marker, and
    registered as a narrator if the roster missed them - which it will, since
    a narrator who never speaks aloud looks like nobody at all.
    """
    current = ""
    for s in sentences:
        if s.kind in ("heading", "chapter", "title"):
            head = re.match(r"^([A-Z][A-Z'\-]{1,})\b", s.text.strip())
            if head:
                name = head.group(1).upper()
                current = name
                if name not in roster:
                    roster[name] = Character(name=name, is_narrator=True)
                else:
                    roster[name].is_narrator = True
        s.pov = current


# ------------------------------------------------------------------ cast file


def _line_key(text: str) -> str:
    """Normalised key for a spoken line, stable across small edits."""
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def apply_overrides(sentences: list[Sentence], conf: dict,
                    roster: dict[str, Character]) -> tuple[int, list[str]]:
    """Apply a reviewed cast file. Returns (lines changed, warnings).

    Line overrides are keyed by the spoken text rather than by index, because
    an index moves the moment a sentence is added anywhere earlier in the book
    and would then silently reassign the wrong line.

    A short line recurs -- "Okay.", "I don't know." -- and the speakers differ.
    Suffix a key with #N to address the Nth occurrence:

        "Okay." = "SARA"        # every occurrence
        "Okay.#2" = "JACK"      # only the second

    A bare key applies to occurrences no numbered key claims.
    """
    changed = 0
    warnings: list[str] = []

    lines = conf.get("lines") or {}
    general: dict[str, str] = {}
    numbered: dict[tuple[str, int], str] = {}
    for raw, who in lines.items():
        m = re.match(r"^(.*)#(\d+)$", str(raw).strip())
        if m:
            numbered[(_line_key(m.group(1)), int(m.group(2)))] = who
        else:
            general[_line_key(raw)] = who

    wanted = dict(general)
    seen: set[str] = set()
    occurrence: dict[str, int] = {}
    for s in sentences:
        if s.role != "speech":
            continue
        key = _line_key(s.text)
        n = occurrence[key] = occurrence.get(key, 0) + 1
        who = numbered.get((key, n), general.get(key))
        if who is not None:
            seen.add(key)
            if (key, n) in numbered:
                seen.add(f"{key}#{n}")
            if s.speaker != who:
                s.speaker = who
                changed += 1
    for key in list(general) + [f"{k}#{n}" for k, n in numbered]:
        if key not in seen:
            warnings.append(f'no spoken line matches "{key}"')
    return changed, warnings


def ensure_cast(sentences: list[Sentence], cast: "Cast", available: set[str],
                cfg: DirectorConfig, title: str = "") -> list[str]:
    """Give a voice to anyone who speaks but was never cast.

    A reviewed cast file often names someone the roster pass missed. They must
    not silently fall back to the narrator, or the correction the reader just
    made would do nothing.
    """
    notes: list[str] = []
    speaking = {s.speaker.upper() for s in sentences
                if s.role == "speech" and s.speaker}
    missing = [w for w in sorted(speaking)
               if w not in cast.characters and w not in (NARRATOR, "UNKNOWN")]
    if not missing:
        return notes

    # Ask the model for a gender so the voice is not obviously wrong.
    rows = _ask(ROSTER_SYSTEM,
                f"Novel: {title}\nThese characters speak: {', '.join(missing)}.\n"
                "Give each one's gender as used in the story.",
                cfg, ROSTER_SCHEMA)
    gender = {str(r.get("name", "")).upper(): str(r.get("gender", "unknown")).lower()
              for r in rows}

    used = {c.voice for c in cast.characters.values() if c.voice}
    used.add(cast.narrator_voice)
    for who in missing:
        display = next((s.speaker for s in sentences
                        if s.speaker.upper() == who), who)
        g = gender.get(who, "unknown")
        pool = MALE_POOL if g == "male" else FEMALE_POOL if g == "female" else \
            MALE_POOL + FEMALE_POOL
        voice = next((v for v in pool if v in available and v not in used), "")
        cast.characters[who] = Character(name=display, gender=g, voice=voice)
        used.add(voice)
        notes.append(f"{display} ({g}) -> {voice or 'narrator voice'}")
    return notes


def write_review(path, sentences: list[Sentence], cast: "Cast",
                 pov_voices: dict[str, str]) -> None:
    """Emit an editable cast file describing what the model decided."""
    from collections import Counter

    counts = Counter(s.speaker or "UNKNOWN" for s in sentences if s.role == "speech")
    out = ["# Cast file for narrator.",
           "#",
           "# Edit and pass back with --cast-file. Everything here is optional;",
           "# whatever you leave out keeps the inferred value.",
           "",
           "[narrator]",
           "# POV name -> voice. A section headed with this name narrates in it.",
           ]
    for name, voice in sorted(pov_voices.items()):
        out.append(f'{name} = "{voice}"')
    out += ["", "[characters]", "# character -> voice"]
    for c in sorted(cast.characters.values(), key=lambda c: c.name):
        if c.is_narrator:
            continue
        out.append(f'"{c.name}" = "{c.voice}"   # {c.gender}, '
                   f'{counts.get(c.name, 0)} lines')
    out += ["",
            "[lines]",
            "# Per-line corrections, keyed by the spoken text (punctuation and",
            "# case are ignored). Uncomment and fix any line read by the wrong",
            "# voice. NARRATOR means the point-of-view character.",
            ""]
    for s in sentences:
        if s.role != "speech":
            continue
        who = s.speaker or "UNKNOWN"
        text = s.text.replace('"', "'")
        mark = "" if s.speaker else "  # <- not attributed"
        out.append(f'# "{text}" = "{who}"{mark}')
    pathlib_path = path if hasattr(path, "write_text") else __import__("pathlib").Path(path)
    pathlib_path.write_text("\n".join(out) + "\n")
