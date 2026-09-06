"""The director: reads ahead and decides *how* each sentence should be read.

A TTS model handed one bare sentence guesses at prosody from that sentence
alone. A human narrator has read the paragraph — they know the line is the
quiet turn after an argument, and they slow down and drop their voice for it.
This module buys that context back: an LLM reads a window of sentences and
annotates each with emotion, pace, emphasis and the pause that follows.

Falls back to a decent heuristic pass when no LLM is reachable, so the
pipeline always runs.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .document import Sentence

CACHE_DIR = Path.home() / ".cache" / "narrator" / "direction"
# Bump when the prompt or schema changes, so stale direction is not reused.
DIRECTION_VERSION = "2"

EMOTIONS = [
    "neutral", "warm", "curious", "excited", "urgent", "tense", "somber",
    "reflective", "wry", "authoritative", "tender", "ominous",
]

# How much direction is worth generating depends entirely on what the engine
# can honour. Kokoro ignores emotion labels and prose notes, so asking a local
# model to write them costs minutes of generation for audio that never changes.
PROFILES: dict[str, list[str]] = {
    # engine expressiveness <= 1: only timing reaches the voice
    "timing":  ["i", "pace", "pause_after"],
    # == 2: pitch and stress are controllable too
    "prosody": ["i", "emotion", "pace", "emphasis", "pause_after"],
    # >= 3: the engine reads direction as direction
    "full":    ["i", "emotion", "pace", "emphasis", "pause_after", "note"],
}

FIELD_SPEC = {
    "i": '"i": <the sentence number>',
    "emotion": '"emotion": one of EMOTIONS',
    "pace": '"pace": 0.75-1.25 (1.0 normal; below 1 slower and weightier)',
    "emphasis": '"emphasis": [up to 3 words appearing verbatim in the sentence]',
    "pause_after": '"pause_after": 0.1-2.0 seconds of silence following',
    "note": '"note": "<=12 words of performance direction"',
}

FIELD_TYPES = {
    "i": {"type": "integer"},
    "emotion": {"type": "string", "enum": EMOTIONS},
    "pace": {"type": "number"},
    "emphasis": {"type": "array", "items": {"type": "string"}},
    "pause_after": {"type": "number"},
    "note": {"type": "string"},
}

PRINCIPLES = """Direction principles:
- Vary. A performance that is uniformly "warm" is as dead as one that is
  uniformly flat. Let the direction track what the text is actually doing.
- Slow down (pace 0.8-0.9) for: definitions, the turn in an argument, the
  emotional beat of a scene, anything the reader must hold onto.
- Speed up slightly (1.05-1.15) for: asides, lists of examples, momentum,
  a build toward a point.
- Pause length is punctuation for the ear. 0.15 inside a flowing thought,
  0.4 at a paragraph end, 0.8-1.5 before a revelation or after a question
  the reader should sit with.
- A heading is an arrival: slower, with a real pause after."""

EXPRESSIVE_PRINCIPLES = """
- Most sentences are neutral or warm. Reserve the strong colours for lines
  that earn them.
- Emphasis is contrastive: mark the word that carries the *new* information
  or the pivot ("but", "never", the surprising noun) - not every adjective.
- A (speech) line is dialogue, with the quote marks already removed. Perform
  the speaker's state, not the narrator's. A (tag) line is "he said" and is
  subordinate: it is thrown away quickly, never performed."""


def build_system(profile: str) -> str:
    fields = PROFILES[profile]
    spec = ",\n   ".join(FIELD_SPEC[f] for f in fields)
    emotions = ("\nEMOTIONS: " + " ".join(EMOTIONS) + "\n") if "emotion" in fields else ""
    extra = EXPRESSIVE_PRINCIPLES if len(fields) > 3 else ""
    return f"""You are the director of an audiobook recording session.

You receive numbered sentences from a document, in order, with the surrounding
context. For each sentence you decide how the narrator should perform it.

Return ONLY JSON: {{"directions": [ ... ]}} with one object per sentence given,
in order:
  {{{spec}}}
{emotions}
{PRINCIPLES}{extra}
"""


def build_schema(profile: str) -> dict:
    fields = PROFILES[profile]
    return {
        "type": "object",
        "properties": {
            "directions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {f: FIELD_TYPES[f] for f in fields},
                    # All required: a smaller model silently skips optional
                    # fields, and emphasis is the first thing it drops.
                    "required": list(fields),
                },
            }
        },
        "required": ["directions"],
    }


USER_TMPL = """Document: {title}

{before}
--- DIRECT THESE ---
{target}
--- END ---
{after}

Return {{"directions": [...]}} covering sentences {first} to {last} only."""


# Where the direction is thought up.
#   "cloud" — Anthropic. Best judgement; your text is sent over the network.
#   "local" — a model on this machine via Ollama. Nothing leaves the machine.
#   "off"   — heuristics only. Nothing leaves, nothing is thought.
Provider = str

OLLAMA_URL = "http://127.0.0.1:11434"
LOCAL_DEFAULT = "qwen2.5:14b"


@dataclass
class DirectorConfig:
    enabled: bool = True
    provider: Provider = "cloud"
    model: str = "claude-sonnet-5"
    window: int = 24        # sentences directed per call
    context: int = 6        # sentences of surrounding context shown
    cache: bool = True
    jobs: int = 3           # batches directed concurrently
    profile: str = "full"   # how much direction to generate

    @classmethod
    def local(cls, model: str = LOCAL_DEFAULT, jobs: int = 4,
              profile: str = "full") -> "DirectorConfig":
        # A 14B model holds a shorter window in mind reliably than Sonnet does.
        return cls(provider="local", model=model, window=12, context=4,
                   jobs=jobs, profile=profile)


# An engine can only be directed as far as it can perform.
def profile_for(expressiveness: int) -> str:
    if expressiveness <= 1:
        return "timing"
    if expressiveness == 2:
        return "prosody"
    return "full"


# ------------------------------------------------------------------ heuristics

_STRONG = re.compile(
    r"\b(never|always|nothing|everything|no one|must|cannot|impossible|"
    r"suddenly|finally|but|however|yet|instead|because|therefore)\b", re.I)
_SOMBER = re.compile(r"\b(died|death|lost|grief|alone|war|failed|end(ed)?|dark)\b", re.I)
_EXCITED = re.compile(r"\b(discover|breakthrough|astonish|remarkable|extraordinary|first time)\b", re.I)


def heuristic_direct(sentences: list[Sentence]) -> None:
    """Structure-aware fallback. Not a performance, but not a monotone either."""
    for i, s in enumerate(sentences):
        text = s.text
        if s.kind in ("title", "chapter"):
            s.emotion, s.pace, s.pause_after = "authoritative", 0.85, 1.2
            s.note = "an arrival"
            continue
        if s.kind == "heading":
            s.emotion, s.pace, s.pause_after = "authoritative", 0.9, 0.8
            continue
        if s.kind == "quote":
            s.emotion, s.pace, s.pause_after = "reflective", 0.9, 0.5
            continue
        if s.kind == "list_item":
            s.emotion, s.pace, s.pause_after = "neutral", 1.05, 0.3
            continue

        s.emotion = "neutral"
        if _SOMBER.search(text):
            s.emotion, s.pace = "somber", 0.9
        elif _EXCITED.search(text):
            s.emotion, s.pace = "excited", 1.08
        elif text.rstrip().endswith("?"):
            s.emotion, s.pace = "curious", 0.98

        if text.rstrip().endswith("?"):
            s.pause_after = 0.75
        elif text.rstrip().endswith("!"):
            s.pause_after = 0.6
        else:
            s.pause_after = 0.35

        # last sentence of a block gets a paragraph-length breath
        if i + 1 < len(sentences) and sentences[i + 1].block_index != s.block_index:
            s.pause_after = max(s.pause_after, 0.6)
        elif i + 1 == len(sentences):
            s.pause_after = 1.0

        m = _STRONG.search(text)
        if m:
            s.emphasis = [m.group(0)]
        # short punchy sentence after a long one: let it land
        if len(text) < 60 and i > 0 and len(sentences[i - 1].text) > 160:
            s.pace = min(s.pace, 0.88)
            s.pause_after = max(s.pause_after, 0.7)


# ------------------------------------------------------------------------ llm


def _claude_cli(system: str, user: str, model: str, timeout: int = 180) -> str | None:
    exe = shutil.which("claude")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "-p", user, "--model", model, "--append-system-prompt", system,
             "--output-format", "text"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return r.stdout if r.returncode == 0 else None


def _api(system: str, user: str, model: str) -> str | None:
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model, max_tokens=4096, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
    except Exception:
        return None


def ollama_available(url: str = OLLAMA_URL) -> list[str]:
    """Models served locally right now, or [] if Ollama isn't running."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as r:
            data = json.load(r)
    except Exception:
        return []
    return [m["name"] for m in data.get("models", [])]


def _ollama(system: str, user: str, model: str, schema: dict,
            url: str = OLLAMA_URL, timeout: int = 300) -> str | None:
    """Direct the scene with a model running on this machine."""
    import urllib.request

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": schema,            # constrained decoding — always valid JSON
        "options": {"temperature": 0.6, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        f"{url}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception:
        return None
    return (data.get("message") or {}).get("content")


def _parse(raw: str) -> list[dict]:
    data = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}|\[.*\]", raw or "", re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
    if isinstance(data, dict):
        data = data.get("directions") or data.get("sentences") or []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


def _apply(sentences: list[Sentence], notes: list[dict], offset: int) -> set[int]:
    done: set[int] = set()
    by_index = {s.index: s for s in sentences}
    for note in notes:
        try:
            idx = int(note.get("i", -1))
        except (TypeError, ValueError):
            continue
        s = by_index.get(idx)
        if s is None:
            continue
        emo = str(note.get("emotion", "neutral")).lower().strip()
        s.emotion = emo if emo in EMOTIONS else "neutral"
        try:
            s.pace = min(1.25, max(0.75, float(note.get("pace", 1.0))))
        except (TypeError, ValueError):
            s.pace = 1.0
        try:
            s.pause_after = min(2.0, max(0.1, float(note.get("pause_after", 0.35))))
        except (TypeError, ValueError):
            s.pause_after = 0.35
        emph = note.get("emphasis") or []
        if isinstance(emph, list):
            s.emphasis = [w for w in (str(x) for x in emph[:3])
                          if w and w.lower() in s.text.lower()]
        s.note = str(note.get("note", ""))[:80]
        done.add(idx)
    return done


def _render(sentences: list[Sentence]) -> str:
    def label(s: Sentence) -> str:
        return s.role if s.role != "narration" else s.kind
    return "\n".join(f"[{s.index}] ({label(s)}) {s.text}" for s in sentences)


def _cache_path(title: str, batch: list[Sentence], model: str,
                profile: str) -> Path:
    h = hashlib.sha256()
    h.update(DIRECTION_VERSION.encode())
    h.update(model.encode())
    h.update(profile.encode())
    h.update(title.encode())
    for s in batch:
        h.update(f"{s.index}\x00{s.kind}\x00{s.text}\x00".encode())
    return CACHE_DIR / f"{h.hexdigest()[:24]}.json"


def direct(
    sentences: list[Sentence],
    title: str = "",
    config: DirectorConfig | None = None,
    on_progress=None,
) -> None:
    """Annotate `sentences` in place.

    Batches are independent — each carries its own slice of surrounding
    context and its own cache entry — so they are directed concurrently.
    Order of completion does not matter: every note is applied by the
    sentence index it names.
    """
    cfg = config or DirectorConfig()
    heuristic_direct(sentences)          # always a sane baseline
    if cfg.enabled and sentences:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _direct_batches(sentences, title, cfg, on_progress)

    _shape_dialogue(sentences)

    # A scene break is structural: the director may not shorten it.
    for s_ in sentences:
        if s_.scene_break:
            s_.pause_after = max(s_.pause_after, 1.6)


def _shape_dialogue(sentences: list[Sentence]) -> None:
    """Give an exchange the rhythm of speech rather than of paragraphs.

    Uniform gaps are what make read dialogue sound mechanical. Real turns
    come back fast; an attribution rides on the end of its line almost
    without a seam; and a beat only opens up when someone actually pauses.
    """
    for i, s in enumerate(sentences):
        nxt = sentences[i + 1] if i + 1 < len(sentences) else None

        if s.role == "tag":
            # subordinate: quicker, and it does not get to hold the floor
            s.pace = max(s.pace, 1.08)
            s.pause_after = min(s.pause_after, 0.45)
            continue

        if s.role == "speech":
            if nxt is not None and nxt.role == "tag":
                # "…," she said — one breath, so barely a seam
                s.pause_after = 0.08
            elif nxt is not None and nxt.role == "speech":
                # a fast return; a question earns slightly more air
                s.pause_after = 0.34 if s.text.rstrip().endswith("?") else 0.24
            else:
                # handing back to narration
                s.pause_after = min(s.pause_after, 0.5)
        elif nxt is not None and nxt.role == "speech":
            # narration handing off to a line of dialogue
            s.pause_after = min(s.pause_after, 0.45)


def _direct_batches(sentences: list[Sentence], title: str, cfg: DirectorConfig,
                    on_progress) -> None:
    total = len(sentences)
    batches = [sentences[i:i + cfg.window] for i in range(0, total, cfg.window)]
    system = build_system(cfg.profile)
    schema = build_schema(cfg.profile)

    done_count = 0
    lock = threading.Lock()

    def advance(n: int) -> None:
        nonlocal done_count
        with lock:
            done_count += n
            if on_progress:
                on_progress(min(done_count, total), total)

    def run(batch_no: int, batch: list[Sentence]) -> tuple[list[Sentence], list[dict]]:
        cache_file = _cache_path(title, batch, cfg.model, cfg.profile)
        if cfg.cache and cache_file.exists():
            try:
                notes = json.loads(cache_file.read_text())
                if notes:
                    return batch, notes
            except json.JSONDecodeError:
                pass

        start = batch_no * cfg.window
        before = sentences[max(0, start - cfg.context):start]
        after = sentences[start + len(batch):start + len(batch) + cfg.context]
        user = USER_TMPL.format(
            title=title or "(untitled)",
            before=_render(before) or "(start of document)",
            target=_render(batch),
            after=_render(after) or "(end of document)",
            first=batch[0].index, last=batch[-1].index,
        )
        if cfg.provider == "local":
            raw = _ollama(system, user, cfg.model, schema)
        else:
            raw = _api(system, user, cfg.model) or _claude_cli(system, user, cfg.model)

        notes = _parse(raw) if raw else []
        if notes and cfg.cache:
            try:
                cache_file.write_text(json.dumps(notes))
            except OSError:
                pass
        return batch, notes

    jobs = max(1, min(cfg.jobs, len(batches)))
    if jobs == 1:
        for i, batch in enumerate(batches):
            _, notes = run(i, batch)
            if notes:
                _apply(sentences, notes, 0)
            advance(len(batch))
        return

    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="direct") as pool:
        futures = {pool.submit(run, i, b): b for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                _, notes = fut.result()
            except Exception:
                notes = []          # heuristic baseline stands for this batch
            if notes:
                # _apply addresses sentences by index, so completion order
                # is irrelevant; the lock only guards the progress counter.
                _apply(sentences, notes, 0)
            advance(len(batch))
