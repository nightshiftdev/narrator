"""Render and play. Audio is produced ahead of the ear, never behind it."""
from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio as A
from .document import Document, Sentence
from .engines.base import Clip, Engine


@dataclass
class Rendered:
    sentence: Sentence
    samples: np.ndarray
    rate: int
    start: float = 0.0        # position in the finished programme


def group_for_synthesis(sentences: list[Sentence], max_words: int = 9,
                        max_group: int = 26) -> list[list[Sentence]]:
    """Batch consecutive short lines into one synthesis call.

    Kokoro's stability depends on clip length, not on the words: measured on
    one voice, a three-word sentence came out at 10-12% jitter and a
    twelve-word one at 2.6%. A listener hears the short ones as a shaky,
    machine-like wobble. Synthesising a run of them together roughly halves
    the roughness, because the model has enough context to hold a contour.

    Only lines that share a voice and a role are joined, so no character is
    ever merged into another, and a scene break always ends a group. The
    director's pause for the final line of the group is the one that survives;
    the internal gaps come from the punctuation, which for a rapid exchange is
    the better rhythm anyway.
    """
    groups: list[list[Sentence]] = []
    cur: list[Sentence] = []

    def words(items) -> int:
        return sum(len(x.text.split()) for x in items)

    for s in sentences:
        short = len(s.text.split()) <= max_words
        joinable = (
            cur and short
            and words(cur) + len(s.text.split()) <= max_group
            and cur[-1].speaker == s.speaker
            and cur[-1].role == s.role
            and cur[-1].pov == s.pov
            and cur[-1].kind == s.kind
            and cur[-1].kind not in ("title","chapter","heading")
            and not cur[-1].scene_break
            and cur[-1].pause_after <= 0.5
        )
        if joinable:
            cur.append(s)
            continue
        if cur:
            groups.append(cur)
            cur = []
        if short and s.kind not in ("title","chapter","heading"):
            cur = [s]
        else:
            groups.append([s])
    if cur:
        groups.append(cur)
    return groups


def merge_group(group: list[Sentence]) -> Sentence:
    """One Sentence standing for a whole group, for the engine to speak."""
    if len(group) == 1:
        return group[0]
    head = group[0]
    text = " ".join(x.text.rstrip() for x in group)
    merged = Sentence(
        text, head.block_index, head.kind, index=head.index,
        emotion=head.emotion,
        pace=sum(x.pace for x in group) / len(group),
        emphasis=head.emphasis,
        pause_after=group[-1].pause_after,
        note=head.note,
        scene_break=group[-1].scene_break,
        role=head.role,
        speaker=head.speaker,
        pov=head.pov,
    )
    return merged


@dataclass
class Progress:
    rendered: int = 0
    total: int = 0
    seconds_audio: float = 0.0
    seconds_spent: float = 0.0

    @property
    def realtime_factor(self) -> float:
        return self.seconds_audio / self.seconds_spent if self.seconds_spent else 0.0


class Renderer:
    """Synthesises sentences on a worker thread and publishes finished audio."""

    def __init__(self, engine: Engine, sentences: list[Sentence], *,
                 rate: int | None = None, lead: int = 8, on_progress=None):
        self.engine = engine
        self.sentences = sentences
        self.rate = rate or engine.rate
        self.out: queue.Queue[Rendered | None] = queue.Queue(maxsize=lead)
        self.progress = Progress(total=len(sentences))
        self.on_progress = on_progress
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def start(self) -> Renderer:
        self._thread = threading.Thread(target=self._run, daemon=True, name="render")
        self._thread.start()
        return self

    def _run(self) -> None:
        clock = 0.0
        t0 = time.monotonic()
        try:
            self.engine.warmup()
            for group in group_for_synthesis(self.sentences):
                if self._stop.is_set():
                    break
                s = merge_group(group)
                clip = self.engine.synth(s)
                x = clip.samples
                if len(x) and s.role == "tag":
                    # "he said" is spoken under the line, not level with it
                    x = x * 0.72
                if len(x):
                    x = A.trim_silence(x, clip.rate)
                    x = A.fade(x, clip.rate)
                    if clip.rate != self.rate:
                        x = A.resample(x, clip.rate, self.rate)
                # Pause is ours, not the engine's — that is what makes the
                # rhythm feel deliberate rather than accidental.
                x = np.concatenate([x, np.zeros(int(s.pause_after * self.rate),
                                                dtype=np.float32)])
                item = Rendered(s, x, self.rate, clock)
                clock += len(x) / self.rate

                self.progress.rendered += len(group)
                self.progress.seconds_audio = clock
                self.progress.seconds_spent = time.monotonic() - t0
                if self.on_progress:
                    self.on_progress(self.progress, s)

                while not self._stop.is_set():
                    try:
                        self.out.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:            # surfaced by the consumer
            self.error = exc
        finally:
            self.out.put(None)

    def stop(self) -> None:
        self._stop.set()

    def __iter__(self):
        while True:
            item = self.out.get()
            if item is None:
                break
            yield item
        if self.error:
            raise self.error


class Player:
    """Plays clips through afplay, batching so the seams stay inaudible."""

    def __init__(self, rate: int, batch_seconds: float = 6.0, speed: float = 1.0):
        self.rate = rate
        self.batch_seconds = batch_seconds
        self.speed = speed
        self._dir = Path(tempfile.mkdtemp(prefix="narrator-play-"))
        self._n = 0
        self._proc: subprocess.Popen | None = None
        self.stopped = threading.Event()

    def _play_buffer(self, buf: list[np.ndarray]) -> None:
        if not buf:
            return
        chunk = np.concatenate(buf)
        self._n += 1
        path = self._dir / f"{self._n:05d}.wav"
        A.write_wav(path, chunk, self.rate)
        cmd = ["afplay", str(path)]
        if abs(self.speed - 1.0) > 0.01:
            cmd += ["-r", f"{self.speed:.2f}"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        self._proc.wait()
        self._proc = None
        path.unlink(missing_ok=True)

    def feed(self, items) -> None:
        buf: list[np.ndarray] = []
        held = 0.0
        for item in items:
            if self.stopped.is_set():
                break
            buf.append(item.samples)
            held += len(item.samples) / self.rate
            # Flush on a structural boundary or once we have enough to be safe.
            boundary = item.sentence.pause_after >= 0.6
            if held >= self.batch_seconds or (boundary and held >= 2.0):
                self._play_buffer(buf)
                buf, held = [], 0.0
        if not self.stopped.is_set():
            self._play_buffer(buf)

    def stop(self) -> None:
        self.stopped.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def close(self) -> None:
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)


@dataclass
class Result:
    samples: np.ndarray
    rate: int
    chapters: list[tuple[str, float]] = field(default_factory=list)
    duration: float = 0.0


def collect(renderer: Renderer, doc: Document, *, play: bool = False,
            speed: float = 1.0) -> Result:
    """Drive the renderer, optionally playing live, and return the programme."""
    pieces: list[np.ndarray] = []
    chapters: list[tuple[str, float]] = []
    clock = 0.0
    seen_blocks: set[int] = set()
    heading_kinds = {"title", "chapter"}
    block_kind = {b.index: b.kind for b in doc.blocks}
    block_text = {b.index: b.text for b in doc.blocks}

    player = Player(renderer.rate, speed=speed) if play else None

    def tap():
        nonlocal clock
        for item in renderer:
            bi = item.sentence.block_index
            if (block_kind.get(bi) in heading_kinds and bi not in seen_blocks):
                chapters.append((block_text.get(bi, "Chapter"), clock))
            seen_blocks.add(bi)
            pieces.append(item.samples)
            clock += len(item.samples) / renderer.rate
            yield item

    try:
        if player:
            player.feed(tap())
        else:
            for _ in tap():
                pass
    finally:
        if player:
            player.close()

    samples = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    samples = A.normalise_loudness(samples, renderer.rate)
    return Result(samples, renderer.rate, chapters, len(samples) / renderer.rate)
