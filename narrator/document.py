"""The intermediate representation everything in the pipeline agrees on."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal[
    "title",      # book / document title
    "chapter",    # chapter heading
    "heading",    # sub-heading
    "para",       # ordinary prose
    "quote",      # block quote — read a touch slower, softer
    "list_item",
    "code",       # usually skipped, or announced and summarised
    "caption",
    "footnote",
]


@dataclass
class Block:
    """One structural unit of the source document."""

    kind: BlockKind
    text: str
    level: int = 0          # heading depth, list nesting
    index: int = 0          # position in the document
    source_page: int | None = None

    @property
    def is_heading(self) -> bool:
        return self.kind in ("title", "chapter", "heading")


@dataclass
class Sentence:
    """A synthesis unit: one sentence, plus the direction for reading it."""

    text: str
    block_index: int
    kind: BlockKind
    index: int = 0
    # --- filled in by the director ---
    emotion: str = "neutral"
    pace: float = 1.0          # multiplier, 0.8 = slower/heavier
    emphasis: list[str] = field(default_factory=list)
    pause_after: float = 0.35  # seconds of silence following
    note: str = ""             # free-text direction, used by expressive engines
    scene_break: bool = False  # a "* * *" divider follows: hold the silence
    # "narration" | "speech" (inside quotes) | "tag" (he said / she asked)
    role: str = "narration"


@dataclass
class Document:
    title: str
    blocks: list[Block]
    source: str = ""

    def chapters(self) -> list[tuple[str, list[Block]]]:
        """Split into (chapter title, blocks) pairs for m4b chapter marks."""
        out: list[tuple[str, list[Block]]] = []
        current_title = self.title
        current: list[Block] = []
        for b in self.blocks:
            if b.kind in ("chapter", "title") and current:
                out.append((current_title, current))
                current_title, current = b.text, [b]
            else:
                if b.kind in ("chapter", "title"):
                    current_title = b.text
                current.append(b)
        if current:
            out.append((current_title, current))
        return out
