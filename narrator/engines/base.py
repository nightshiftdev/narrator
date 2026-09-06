"""Engine interface. Everything downstream sees only this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..document import Sentence


@dataclass
class Clip:
    """Mono float32 audio in [-1, 1] plus its sample rate."""

    samples: np.ndarray
    rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.rate if self.rate else 0.0


class Engine(ABC):
    """A voice.

    Implementations differ wildly in how much direction they can honour;
    `expressiveness` tells the pipeline how much to lean on the director.
      0 - ignores direction entirely
      1 - honours pace and pauses (mechanical prosody control)
      2 - honours pace, pauses and emphasis
      3 - takes free-text emotional direction
    """

    name: str = "base"
    rate: int = 24_000
    expressiveness: int = 0

    @abstractmethod
    def synth(self, sentence: Sentence) -> Clip:
        """Render one directed sentence to audio."""

    def warmup(self) -> None:
        """Optional: load weights before the clock starts."""

    def voices(self) -> list[str]:
        return []

    def close(self) -> None:
        pass


def silence(seconds: float, rate: int) -> np.ndarray:
    return np.zeros(max(0, int(seconds * rate)), dtype=np.float32)
