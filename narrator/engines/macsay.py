"""macOS `say`. No install, no download, always available.

The bar it has to clear is low, but it is not nothing: the classic speech
synthesis manager honours embedded [[...]] commands, so the director's pace
and emphasis decisions do reach it. Useful as a baseline you can A/B against
the neural engines, and as a guaranteed fallback.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..document import Sentence
from .base import Clip, Engine

# Voices that are actually pleasant to listen to for more than a minute.
PREFERRED = ["Ava (Premium)", "Zoe (Premium)", "Evan (Enhanced)", "Ava (Enhanced)",
             "Allison", "Samantha", "Daniel", "Alex"]

BASE_WPM = 180
_WORD = re.compile(r"[\w']+")


class MacSayEngine(Engine):
    name = "macsay"
    rate = 22_050
    expressiveness = 2

    def __init__(self, voice: str | None = None, wpm: int = BASE_WPM):
        self.voice = voice or self._pick_voice()
        self.wpm = wpm
        self._tmp = Path(tempfile.mkdtemp(prefix="narrator-say-"))
        self._n = 0

    # ------------------------------------------------------------------ voices
    def voices(self) -> list[str]:
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        except OSError:
            return []
        found = []
        for line in out.splitlines():
            m = re.match(r"^(.+?)\s{2,}(\S+)\s+#", line)
            if m and m.group(2).startswith("en"):
                found.append(m.group(1).strip())
        return found

    def _pick_voice(self) -> str:
        available = set(self.voices())
        for v in PREFERRED:
            if v in available:
                return v
        return next(iter(available), "Samantha")

    # ------------------------------------------------------------------ direct
    def _mark_up(self, s: Sentence) -> str:
        """Translate direction into embedded speech commands."""
        text = s.text

        # Emphasis: wrap the chosen words. [[emph +]] applies to the next word.
        for word in s.emphasis:
            pattern = re.compile(rf"(?<!\w)({re.escape(word)})(?!\w)", re.I)
            text = pattern.sub(r"[[emph +]]\1[[emph -]]", text, count=1)

        wpm = int(self.wpm * s.pace)
        # Emotion nudges pitch and volume; crude, but audibly better than flat.
        pitch, volume = {
            "excited":       (+8, 1.0),
            "urgent":        (+5, 1.0),
            "curious":       (+6, 0.97),
            "warm":          (+2, 0.95),
            "tender":        (-2, 0.85),
            "somber":        (-6, 0.85),
            "ominous":       (-8, 0.82),
            "reflective":    (-3, 0.88),
            "wry":           (+3, 0.92),
            "authoritative": (-4, 1.0),
            "tense":         (+2, 0.93),
        }.get(s.emotion, (0, 0.95))

        return f"[[rate {wpm}]][[pbas {50 + pitch}]][[volm {volume:.2f}]] {text}"

    # ------------------------------------------------------------------- synth
    def synth(self, sentence: Sentence) -> Clip:
        self._n += 1
        out = self._tmp / f"{self._n:06d}.wav"
        marked = self._mark_up(sentence)
        try:
            subprocess.run(
                ["say", "-v", self.voice, "-o", str(out),
                 "--data-format=LEI16@22050", marked],
                capture_output=True, check=True, timeout=120,
            )
            data, sr = sf.read(out, dtype="float32", always_2d=False)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError):
            return Clip(np.zeros(0, dtype=np.float32), self.rate)
        finally:
            out.unlink(missing_ok=True)

        if data.ndim > 1:
            data = data.mean(axis=1)
        return Clip(np.ascontiguousarray(data, dtype=np.float32), int(sr))

    def close(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
