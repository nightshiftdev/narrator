"""Kokoro-82M — the local workhorse.

82 million parameters, Apache licensed, runs several times faster than
realtime on Apple silicon, and sounds far better than its size suggests.
Nothing leaves the machine and nothing is metered.

It has no emotion input, so the director's work reaches it through the two
levers it does expose — speaking rate, and the pauses we place around it —
plus a light punctuation rewrite that gives the model the prosodic cues it
was trained on. That gets a surprising distance: most of what reads as
"expression" in narration is timing.

Weights (downloaded once, ~330 MB) live in ~/.cache/narrator/kokoro.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..document import Sentence
from .base import Clip, Engine

CACHE = Path.home() / ".cache" / "narrator" / "kokoro"
MODEL = CACHE / "kokoro-v1.0.onnx"
VOICES = CACHE / "voices-v1.0.bin"
RELEASE = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
           "model-files-v1.0")

# Voices worth using for long-form narration, best first.
PREFERRED = ["am_michael", "bm_george", "af_heart", "bf_emma", "am_fenrir",
             "af_bella", "am_puck"]


class KokoroEngine(Engine):
    name = "kokoro"
    rate = 24_000
    expressiveness = 1

    def __init__(self, voice: str | None = None, speed: float = 1.0,
                 dialogue_voice: str | None = None):
        self._ensure_weights()
        self._ensure_espeak()

        import logging
        # phonemizer chatters about word-count mismatches on every call
        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        logging.getLogger("kokoro_onnx").setLevel(logging.ERROR)

        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise RuntimeError(
                "the kokoro voice is not installed.\n"
                "  uv sync --extra kokoro       (or: pip install 'narrator[kokoro]')\n"
                "\n"
                "It is an optional extra because its dependency chain\n"
                "(kokoro-onnx -> phonemizer -> espeak-ng) is GPL-3.0."
            ) from exc

        self._k = Kokoro(str(MODEL), str(VOICES))
        available = set(self._k.get_voices())
        self.voice = voice if voice in available else next(
            (v for v in PREFERRED if v in available), sorted(available)[0]
        )
        self.base_speed = speed
        # Optional: a second voice for spoken lines. Kokoro cannot colour a
        # single voice, so casting is the only lever it has for dialogue.
        self.dialogue_voice = dialogue_voice if dialogue_voice in available else None

    # ---------------------------------------------------------------- setup
    @staticmethod
    def _ensure_weights() -> None:
        missing = [p for p in (MODEL, VOICES) if not p.exists()]
        if not missing:
            return
        names = ", ".join(p.name for p in missing)
        raise RuntimeError(
            f"Kokoro weights missing ({names}). Fetch them once with:\n"
            f"  mkdir -p {CACHE}\n"
            f"  curl -fL -o {MODEL} {RELEASE}/{MODEL.name}\n"
            f"  curl -fL -o {VOICES} {RELEASE}/{VOICES.name}"
        )

    @staticmethod
    def _ensure_espeak() -> None:
        """Point phonemizer at the pip-installed espeak-ng, so no Homebrew."""
        if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
            return
        try:
            import espeakng_loader
        except ImportError:
            return
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeakng_loader.get_library_path())
        os.environ["PHONEMIZER_ESPEAK_PATH"] = str(espeakng_loader.get_data_path())
        try:
            espeakng_loader.make_library_available()
        except Exception:
            pass

    # ------------------------------------------------------------- direction
    @staticmethod
    def _shape(s: Sentence) -> str:
        """Rewrite punctuation into cues Kokoro's prosody model responds to.

        The model learned English punctuation, not emotion labels. A comma
        buys a short breath; an ellipsis buys a longer, softer one; a full
        stop resets the contour. So we spend punctuation, carefully — never
        enough to change the words the listener hears.
        """
        text = s.text.strip()

        # A heading is an announcement: let it settle before the pause we add.
        if s.kind in ("title", "chapter", "heading") and not text.endswith((".", "?", "!")):
            text += "."

        # Weighty lines: a beat before the final clause, where a reader breathes.
        if s.pace <= 0.85 and "," in text and "..." not in text:
            head, sep, tail = text.rpartition(", ")
            if sep and len(tail) > 12:
                text = f"{head}...{' '}{tail}"

        # Kokoro has no stress token and no emotion input. Emphasis and
        # emotion are therefore not requested for it at all — see the
        # "timing" director profile.
        return text

    def synth(self, sentence: Sentence) -> Clip:
        text = self._shape(sentence)
        speed = float(np.clip(self.base_speed * sentence.pace, 0.6, 1.6))
        voice = (self.dialogue_voice if sentence.role == "speech"
                 and self.dialogue_voice else self.voice)
        try:
            samples, sr = self._k.create(text, voice=voice, speed=speed, lang="en-us")
        except Exception:
            return Clip(np.zeros(0, dtype=np.float32), self.rate)
        return Clip(np.asarray(samples, dtype=np.float32), int(sr))

    def voices(self) -> list[str]:
        try:
            return sorted(self._k.get_voices())
        except Exception:
            return list(PREFERRED)

    def warmup(self) -> None:
        try:
            self._k.create("Ready.", voice=self.voice, speed=1.0, lang="en-us")
        except Exception:
            pass
