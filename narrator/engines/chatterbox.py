"""Chatterbox — expressive local TTS with voice cloning.

Heavier than Kokoro (PyTorch, roughly realtime on an M-series GPU) but it has
something the others lack locally: an explicit *exaggeration* control over
emotional intensity, and the ability to clone a voice from about ten seconds
of reference audio. If you want a specific person reading your books, this is
the engine.

Opt-in, because it pulls in PyTorch:

    uv add chatterbox-tts

To clone a voice, pass a reference clip:

    narrate book.pdf -e chatterbox --voice ~/voices/reference.wav
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..document import Sentence
from .base import Clip, Engine


class ChatterboxEngine(Engine):
    name = "chatterbox"
    rate = 24_000
    expressiveness = 3

    # How strongly each direction pushes the exaggeration control.
    # 0.5 is the model's neutral; much above 0.9 it starts to ham.
    INTENSITY = {
        "neutral": 0.45, "warm": 0.55, "curious": 0.6, "reflective": 0.5,
        "wry": 0.6, "authoritative": 0.6, "tender": 0.55, "somber": 0.55,
        "tense": 0.7, "ominous": 0.65, "urgent": 0.8, "excited": 0.8,
    }

    def __init__(self, voice: str | None = None, device: str | None = None):
        try:
            import torch
            from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            raise RuntimeError(
                "chatterbox is not installed.\n"
                "  uv add chatterbox-tts        (~2.5 GB, pulls PyTorch)"
            ) from exc

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._m = ChatterboxTTS.from_pretrained(device=device)
        self.rate = int(getattr(self._m, "sr", 24_000))

        # A voice here is a path to reference audio, not a name.
        self.voice = None
        if voice:
            p = Path(voice).expanduser()
            if not p.exists():
                raise RuntimeError(
                    f"reference audio not found: {p}\n"
                    "chatterbox clones a voice from a wav/mp3 clip of ~10 seconds"
                )
            self.voice = str(p)

    def synth(self, sentence: Sentence) -> Clip:
        exaggeration = self.INTENSITY.get(sentence.emotion, 0.5)
        # cfg_weight trades fidelity against pace; lowering it lets slow
        # direction actually slow the delivery rather than just stretch it.
        cfg = 0.3 if sentence.pace <= 0.9 else 0.5
        kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg}
        if self.voice:
            kwargs["audio_prompt_path"] = self.voice
        try:
            wav = self._m.generate(sentence.text, **kwargs)
        except Exception:
            return Clip(np.zeros(0, dtype=np.float32), self.rate)

        x = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        return Clip(x, self.rate)

    def voices(self) -> list[str]:
        return ["<path to a 10s reference clip, or omit for the default voice>"]

    def warmup(self) -> None:
        try:
            self._m.generate("Ready.")
        except Exception:
            pass
