"""ElevenLabs. The most expressive option, and the only one that reads
the director's prose notes as prose.

    export ELEVENLABS_API_KEY=...

Two things here matter more than the model choice:

1. `previous_text` / `next_text`. The API will condition the delivery of this
   sentence on the surrounding ones. This single field is most of the
   difference between "sentences read in a row" and "a paragraph performed".
2. Audio tags. The v3 model honours inline directions like [thoughtful] or
   [whispering], so the director's emotion maps straight onto the voice.

Roughly $1-3 per hour of finished audio at the time of writing.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from ..document import Sentence
from .base import Clip, Engine

API = "https://api.elevenlabs.io/v1/text-to-speech"

# Warm, unhurried narration voices from the public library.
DEFAULT_VOICE = "N2lVS1w4EtoT3dr4eOWO"   # Callum — dark, measured
VOICE_ALIASES = {
    "callum": "N2lVS1w4EtoT3dr4eOWO",
    "charlotte": "XB0fDUnXU5powFXDhCwa",
    "george": "JBFqnCBsd6RMkjVDRZzb",
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "daniel": "onwK4e9ZLuTAKqWW03F9",
    "lily": "pFZP5JQG7iQjIQuC4Bku",
}

# The director's vocabulary, translated into v3 audio tags.
TAGS = {
    "warm": "[warmly]", "curious": "[curious]", "excited": "[excited]",
    "urgent": "[urgent]", "tense": "[tense]", "somber": "[somber]",
    "reflective": "[thoughtful]", "wry": "[wry]", "tender": "[gently]",
    "authoritative": "[with authority]", "ominous": "[ominous]",
}


class ElevenLabsEngine(Engine):
    name = "elevenlabs"
    rate = 24_000
    expressiveness = 3

    def __init__(self, voice: str | None = None, model: str = "eleven_v3",
                 stability: float = 0.45, similarity: float = 0.75,
                 style: float = 0.35, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set.\n"
                "  export ELEVENLABS_API_KEY=sk_...   (get one at elevenlabs.io)"
            )
        key = (voice or "").lower().strip()
        self.voice = VOICE_ALIASES.get(key, voice or DEFAULT_VOICE)
        self.model = model
        # Lower stability = more emotional range. Too low and it drifts between
        # sentences, which is worse than flat. 0.4-0.5 is the audiobook sweet spot.
        self.stability = stability
        self.similarity = similarity
        self.style = style
        self._prev = ""
        self._all: list[Sentence] = []

    def prime(self, sentences: list[Sentence]) -> None:
        """Give the engine the full script so it can look ahead."""
        self._all = sentences

    def _next_text(self, s: Sentence) -> str:
        for other in self._all:
            if other.index == s.index + 1:
                return other.text
        return ""

    def _decorate(self, s: Sentence) -> str:
        text = s.text
        tag = TAGS.get(s.emotion, "")
        if s.pace <= 0.85:
            tag += "[slowly]"
        elif s.pace >= 1.12:
            tag += "[quickly]"
        # Emphasis: capitalisation is the documented way to stress a word.
        for word in s.emphasis:
            idx = text.lower().find(word.lower())
            if idx >= 0:
                text = text[:idx] + text[idx:idx + len(word)].upper() + text[idx + len(word):]
        return f"{tag} {text}".strip() if tag else text

    def synth(self, sentence: Sentence) -> Clip:
        payload = {
            "text": self._decorate(sentence),
            "model_id": self.model,
            "previous_text": self._prev[-600:],
            "next_text": self._next_text(sentence)[:600],
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity,
                "style": self.style,
                "use_speaker_boost": True,
            },
        }
        url = f"{API}/{self.voice}?output_format=pcm_24000"
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"ElevenLabs {exc.code}: {detail}") from exc

        self._prev = (self._prev + " " + sentence.text)[-1200:]
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return Clip(pcm, self.rate)

    def voices(self) -> list[str]:
        req = urllib.request.Request("https://api.elevenlabs.io/v1/voices",
                                     headers={"xi-api-key": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
        except Exception:
            return sorted(VOICE_ALIASES)
        return [f"{v['name']:<20} {v['voice_id']}" for v in data.get("voices", [])]
