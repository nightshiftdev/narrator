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

# Measured once per English voice on a fixed line: (median F0 in Hz, pitch
# spread in semitones). Used to find directions in style space, not to pick a
# voice — see _build_axes.
VOICE_STATS: dict[str, tuple[float, float]] = {
    "af_alloy": (143.7, 3.30),
    "af_aoede": (175.2, 4.49),
    "af_bella": (201.7, 3.25),
    "af_heart": (193.5, 4.20),
    "af_jessica": (205.1, 6.19),
    "af_kore": (148.6, 4.30),
    "af_nicole": (162.7, 4.28),
    "af_nova": (158.9, 3.86),
    "af_river": (180.5, 3.33),
    "af_sarah": (192.0, 3.54),
    "af_sky": (162.2, 5.01),
    "am_adam": (120.3, 4.96),
    "am_echo": (103.0, 5.14),
    "am_eric": (156.9, 4.41),
    "am_fenrir": (133.3, 4.58),
    "am_liam": (118.2, 5.46),
    "am_michael": (114.8, 3.59),
    "am_onyx": (85.6, 4.35),
    "am_puck": (113.7, 3.72),
    "am_santa": (208.7, 5.69),
    "bf_alice": (218.2, 4.54),
    "bf_emma": (177.8, 3.15),
    "bf_isabella": (205.1, 2.74),
    "bf_lily": (181.8, 4.24),
    "bm_daniel": (127.7, 3.59),
    "bm_fable": (114.3, 4.84),
    "bm_george": (143.7, 4.35),
    "bm_lewis": (94.7, 5.86),
}


# Emotion as an offset along those two directions, in units of the base
# voice's norm. Small: past about 0.2 the speaker stops sounding like the
# same person. (pitch, animation)
EMOTION_STYLE: dict[str, tuple[float, float]] = {
    "neutral":       (0.00,  0.00),
    "warm":          (0.015, 0.035),
    "curious":       (0.035, 0.055),
    "excited":       (0.065, 0.095),
    "urgent":        (0.055, 0.080),
    "tense":         (0.030, 0.055),
    "wry":           (0.020, 0.040),
    "authoritative": (-0.025, 0.015),
    "reflective":    (-0.035, 0.015),
    "tender":        (-0.020, -0.010),
    "somber":        (-0.060, -0.025),
    "ominous":       (-0.080, -0.015),
}

# The offsets above are deliberately small. Stepping along these axes moves
# the style vector away from anything the model was trained on, and the voice
# gets rough as it goes: measured over one chapter, jitter tracked offset size
# almost exactly, and the largest offsets produced an audible resonance that a
# listener reported as a bug. Roughly two thirds of the original values keeps
# the emotional range while bringing jitter back to the neutral baseline.

# Spoken lines sit slightly forward of the narration around them: a shade
# brighter and more animated, the way a reader lifts into a character.
SPEECH_LIFT = (0.035, 0.05)


# Voices worth using for long-form narration, best first.
PREFERRED = ["am_michael", "bm_george", "af_heart", "bf_emma", "am_fenrir",
             "af_bella", "am_puck"]


class KokoroEngine(Engine):
    name = "kokoro"
    rate = 24_000
    # Kokoro takes no emotion input, but its style vector *is* a delivery
    # control: moving along measured pitch/animation directions changes how a
    # line is performed while leaving the speaker recognisably the same.
    expressiveness = 2

    def __init__(self, voice: str | None = None, speed: float = 1.0,
                 dialogue_voice: str | None = None,
                 cast: "object | None" = None,
                 pov_voices: dict[str, str] | None = None):
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
        self._voice_names = available
        self.voice = voice if voice in available else next(
            (v for v in PREFERRED if v in available), sorted(available)[0]
        )
        self.base_speed = speed
        self._pitch_axis, self._anim_axis = self._build_axes()
        self._style_cache: dict[tuple, np.ndarray] = {}
        self.retries = 0
        self.rough_saved = 0.0
        # Casting: a per-character voice map, and a narrating voice per
        # point-of-view section for books that change narrator.
        self.cast = cast
        self.pov_voices = {k.upper(): v for k, v in (pov_voices or {}).items()
                           if v in available}
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

    # ---------------------------------------------------------- style space
    def _build_axes(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Find the directions in style space that carry pitch and liveliness.

        Kokoro conditions delivery on a per-voice style vector. Regressing
        those vectors against measured pitch and pitch-spread gives two
        directions that behave like performance controls: step along them and
        the same speaker reads the line higher/lower and livelier/flatter.

        Derived from the local voice file, so it costs no synthesis.
        """
        names = [v for v in VOICE_STATS if v in self._voice_names]
        if len(names) < 8:
            return None, None
        try:
            styles = np.stack([self._k.get_voice_style(v) for v in names])
        except Exception:
            return None, None
        centroid = styles.mean(0)

        def direction(values: np.ndarray) -> np.ndarray | None:
            spread = values.std()
            if spread < 1e-6:
                return None
            z = (values - values.mean()) / spread
            axis = (z[:, None, None, None] * (styles - centroid)).sum(0)
            norm = np.linalg.norm(axis)
            return axis / norm if norm > 0 else None

        f0 = np.log(np.array([VOICE_STATS[v][0] for v in names]))
        spread = np.array([VOICE_STATS[v][1] for v in names])
        return direction(f0), direction(spread)

    def _base_voice(self, sentence: Sentence) -> str:
        """Whose voice reads this line: a cast character, or the narrator."""
        if self.cast is not None:
            v = self.cast.voice_for(sentence)
            if v and v in self._voice_names:
                return v
        if sentence.pov and sentence.pov in self.pov_voices:
            return self.pov_voices[sentence.pov]
        return self.voice

    def _style_for(self, sentence: Sentence) -> "str | np.ndarray":
        """The style vector to read this sentence with."""
        base_voice = self._base_voice(sentence)
        if self._pitch_axis is None:
            return base_voice
        if self.cast is not None and self.cast.is_flat(sentence):
            # affectless by instruction: no emotion, no lift, no heading colour
            return base_voice
        pitch, anim = EMOTION_STYLE.get(sentence.emotion, (0.0, 0.0))
        cast_voiced = base_voice != self.voice
        if sentence.role == "speech" and not cast_voiced:
            # only lift when the narrator is doing the voice themselves
            pitch += SPEECH_LIFT[0]
            anim += SPEECH_LIFT[1]
        elif sentence.role == "tag":
            # an attribution drops back out of the character and under the line
            pitch -= 0.03
            anim -= 0.05
        if sentence.kind in ("title", "chapter", "heading"):
            pitch, anim = pitch - 0.04, anim + 0.02

        key = (base_voice, round(pitch, 3), round(anim, 3))
        if key[1] == 0.0 and key[2] == 0.0:
            return base_voice
        cached = self._style_cache.get(key)
        if cached is None:
            base = self._k.get_voice_style(base_voice)
            scale = float(np.linalg.norm(base))
            cached = base + scale * (key[1] * self._pitch_axis
                                     + key[2] * self._anim_axis)
            self._style_cache[key] = cached
        return cached

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

    # Above this, a clip is rough enough to be heard as a defect rather than
    # as a voice. Measured across a chapter, a clean line sits near 4-5%.
    ROUGH = 8.0
    # At or below this many words, an utterance is mostly its own ending.
    SHORT_WORDS = 3
    # Spoken after a short line so the lengthening lands on words we discard.
    CARRIER = "He said nothing more."

    def synth(self, sentence: Sentence) -> Clip:
        if len(sentence.text.split()) <= self.SHORT_WORDS:
            carried = self._synth_short(sentence)
            if carried is not None:
                return carried
        clip = self._synth_at(sentence, 1.0)
        if not len(clip.samples):
            return clip
        from .. import audio as A
        rough = A.jitter(clip.samples, clip.rate)
        if rough <= self.ROUGH:
            return clip
        # Kokoro's roughness is not a property of the text: nudging the rate
        # by a few percent re-rolls the generation, and one of the takes is
        # usually clean. The change in pace is far below what a listener can
        # detect; the change in roughness is not.
        best, best_rough = clip, rough
        for nudge in (1.04, 0.96):
            alt = self._synth_at(sentence, nudge)
            if not len(alt.samples):
                continue
            alt_rough = A.jitter(alt.samples, alt.rate)
            if alt_rough < best_rough:
                best, best_rough = alt, alt_rough
            if best_rough <= self.ROUGH:
                break
        self.retries += 1
        self.rough_saved += rough - best_rough
        return best

    def _synth_short(self, sentence: Sentence) -> "Clip | None":
        """Speak a very short line without the drawl it gets on its own.

        The model lengthens the end of an utterance, and a one-word line is
        nothing but its end: "No." runs at 1.46 words per second against 2.76
        for prose, which a listener hears as an awkward drag. Speaking it as
        the first sentence of a longer one puts the lengthening on words that
        are then discarded, and the line comes back at conversational speed.

        The cut is only taken when a real silence is found where the boundary
        must be; otherwise the plain rendering stands, because a cut in the
        wrong place clips a word and that is far worse than a slow one.
        """
        from .. import audio as A

        plain = self._synth_at(sentence, 1.0)
        if not len(plain.samples):
            return None
        alone = plain.duration
        carried = Sentence(
            f"{sentence.text.rstrip()} {self.CARRIER}", sentence.block_index,
            sentence.kind, emotion=sentence.emotion, pace=sentence.pace,
            pause_after=sentence.pause_after, role=sentence.role,
            speaker=sentence.speaker, pov=sentence.pov,
        )
        long = self._synth_at(carried, 1.0)
        if not len(long.samples):
            return None
        cut = A.first_gap(long.samples, long.rate,
                          lo=0.30 * alone, hi=1.30 * alone)
        if cut is None:
            return None
        return Clip(np.ascontiguousarray(long.samples[:cut]), long.rate)

    def _synth_at(self, sentence: Sentence, nudge: float) -> Clip:
        text = self._shape(sentence)
        speed = float(np.clip(self.base_speed * sentence.pace * nudge, 0.6, 1.6))
        voice = (self.dialogue_voice if sentence.role == "speech"
                 and self.dialogue_voice else self._style_for(sentence))
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
