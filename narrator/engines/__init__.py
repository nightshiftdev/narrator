"""Engine registry. Add a backend here and the CLI picks it up."""
from __future__ import annotations

from .base import Clip, Engine

_BUILDERS: dict[str, str] = {
    "macsay": "narrator.engines.macsay:MacSayEngine",
    "kokoro": "narrator.engines.kokoro:KokoroEngine",
    "elevenlabs": "narrator.engines.elevenlabs:ElevenLabsEngine",
    "chatterbox": "narrator.engines.chatterbox:ChatterboxEngine",
}

AVAILABLE = list(_BUILDERS)

# How much direction each backend can actually honour. Kept here as data so
# the director can size its work without importing (and loading) an engine.
EXPRESSIVENESS = {
    "macsay": 2,
    "kokoro": 1,
    "elevenlabs": 3,
    "chatterbox": 3,
}


def load_engine(name: str, **kwargs) -> Engine:
    if name not in _BUILDERS:
        raise ValueError(f"unknown engine {name!r}; choose from {', '.join(AVAILABLE)}")
    import importlib

    module_path, _, cls_name = _BUILDERS[name].partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RuntimeError(
            f"engine {name!r} is not installed: {exc}\n"
            f"see narrator/engines/{name}.py for the install command"
        ) from exc
    cls = getattr(module, cls_name)
    import inspect
    accepted = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in kwargs.items()
                  if v is not None and k in accepted})


__all__ = ["Clip", "Engine", "load_engine", "AVAILABLE", "EXPRESSIVENESS"]
