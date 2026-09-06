"""Mastering: level, fades, silence, and container writing.

Everything here is stdlib + numpy + soundfile + macOS's own `afconvert`,
so there is no ffmpeg or Homebrew dependency.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resample. Adequate here: engines are consistent within a run."""
    if src == dst or len(x) == 0:
        return x
    n = int(round(len(x) * dst / src))
    src_idx = np.linspace(0.0, len(x) - 1, n, dtype=np.float64)
    return np.interp(src_idx, np.arange(len(x)), x).astype(np.float32)


def trim_silence(x: np.ndarray, rate: int, threshold_db: float = -45.0,
                 keep_ms: int = 40) -> np.ndarray:
    """Strip engine-added lead-in/tail so *our* pauses set the rhythm."""
    if len(x) == 0:
        return x
    thresh = 10 ** (threshold_db / 20)
    win = max(1, rate // 200)
    env = np.abs(x)
    # cheap moving max
    pad = np.pad(env, (0, (-len(env)) % win))
    frames = pad.reshape(-1, win).max(axis=1)
    loud = np.flatnonzero(frames > thresh)
    if len(loud) == 0:
        return x[:0]
    keep = int(rate * keep_ms / 1000)
    start = max(0, loud[0] * win - keep)
    end = min(len(x), (loud[-1] + 1) * win + keep)
    return x[start:end]


def fade(x: np.ndarray, rate: int, ms: float = 8.0) -> np.ndarray:
    """Tiny in/out ramps so concatenation never clicks."""
    n = min(int(rate * ms / 1000), len(x) // 2)
    if n <= 0:
        return x
    x = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def _k_weight(x: np.ndarray, rate: int) -> np.ndarray:
    """Rough ITU-R BS.1770 pre-filter: shelf + high-pass, biquads by hand."""
    def biquad(sig, b, a):
        out = np.zeros_like(sig)
        x1 = x2 = y1 = y2 = 0.0
        for i, s in enumerate(sig):
            y = b[0] * s + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            out[i] = y
            x2, x1 = x1, s
            y2, y1 = y1, y
        return out

    # Only used on a decimated signal, so the python loop stays cheap.
    step = max(1, rate // 8000)
    sig = x[::step].astype(np.float64)
    sig = biquad(sig, [1.53512485958697, -2.69169618940638, 1.19839281085285],
                 [1.0, -1.69065929318241, 0.73248077421585])
    sig = biquad(sig, [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])
    return sig


def loudness_lufs(x: np.ndarray, rate: int) -> float:
    """Integrated loudness, gated. Close enough to EBU R128 for our purpose."""
    if len(x) < rate // 10:
        return -70.0
    y = _k_weight(x, rate)
    r = 8000
    block = int(0.4 * r)
    hop = block // 4
    if len(y) < block:
        return -0.691 + 10 * np.log10(np.mean(y ** 2) + 1e-12)
    blocks = np.array([
        np.mean(y[i:i + block] ** 2) for i in range(0, len(y) - block, hop)
    ])
    lk = -0.691 + 10 * np.log10(blocks + 1e-12)
    keep = lk > -70.0
    if not keep.any():
        return -70.0
    relative = -0.691 + 10 * np.log10(np.mean(blocks[keep]) + 1e-12) - 10.0
    keep &= lk > relative
    if not keep.any():
        return -70.0
    return float(-0.691 + 10 * np.log10(np.mean(blocks[keep]) + 1e-12))


def normalise_loudness(x: np.ndarray, rate: int, target_lufs: float = -19.0,
                       peak_ceiling: float = 0.95) -> np.ndarray:
    """Audiobook convention is about -18 to -20 LUFS with headroom."""
    if len(x) == 0:
        return x
    current = loudness_lufs(x, rate)
    gain = 10 ** ((target_lufs - current) / 20) if current > -70 else 1.0
    y = x * gain
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak > peak_ceiling:
        y *= peak_ceiling / peak
    return y.astype(np.float32)


def write_wav(path: Path | str, x: np.ndarray, rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, rate, subtype="PCM_16")


def to_m4b(wav: Path, out: Path, *, bitrate: int = 64_000,
           chapters: list[tuple[str, float]] | None = None,
           title: str = "", author: str = "Narrator") -> Path:
    """Encode to a chaptered audiobook using macOS's built-in afconvert."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["afconvert", "-f", "m4bf", "-d", "aac", "-b", str(bitrate),
           "-q", "127", "-s", "3", str(wav), str(out)]
    subprocess.run(cmd, check=True, capture_output=True)

    if chapters:
        _write_chapter_sidecar(out.with_suffix(".chapters.txt"), chapters)
    return out


def _write_chapter_sidecar(path: Path, chapters: list[tuple[str, float]]) -> None:
    """Simple, human-readable chapter marks (also importable by most players)."""
    lines = []
    for title, start in chapters:
        h, rem = divmod(int(start), 3600)
        m, s = divmod(rem, 60)
        ms = int((start - int(start)) * 1000)
        lines.append(f"{h:02d}:{m:02d}:{s:02d}.{ms:03d} {title}")
    path.write_text("\n".join(lines) + "\n")
