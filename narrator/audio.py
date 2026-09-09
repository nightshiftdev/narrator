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


def _biquad_fft(sig: np.ndarray, b, a) -> np.ndarray:
    """Apply a biquad in the frequency domain.

    The direct form is a two-sample recurrence, which in Python costs one
    interpreted iteration per sample -- minutes over an audiobook. The same
    filter is a multiplication against its frequency response, which numpy
    does in one pass. Edge effects from circular convolution are irrelevant
    for a loudness measurement.
    """
    n = len(sig)
    spec = np.fft.rfft(sig, n)
    w = np.exp(-2j * np.pi * np.fft.rfftfreq(n))
    num = b[0] + b[1] * w + b[2] * w * w
    den = a[0] + a[1] * w + a[2] * w * w
    return np.fft.irfft(spec * (num / den), n)


def _k_weight(x: np.ndarray, rate: int) -> np.ndarray:
    """ITU-R BS.1770 pre-filter: a high shelf, then a high-pass."""
    step = max(1, rate // 8000)
    sig = x[::step].astype(np.float64)
    sig = _biquad_fft(sig, [1.53512485958697, -2.69169618940638, 1.19839281085285],
                      [1.0, -1.69065929318241, 0.73248077421585])
    sig = _biquad_fft(sig, [1.0, -2.0, 1.0],
                      [1.0, -1.99004745483398, 0.99007225036621])
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
    # one strided view over the signal instead of a slice per block
    starts = np.arange(0, len(y) - block, hop)
    idx = starts[:, None] + np.arange(block)[None, :]
    blocks = np.mean(y[idx] ** 2, axis=1)
    lk = -0.691 + 10 * np.log10(blocks + 1e-12)
    keep = lk > -70.0
    if not keep.any():
        return -70.0
    relative = -0.691 + 10 * np.log10(np.mean(blocks[keep]) + 1e-12) - 10.0
    keep &= lk > relative
    if not keep.any():
        return -70.0
    return float(-0.691 + 10 * np.log10(np.mean(blocks[keep]) + 1e-12))


def limit(x: np.ndarray, rate: int, ceiling: float = 0.89,
          attack_ms: float = 3.0, release_ms: float = 60.0) -> np.ndarray:
    """Hold peaks under the ceiling without pulling the whole programme down.

    Scaling the entire file to fit its loudest sample is what a naive master
    does, and it is badly wrong over hours: in one book, 21 samples out of 210
    million held the whole thing 3.2 dB below target. A limiter attenuates
    only around the overshoots, and does it smoothly enough not to pump.
    """
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak <= ceiling:
        return x

    # Required attenuation per sample, then smoothed: fast to duck, slow to
    # recover, so the gain change is inaudible.
    need = np.minimum(1.0, ceiling / np.maximum(np.abs(x), 1e-9))

    # Look ahead by the attack time so the duck starts before the peak.
    look = int(rate * attack_ms / 1000)
    if look:
        need = np.concatenate([need[look:], np.ones(look, dtype=need.dtype)])

    # The envelope is a one-pole recurrence, so it cannot be vectorised
    # directly. It is instead run on a control signal at 1/CTRL the sample
    # rate -- taking the minimum over each block, so no peak is escapes it --
    # and interpolated back up. A block of 8 samples is a third of a
    # millisecond, far finer than the 3 ms attack, so the result tracks the
    # per-sample version to within 0.01 while the loop runs eight times
    # shorter.
    CTRL = 8
    pad = (-len(need)) % CTRL
    padded = np.concatenate([need, np.ones(pad, dtype=need.dtype)])
    ctrl = padded.reshape(-1, CTRL).min(axis=1)

    a_att = float(np.exp(-CTRL / max(1.0, rate * attack_ms / 1000)))
    a_rel = float(np.exp(-CTRL / max(1.0, rate * release_ms / 1000)))
    env = np.empty_like(ctrl)
    cur = 1.0
    for i, want in enumerate(ctrl):
        coeff = a_att if want < cur else a_rel
        cur = coeff * cur + (1.0 - coeff) * want
        env[i] = cur

    centres = np.arange(len(ctrl)) * CTRL + CTRL // 2
    g = np.interp(np.arange(len(x)), centres, env, left=env[0], right=env[-1])
    y = x * g
    # Any residual overshoot is tiny; clamp it rather than re-scaling.
    return np.clip(y, -ceiling, ceiling).astype(np.float32)


def normalise_loudness(x: np.ndarray, rate: int, target_lufs: float = -19.0,
                       peak_ceiling: float = 0.89) -> np.ndarray:
    """Audiobook convention is about -18 to -20 LUFS with headroom.

    The ceiling is -1 dBFS, not the -0.45 dBFS that looks safe in the WAV:
    a lossy encoder reconstructs values between the samples it was given, so
    a file that peaks at 0.95 before AAC comes back over 1.0 after it, and
    clips. The extra headroom costs nothing, since loudness is set by the
    gain rather than by the peak.
    """
    if len(x) == 0:
        return x
    current = loudness_lufs(x, rate)
    gain = 10 ** ((target_lufs - current) / 20) if current > -70 else 1.0
    return limit(x * gain, rate, peak_ceiling)


def write_wav(path: Path | str, x: np.ndarray, rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, rate, subtype="PCM_16")


def to_m4b(wav: Path, out: Path, *, bitrate: int = 64_000,
           chapters: list[tuple[str, float]] | None = None,
           title: str = "", author: str = "",
           cover: Path | str | None = None) -> Path:
    """Encode to a chaptered audiobook using macOS's built-in afconvert."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["afconvert", "-f", "m4bf", "-d", "aac", "-b", str(bitrate),
           "-q", "127", "-s", "3", str(wav), str(out)]
    subprocess.run(cmd, check=True, capture_output=True)

    if chapters:
        _write_chapter_sidecar(out.with_suffix(".chapters.txt"), chapters)
        # afconvert writes no chapter atoms, so add them afterwards: a
        # QuickTime chapter track for Apple Books, and a Nero chpl for
        # everything else.
        try:
            from .mp4chapters import embed
            embed(out, chapters)
        except Exception:
            pass          # the audio is fine; only navigation is lost

    if title or author or cover:
        # Without these a player shows an untitled recording by nobody, and
        # shelves it with music rather than with books.
        try:
            from .mp4meta import write_tags
            write_tags(out, title=title, author=author, cover_path=cover)
        except Exception:
            pass
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


def jitter(x: np.ndarray, rate: int) -> float:
    """Cycle-to-cycle pitch instability, as a percentage.

    This is the standard measure of a rough voice, and it is what a listener
    hears as a "vibration" or "resonance" in a synthesised line. Kokoro
    produces it unpredictably: the same voice reading the same paragraph is
    clean in one sentence and rough in the next, and nothing in the text
    predicts which.

    Autocorrelation via FFT, so it costs little enough to run on every clip.
    """
    if len(x) < rate // 8:
        return 0.0
    win, hop = int(0.04 * rate), int(0.01 * rate)
    lo, hi = int(rate / 350), int(rate / 60)
    n = 1 << (2 * win - 1).bit_length()
    f0 = []
    for i in range(0, len(x) - win, hop):
        w = x[i:i + win]
        if np.sqrt((w ** 2).mean()) < 0.02:
            continue
        w = w - w.mean()
        spec = np.fft.rfft(w, n)
        ac = np.fft.irfft(spec * np.conj(spec), n)[:win]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if not len(seg):
            continue
        peak = int(np.argmax(seg)) + lo
        if ac[peak] / ac[0] < 0.3:
            continue
        f0.append(rate / peak)
    if len(f0) < 3:
        return 0.0
    f0 = np.asarray(f0)
    return float(np.mean(np.abs(np.diff(f0))) / np.mean(f0) * 100)


def first_gap(x: np.ndarray, rate: int, lo: float, hi: float,
              min_silence: float = 0.05) -> int | None:
    """First real silence between `lo` and `hi` seconds, or None.

    Used to cut a short line away from the carrier phrase spoken after it.
    A run this long is a pause between sentences rather than a stop consonant
    inside a word, which is the distinction that matters: cutting inside a
    word is worse than not cutting at all.
    """
    win = max(1, int(0.01 * rate))
    frames = len(x) // win
    if frames < 4:
        return None
    env = np.abs(x[:frames * win]).reshape(-1, win).max(axis=1)
    quiet = env < max(0.012, float(env.max()) * 0.05)
    a = max(1, int(lo / 0.01))
    b = min(frames, int(hi / 0.01))
    need = max(2, int(min_silence / 0.01))
    run = 0
    start = 0
    for i in range(a, b):
        if quiet[i]:
            if run == 0:
                start = i
            run += 1
            if run >= need:
                return int((start + 2) * win)
        else:
            run = 0
    return None
