"""Acoustic stress for TRACE (arXiv 2609.29452): stressed copies of a caller clip, one per arm.

Noise levels are set on whole-clip power, pauses included. Every arm is seeded from the clip key, so reruns match.
"""

from __future__ import annotations

import hashlib

import numpy as np

from speech.audio import SAMPLE_RATE

ARMS = ["clean", "white@20", "white@10", "white@5", "babble@20", "babble@10", "babble@5", "reverb@0.3",
        "reverb@0.8", "competing"]
BABBLE_TALKERS = 4
COMPETING_SIR_DB = 5.0


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    ps = np.mean(signal.astype(np.float64) ** 2)
    pn = np.mean(noise.astype(np.float64) ** 2)
    return 10 * np.log10(ps / pn)


def _fit(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Tile x to at least n samples and cut a random window, so every clip hears a different stretch."""
    x = np.asarray(x, dtype=np.float64)
    reps = int(np.ceil((n + len(x)) / max(1, len(x))))
    tiled = np.tile(x, reps)
    start = int(rng.integers(0, len(tiled) - n + 1))
    return tiled[start:start + n]


def add_noise(clean: np.ndarray, noise: np.ndarray, snr: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    c = clean.astype(np.float64)
    nz = _fit(noise, len(c), rng)
    scale = np.sqrt(np.mean(c ** 2) / (np.mean(nz ** 2) * 10 ** (snr / 10)))
    return np.clip(c + scale * nz, -32768, 32767).astype(np.int16)


def rir(rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Synthetic room impulse: a direct path and exponentially decaying noise, 60 dB down at rt60."""
    t = np.arange(int(1.5 * rt60 * SAMPLE_RATE)) / SAMPLE_RATE
    h = rng.standard_normal(len(t)) * np.exp(-6.908 * t / rt60) * 0.3
    h[0] = 1.0
    return h


def reverb(clean: np.ndarray, rt60: float, rng: np.random.Generator) -> np.ndarray:
    c = clean.astype(np.float64)
    h = rir(rt60, rng)
    n = len(c) + len(h) - 1
    y = np.fft.irfft(np.fft.rfft(c, n) * np.fft.rfft(h, n), n)[: len(c)]
    y *= np.sqrt(np.mean(c ** 2) / max(np.mean(y ** 2), 1e-9))
    return np.clip(y, -32768, 32767).astype(np.int16)


def _rng(key: str, arm: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1(f"{key}|{arm}".encode()).hexdigest()[:12], 16))


def stress(pcm: np.ndarray, arm: str, key: str, talkers: list[np.ndarray] | None = None) -> np.ndarray:
    if arm == "clean":
        return pcm
    rng = _rng(key, arm)
    kind, _, level = arm.partition("@")
    if kind == "white":
        return add_noise(pcm, rng.standard_normal(len(pcm)), float(level), rng)
    if kind == "reverb":
        return reverb(pcm, float(level), rng)
    if not talkers:
        raise ValueError(f"{arm} needs other talkers")
    if kind == "babble":
        picks = rng.choice(len(talkers), size=min(BABBLE_TALKERS, len(talkers)), replace=False)
        noise = sum(_fit(talkers[i], len(pcm), rng) for i in picks)
        return add_noise(pcm, noise, float(level), rng)
    if kind == "competing":
        # One other caller line at 5 dB below the caller, the way a TV or a second person sounds.
        other = talkers[int(rng.integers(len(talkers)))]
        return add_noise(pcm, other, COMPETING_SIR_DB, rng)
    raise ValueError(f"unknown arm {arm}")
