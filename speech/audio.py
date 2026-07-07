"""Small PCM helpers. All pipeline audio is 16 kHz mono int16."""

from __future__ import annotations

import io
import subprocess
import wave

import numpy as np

SAMPLE_RATE = 16_000


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    if src_rate == dst_rate or len(pcm) == 0:
        return pcm.astype(np.int16)
    n_out = int(round(len(pcm) * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm.astype(np.float64)).astype(np.int16)


def silence(seconds: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.int16)


def read_wav(path_or_bytes) -> tuple[np.ndarray, int]:
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    with wave.open(src, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {8 * width}-bit")
    pcm = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return pcm, rate


def write_wav(path_or_buffer, pcm: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    with wave.open(path_or_buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype(np.int16).tobytes())


def wav_bytes(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    write_wav(buf, pcm, rate)
    return buf.getvalue()


def decode_any(data: bytes) -> np.ndarray:
    """Decode any container ffmpeg understands (webm/ogg from a browser) to 16 kHz mono PCM."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=data, capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16)
