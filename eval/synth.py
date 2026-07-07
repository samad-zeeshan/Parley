"""Render synthetic caller audio from (language, text) segments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from speech.audio import silence, write_wav
from speech.tts import PiperTTS, SapiTTS

_piper: PiperTTS | None = None
_sapi: dict[str, SapiTTS] = {}


def render(segments: list[tuple[str, str]], voice: str = "piper", gap: float = 0.12,
           length_scale: float | None = None) -> np.ndarray:
    global _piper
    parts = [silence(0.25)]
    for lang, text in segments:
        if voice.startswith("sapi") and lang == "en":
            name = "Microsoft David Desktop" if voice == "sapi-david" else "Microsoft Zira Desktop"
            _sapi.setdefault(name, SapiTTS(name))
            parts.append(_sapi[name].synthesize(text, "en").pcm)
        else:
            _piper = _piper or PiperTTS()
            parts.append(_piper.synthesize(text, lang, length_scale=length_scale).pcm)
        parts.append(silence(gap))
    parts.append(silence(0.25))
    return np.concatenate(parts)


def render_to(path: Path, segments, voice: str = "piper", **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_wav(str(path), render(segments, voice, **kw))
    return path
