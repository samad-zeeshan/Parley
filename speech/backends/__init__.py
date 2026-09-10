"""Speech configurations by name: which recognizer, which prompt, which voice. PARLEY_SPEECH picks one."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..asr import DOMAIN_PROMPT, MIXED_PROMPT


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeechConfig:
    name: str
    description: str
    asr_model: str | None
    prompt: str | None = MIXED_PROMPT
    segmented: bool = False
    available: bool = True


CONFIGS = {c.name: c for c in [
    SpeechConfig("local", "faster-whisper small, mixed-language prompt, Piper", "small"),
    SpeechConfig("local-v1", "faster-whisper small, v1 domain prompt, Piper", "small", DOMAIN_PROMPT),
    SpeechConfig("local-2pass", "faster-whisper small, mixed prompt, decode split at pauses", "small", segmented=True),
    SpeechConfig("local-2pass-v1", "faster-whisper small, v1 prompt, decode split at pauses", "small", DOMAIN_PROMPT,
                 segmented=True),
    SpeechConfig("local-base", "faster-whisper base, mixed-language prompt, Piper", "base"),
    SpeechConfig("local-tiny", "faster-whisper tiny, mixed-language prompt, Piper", "tiny"),
    # Listed so every results table carries its column. No hosted client is in this build and no
    # hosted speech credentials exist on the development machine, so its cells read "not run".
    SpeechConfig("hosted", "hosted ASR and TTS", None, None, available=False),
]}


@dataclass
class Backend:
    config: SpeechConfig
    asr: object
    tts: object


def load(name: str | None = None) -> Backend:
    name = name or os.environ.get("PARLEY_SPEECH", "local")
    if name not in CONFIGS:
        raise ValueError(f"unknown speech config {name!r}; one of {sorted(CONFIGS)}")
    cfg = CONFIGS[name]
    if not cfg.available:
        raise BackendUnavailable(f"{name}: not available in this build")
    from .local import build

    return build(cfg)
