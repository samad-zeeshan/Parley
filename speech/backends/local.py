"""The offline stack: faster-whisper on CPU for ASR and Piper for the agent's voice."""

from __future__ import annotations


def build_asr(cfg):
    from ..asr import SegmentedASR, WhisperASR

    asr = WhisperASR(cfg.asr_model, prompt=cfg.prompt)
    return SegmentedASR(asr) if cfg.segmented else asr


def build(cfg):
    from ..tts import PiperTTS
    from . import Backend

    return Backend(cfg, build_asr(cfg), PiperTTS())
