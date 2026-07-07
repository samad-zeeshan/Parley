"""Offline text to speech.

Two engines were found to work on the reference machine without a GPU or a
network call after download (see docs/adr/0001-model-choices.md):

- Piper (onnx, CPU): en_US-lessac-medium for English, ar_JO-kareem-medium for
  Arabic. The only open Arabic Piper voice is a Jordanian speaker, so "Gulf"
  audio made with it is Gulf wording in a Levantine voice.
- Windows SAPI (System.Speech): English only on this machine (David, Zira).
  Used as a second English voice for evaluation variety.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, read_wav, resample

MODELS_DIR = Path(os.environ.get("MAJLIS_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))

PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "ar": "ar_JO-kareem-medium",
}


@dataclass
class Speech:
    pcm: np.ndarray  # 16 kHz mono int16
    voice: str

    @property
    def seconds(self) -> float:
        return len(self.pcm) / SAMPLE_RATE


class PiperTTS:
    """Loads one Piper voice per language on first use."""

    def __init__(self, models_dir: Path = MODELS_DIR / "piper", length_scale: float | None = None):
        self.models_dir = Path(models_dir)
        self.length_scale = length_scale
        self._voices: dict[str, object] = {}

    def _voice(self, lang: str):
        if lang not in self._voices:
            from piper import PiperVoice

            name = PIPER_VOICES[lang]
            self._voices[lang] = PiperVoice.load(str(self.models_dir / f"{name}.onnx"))
        return self._voices[lang]

    def synthesize(self, text: str, lang: str, length_scale: float | None = None) -> Speech:
        from piper import SynthesisConfig

        voice = self._voice(lang)
        scale = length_scale or self.length_scale
        cfg = SynthesisConfig(length_scale=scale) if scale else None
        chunks = [c.audio_int16_array for c in voice.synthesize(text, syn_config=cfg)]
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        rate = voice.config.sample_rate
        return Speech(resample(pcm, rate, SAMPLE_RATE), PIPER_VOICES[lang])


class SapiTTS:
    """Windows System.Speech voices through PowerShell. English only here."""

    def __init__(self, voice: str = "Microsoft Zira Desktop", rate: int = 0):
        self.voice = voice
        self.rate = rate

    def synthesize(self, text: str, lang: str = "en") -> Speech:
        if lang != "en":
            raise ValueError("no offline SAPI voice for this language on this machine")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.wav"
            txt = Path(tmp) / "in.txt"
            txt.write_text(text, encoding="utf-8")
            script = (
                "Add-Type -AssemblyName System.Speech;"
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                f"$s.SelectVoice('{self.voice}'); $s.Rate = {self.rate};"
                f"$s.SetOutputToWaveFile('{out}');"
                f"$s.Speak([IO.File]::ReadAllText('{txt}', [Text.Encoding]::UTF8)); $s.Dispose()"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True)
            pcm, rate = read_wav(str(out))
        return Speech(resample(pcm, rate, SAMPLE_RATE), self.voice)
