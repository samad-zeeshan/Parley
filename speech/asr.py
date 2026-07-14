"""Streaming speech recognition.

faster-whisper is not a streaming model. Streaming here means: audio arrives in
20 ms frames, the VAD finds where an utterance starts and ends, partial
transcripts are produced by re-decoding the growing buffer every
`partial_every` seconds (optional, it costs CPU), and the final transcript is
decoded once when the VAD closes the utterance. See ADR 0003.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE
from .vad import FRAME, EnergyVAD

# Bilingual vocabulary prompt: area names and booking words in both scripts.
# It biases decoding toward the domain and contains no evaluation sentence (ADR 0001).
DOMAIN_PROMPT = ("Viewing booking in Dubai Marina, Al Barsha, JLT, Downtown, Al Reem Island, Khalifa City. "
                 "حجز معاينة شقة في دبي مارينا والبرشاء وجزيرة الريم، غرفتين، درهم، الساعة.")


@dataclass
class Transcript:
    text: str
    lang: str | None = None
    lang_prob: float = 0.0
    seconds: float = 0.0  # decode time


class WhisperASR:
    """faster-whisper on CPU. Model size from MAJLIS_ASR_MODEL (default small)."""

    def __init__(self, size: str | None = None, prompt: str | None = DOMAIN_PROMPT, compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.size = size or os.environ.get("MAJLIS_ASR_MODEL", "small")
        self.prompt = prompt
        threads = int(os.environ.get("MAJLIS_ASR_THREADS", "0")) or max(1, (os.cpu_count() or 4) - 2)
        self.model = WhisperModel(self.size, device="cpu", compute_type=compute_type, cpu_threads=threads)

    def transcribe(self, pcm: np.ndarray, partial: bool = False) -> Transcript:
        t0 = time.perf_counter()
        audio = pcm.astype(np.float32) / 32768.0
        segs, info = self.model.transcribe(audio, beam_size=1, vad_filter=False, initial_prompt=self.prompt,
                                           without_timestamps=True, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        return Transcript(text, info.language, info.language_probability, time.perf_counter() - t0)


class FakeASR:
    """Returns scripted transcripts in order. For tests and for timing the rest of the pipeline."""

    def __init__(self, texts: list[str], partial: bool = False, lang: str | None = None):
        self.texts = list(texts)
        self.partial = partial
        self.lang = lang
        self.seen_seconds: list[float] = []

    def transcribe(self, pcm: np.ndarray, partial: bool = False) -> Transcript:
        if partial:
            words = (self.texts[0] if self.texts else "").split()
            n = max(1, int(len(pcm) / SAMPLE_RATE / 0.4))
            return Transcript(" ".join(words[:n]), self.lang)
        self.seen_seconds.append(len(pcm) / SAMPLE_RATE)
        return Transcript(self.texts.pop(0) if self.texts else "", self.lang)


@dataclass
class ASREvent:
    kind: str                    # speech_start | partial | final
    text: str = ""
    lang: str | None = None
    audio_seconds: float = 0.0
    decode_seconds: float = 0.0
    pcm: np.ndarray | None = None


class StreamingRecognizer:
    def __init__(self, asr, vad: EnergyVAD | None = None, preroll: float = 0.3, partial_every: float | None = None):
        self.asr = asr
        self.vad = vad or EnergyVAD()
        self.preroll = deque(maxlen=int(preroll * SAMPLE_RATE / FRAME))
        self.partial_every = partial_every
        self.buffer: list[np.ndarray] = []
        self._since_partial = 0

    @property
    def in_speech(self) -> bool:
        return self.vad.in_speech

    def push(self, frame: np.ndarray, extra_db: float = 0.0) -> list[ASREvent]:
        events: list[ASREvent] = []
        ev = self.vad.push(frame, extra_db=extra_db)
        if ev == "start":
            self.buffer = list(self.preroll) + [frame]
            self._since_partial = 0
            events.append(ASREvent("speech_start"))
        elif self.vad.in_speech:
            self.buffer.append(frame)
            self._since_partial += 1
            if self.partial_every and self._since_partial * FRAME >= self.partial_every * SAMPLE_RATE:
                self._since_partial = 0
                pcm = np.concatenate(self.buffer)
                t = self.asr.transcribe(pcm, partial=True)
                events.append(ASREvent("partial", t.text, t.lang, len(pcm) / SAMPLE_RATE, t.seconds))
        elif ev == "end":
            self.buffer.append(frame)
            pcm = np.concatenate(self.buffer)
            t0 = time.perf_counter()
            t = self.asr.transcribe(pcm)
            events.append(ASREvent("final", t.text, t.lang, len(pcm) / SAMPLE_RATE,
                                   t.seconds or time.perf_counter() - t0, pcm))
            self.buffer = []
        self.preroll.append(frame)
        return events

    def flush(self) -> list[ASREvent]:
        """End of stream: close an open utterance."""
        if not self.vad.in_speech or not self.buffer:
            return []
        pcm = np.concatenate(self.buffer)
        self.buffer = []
        self.vad.reset()
        t = self.asr.transcribe(pcm)
        return [ASREvent("final", t.text, t.lang, len(pcm) / SAMPLE_RATE, t.seconds, pcm)]
