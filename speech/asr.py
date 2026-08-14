"""Speech recognition: faster-whisper on CPU, a pause-segmented variant, and the VAD-driven streaming front end.

Whisper is not a streaming model. Streaming here means VAD endpointing, optional partial re-decodes, and one final decode.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE
from .vad import FRAME, EnergyVAD

# v1 prompt: area names and booking words in both scripts, no evaluation sentence.
DOMAIN_PROMPT = ("Viewing booking in Dubai Marina, Al Barsha, JLT, Downtown, Al Reem Island, Khalifa City. "
                 "حجز معاينة شقة في دبي مارينا والبرشاء وجزيرة الريم، غرفتين، درهم، الساعة.")

# v2 prompt. Whisper copies the style of its prompt, so an example of Gulf Arabic with English left in
# Latin script makes it keep "Sunday" instead of writing "سندي". A test checks it shares no three-word
# run with any scripted line.
MIXED_PROMPT = ("Booking a viewing, studio or bedroom flat, budget in dirhams, JLT, Downtown, Business Bay, "
                "Saadiyat. OK يعني I need a flat قريب من the metro، والإيجار around ninety thousand. "
                "Monday بالليل or next week. تمام, book it for me.")


SHORT_CLIP_S = 1.5


@dataclass
class Transcript:
    text: str
    lang: str | None = None
    lang_prob: float = 0.0
    seconds: float = 0.0  # decode time


class WhisperASR:
    """faster-whisper on CPU. Model size from PARLEY_ASR_MODEL (default small)."""

    def __init__(self, size: str | None = None, prompt: str | None = DOMAIN_PROMPT, compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.size = size or os.environ.get("PARLEY_ASR_MODEL", "small")
        self.prompt = prompt
        threads = int(os.environ.get("PARLEY_ASR_THREADS", "0")) or max(1, (os.cpu_count() or 4) - 2)
        self.model = WhisperModel(self.size, device="cpu", compute_type=compute_type, cpu_threads=threads)

    def transcribe(self, pcm: np.ndarray, partial: bool = False, lang_hint: str | None = None) -> Transcript:
        t0 = time.perf_counter()
        audio = pcm.astype(np.float32) / 32768.0
        # Language detection on a clip under ~1.5 s is unreliable: "نعم" came back as "Love." in the
        # evaluation. For short clips the language of the conversation so far is used instead.
        language = lang_hint if lang_hint and len(pcm) < SHORT_CLIP_S * SAMPLE_RATE else None
        segs, info = self.model.transcribe(audio, beam_size=1, vad_filter=False, initial_prompt=self.prompt,
                                           without_timestamps=True, condition_on_previous_text=False,
                                           language=language)
        text = " ".join(s.text.strip() for s in segs).strip()
        return Transcript(text, info.language, info.language_probability, time.perf_counter() - t0)


def split_at_pauses(pcm: np.ndarray, min_gap_s: float = 0.15, min_db: float = 40.0, pad_frames: int = 3):
    """Cut a clip at silences of at least min_gap_s. Shorter gaps are inside words and stay joined."""
    from .vad import frame_db

    n = len(pcm) // FRAME
    voiced = [frame_db(pcm[k * FRAME:(k + 1) * FRAME]) > min_db for k in range(n)]
    gap = max(1, int(min_gap_s * SAMPLE_RATE / FRAME))
    spans, start, quiet = [], None, 0
    for k, v in enumerate(voiced):
        if v:
            start = k if start is None else start
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap:
                spans.append((start, k - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        spans.append((start, n - quiet))
    return [pcm[max(0, a - pad_frames) * FRAME:min(n, b + pad_frames) * FRAME] for a, b in spans]


class SegmentedASR:
    """Two-pass decode: cut the clip at pauses, then let Whisper pick the language of each piece.

    On this test set the TTS callers pause between languages, which flatters this method. Real speakers
    often switch with no pause at all.
    """

    def __init__(self, asr, min_gap_s: float = 0.15):
        self.asr = asr
        self.min_gap_s = min_gap_s

    def transcribe(self, pcm: np.ndarray, partial: bool = False, lang_hint: str | None = None) -> Transcript:
        chunks = split_at_pauses(pcm, self.min_gap_s) or [pcm]
        if len(chunks) == 1:
            return self.asr.transcribe(pcm, partial=partial, lang_hint=lang_hint)
        parts = [self.asr.transcribe(c, partial=partial, lang_hint=lang_hint) for c in chunks]
        text = " ".join(p.text for p in parts if p.text).strip()
        return Transcript(text, parts[0].lang, parts[0].lang_prob, sum(p.seconds for p in parts))


class FakeASR:
    """Returns scripted transcripts in order. For tests and for timing the rest of the pipeline."""

    def __init__(self, texts: list[str], partial: bool = False, lang: str | None = None):
        self.texts = list(texts)
        self.partial = partial
        self.lang = lang
        self.seen_seconds: list[float] = []

    def transcribe(self, pcm: np.ndarray, partial: bool = False, lang_hint: str | None = None) -> Transcript:
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
    hidden_seconds: float = 0.0  # decode time that runs inside the hangover, before the turn closes


class StreamingRecognizer:
    def __init__(self, asr, vad: EnergyVAD | None = None, preroll: float = 0.3, partial_every: float | None = None,
                 early_final: bool = False):
        self.asr = asr
        self.lang_hint: str | None = None
        self.vad = vad or EnergyVAD()
        self.preroll = deque(maxlen=int(preroll * SAMPLE_RATE / FRAME))
        self.partial_every = partial_every
        # A live system starts a speculative decode on every first quiet frame and drops it if the caller
        # goes on. Only the decode started when the last hangover opened is ever used, so this decodes
        # exactly that audio once, at the end, and reports how much of it the hangover hides.
        self.early_final = early_final
        self.buffer: list[np.ndarray] = []
        self._since_partial = 0
        self._quiet_from = 0

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
            if self.vad._quiet == 1:
                self._quiet_from = len(self.buffer) - 1
            self._since_partial += 1
            if self.partial_every and self._since_partial * FRAME >= self.partial_every * SAMPLE_RATE:
                self._since_partial = 0
                pcm = np.concatenate(self.buffer)
                t = self.asr.transcribe(pcm, partial=True)
                events.append(ASREvent("partial", t.text, t.lang, len(pcm) / SAMPLE_RATE, t.seconds))
        elif ev == "end":
            self.buffer.append(frame)
            hidden = 0.0
            if self.early_final and 0 < self._quiet_from < len(self.buffer):
                hidden = (len(self.buffer) - self._quiet_from) * FRAME / SAMPLE_RATE
                pcm = np.concatenate(self.buffer[:self._quiet_from])
            else:
                pcm = np.concatenate(self.buffer)
            t0 = time.perf_counter()
            t = self.asr.transcribe(pcm, lang_hint=self.lang_hint)
            events.append(ASREvent("final", t.text, t.lang, len(pcm) / SAMPLE_RATE,
                                   t.seconds or time.perf_counter() - t0, pcm, hidden))
            self.buffer = []
            self._quiet_from = 0
        self.preroll.append(frame)
        return events

    def flush(self) -> list[ASREvent]:
        """End of stream: close an open utterance."""
        if not self.vad.in_speech or not self.buffer:
            return []
        pcm = np.concatenate(self.buffer)
        self.buffer = []
        self.vad.reset()
        t = self.asr.transcribe(pcm, lang_hint=self.lang_hint)
        return [ASREvent("final", t.text, t.lang, len(pcm) / SAMPLE_RATE, t.seconds, pcm)]
