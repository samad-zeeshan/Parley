"""The voice call loop: caller frames in, agent speech out.

    frames -> StreamingRecognizer (VAD + ASR) -> Agent.turn -> TTS in the reply language

Turn latency is measured from the moment the VAD closes the caller's utterance
(end of speech plus the hangover) to the moment the reply audio is ready. The
hangover is reported separately because it is a setting, not compute.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .asr import StreamingRecognizer
from .tts import Speech


@dataclass
class VoiceReply:
    transcript: str
    asr_lang: str | None
    turn: object            # dialogue.agent.TurnResult
    speech: Speech | None
    timings: dict = field(default_factory=dict)


class VoiceCall:
    def __init__(self, agent, asr, tts, recognizer: StreamingRecognizer | None = None):
        self.agent = agent
        self.tts = tts
        self.rec = recognizer or StreamingRecognizer(asr)

    def respond(self, transcript: str, asr_lang: str | None = None, asr_seconds: float = 0.0) -> VoiceReply:
        t0 = time.perf_counter()
        turn = self.agent.turn(transcript)
        t1 = time.perf_counter()
        speech = self.tts.synthesize(turn.text, turn.lang) if self.tts else None
        t2 = time.perf_counter()
        timings = {"asr": asr_seconds, "dialogue": t1 - t0, "tts": t2 - t1}
        timings["total"] = asr_seconds + (t2 - t0)
        return VoiceReply(transcript, asr_lang, turn, speech, timings)

    def push(self, frame: np.ndarray, extra_db: float = 0.0) -> list[VoiceReply]:
        self.rec.lang_hint = self.agent.lang
        out = []
        for ev in self.rec.push(frame, extra_db=extra_db):
            if ev.kind == "final" and ev.text.strip():
                out.append(self.respond(ev.text, ev.lang, ev.decode_seconds))
        return out

    def flush(self) -> list[VoiceReply]:
        return [self.respond(ev.text, ev.lang, ev.decode_seconds) for ev in self.rec.flush() if ev.text.strip()]

    def utterance(self, pcm: np.ndarray) -> VoiceReply:
        """Whole-utterance path (push-to-talk): no VAD, decode the clip as one turn."""
        t = self.rec.asr.transcribe(pcm, lang_hint=self.agent.lang)
        return self.respond(t.text, t.lang, t.seconds)
