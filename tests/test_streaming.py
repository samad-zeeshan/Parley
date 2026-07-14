"""Streaming front end: VAD endpointing, the streaming recognizer, and the voice call loop.

Audio here is synthetic (noise bursts stand in for speech) and the ASR and TTS
are fakes, so the suite needs no model and no sound card.
"""

import numpy as np

from speech.asr import FakeASR, StreamingRecognizer
from speech.audio import SAMPLE_RATE
from speech.vad import FRAME, EnergyVAD

RNG = np.random.default_rng(0)


def burst(seconds, level=3000):
    return (RNG.standard_normal(int(seconds * SAMPLE_RATE)) * level).astype(np.int16)


def quiet(seconds, level=30):
    return (RNG.standard_normal(int(seconds * SAMPLE_RATE)) * level).astype(np.int16)


def frames(pcm):
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        yield pcm[i:i + FRAME]


def test_vad_onset_and_offset():
    vad = EnergyVAD()
    audio = np.concatenate([quiet(0.5), burst(1.0), quiet(1.0)])
    events = []
    for i, f in enumerate(frames(audio)):
        ev = vad.push(f)
        if ev:
            events.append((ev, i))
    assert [e for e, _ in events] == ["start", "end"]
    start_frame = events[0][1]
    assert 25 <= start_frame <= 25 + vad.onset_frames      # burst starts at frame 25 (0.5 s)
    end_frame = events[1][1]
    assert 75 <= end_frame <= 75 + vad.hangover_frames + 1  # burst ends at frame 75 (1.5 s)


def test_vad_ignores_a_click():
    vad = EnergyVAD()
    audio = np.concatenate([quiet(0.5), burst(0.02), quiet(0.5)])
    assert [e for e in (vad.push(f) for f in frames(audio)) if e] == []


def test_streaming_recognizer_emits_one_final_per_utterance():
    asr = FakeASR(["first utterance", "second utterance"])
    rec = StreamingRecognizer(asr)
    audio = np.concatenate([quiet(0.3), burst(0.8), quiet(0.8), burst(0.6), quiet(0.8)])
    finals = [e for f in frames(audio) for e in rec.push(f) if e.kind == "final"]
    assert [e.text for e in finals] == ["first utterance", "second utterance"]
    # The segment handed to ASR covers the burst plus a little padding, not the whole stream.
    assert 0.8 <= asr.seen_seconds[0] <= 0.8 + 0.8 + 0.3


def test_streaming_recognizer_emits_partials_when_asked():
    asr = FakeASR(["a longer utterance"], partial=True)
    rec = StreamingRecognizer(asr, partial_every=0.5)
    audio = np.concatenate([quiet(0.3), burst(1.6), quiet(0.8)])
    kinds = [e.kind for f in frames(audio) for e in rec.push(f)]
    assert kinds.count("partial") >= 2 and kinds[-1] == "final"
    assert kinds[0] == "speech_start"


# ---- the voice call loop -----------------------------------------------------

class FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, lang):
        from speech.tts import Speech

        self.calls.append(lang)
        return Speech(burst(0.5), f"fake-{lang}")


def test_voice_call_books_from_audio_with_fake_models(svc, conn, clock):
    from api.audit import verify_chain
    from dialogue.agent import Agent
    from eval.scripts import CALLS
    from speech.pipeline import VoiceCall

    call = CALLS["switch-booking"]
    asr = FakeASR([t["text"] for t in call["turns"]])
    tts = FakeTTS()
    vc = VoiceCall(Agent(svc, conn, clock, session_id="v1"), asr, tts)
    audio = np.concatenate([np.concatenate([quiet(0.7), burst(0.7)]) for _ in call["turns"]] + [quiet(0.8)])
    replies = [r for f in frames(audio) for r in vc.push(f)]
    assert len(replies) == len(call["turns"])
    assert replies[-1].turn.action == "confirmed"
    assert tts.calls == [r.turn.lang for r in replies]
    assert all(r.speech is not None and r.timings["total"] >= 0 for r in replies)
    assert conn.execute("select phone from bookings").fetchone()[0] == call["phone"]
    assert verify_chain(conn)
