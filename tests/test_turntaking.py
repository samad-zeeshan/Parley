"""Turn taking as a finite-state machine: transitions, the barge-in bound and echo immunity."""

import numpy as np
import pytest

from dialogue.turntaking import BadTransition, State, TurnTaking
from speech.audio import SAMPLE_RATE
from speech.vad import FRAME

RNG = np.random.default_rng(3)


def loud(frames=1, level=3000):
    return [(RNG.standard_normal(FRAME) * level).astype(np.int16) for _ in range(frames)]


def quiet(frames=1, level=20):
    return [(RNG.standard_normal(FRAME) * level).astype(np.int16) for _ in range(frames)]


def run(fsm, frames, out=None):
    for f in frames:
        fsm.mic(f, out)


def test_starts_listening():
    assert TurnTaking().state is State.LISTENING


def test_caller_turn_goes_listening_speaking_endpointing_thinking():
    fsm = TurnTaking()
    run(fsm, quiet(10) + loud(5))
    assert fsm.state is State.CALLER_SPEAKING
    run(fsm, quiet(3))
    assert fsm.state is State.ENDPOINTING
    run(fsm, quiet(fsm.vad.hangover_frames))
    assert fsm.state is State.THINKING
    assert [t[1] for t in fsm.history] == [State.CALLER_SPEAKING, State.ENDPOINTING, State.THINKING]


def test_speech_during_the_hangover_resumes_the_same_turn():
    fsm = TurnTaking()
    run(fsm, quiet(10) + loud(5) + quiet(5) + loud(2))
    assert fsm.state is State.CALLER_SPEAKING
    assert fsm.turns_closed == 0


def test_reply_then_playback_done_returns_to_listening():
    fsm = TurnTaking()
    run(fsm, quiet(10) + loud(5) + quiet(fsm.vad.hangover_frames + 3))
    fsm.reply_ready()
    assert fsm.state is State.AGENT_SPEAKING
    fsm.playback_done()
    assert fsm.state is State.LISTENING


def test_reply_ready_outside_thinking_is_rejected():
    with pytest.raises(BadTransition):
        TurnTaking().reply_ready()


def test_barge_in_stops_playback_within_the_bound():
    fsm = TurnTaking()
    fsm.state = State.AGENT_SPEAKING
    agent_frame = loud(1, level=2500)[0]
    run(fsm, quiet(20), agent_frame)
    frames_to_stop = 0
    for f in loud(10):
        frames_to_stop += 1
        fsm.mic(f, agent_frame * 0.25)
        if fsm.state is State.CALLER_SPEAKING:
            break
    assert fsm.barge_ins == 1
    assert frames_to_stop * FRAME / SAMPLE_RATE <= 0.1


def test_own_echo_never_barges_in():
    fsm = TurnTaking()
    fsm.state = State.AGENT_SPEAKING
    for f in loud(200, level=2500):
        fsm.mic((f * 0.25).astype(np.int16), f)
    assert fsm.state is State.AGENT_SPEAKING and fsm.barge_ins == 0


@pytest.mark.parametrize("seed", range(25))
def test_random_event_sequences_only_take_legal_transitions(seed):
    """Property test: any mix of mic frames and agent events keeps the machine legal."""
    rng = np.random.default_rng(seed)
    fsm = TurnTaking()
    for _ in range(600):
        r = rng.random()
        if r < 0.45:
            fsm.mic((rng.standard_normal(FRAME) * 3000).astype(np.int16), None)
        elif r < 0.9:
            out = (rng.standard_normal(FRAME) * 2500).astype(np.int16) if fsm.state is State.AGENT_SPEAKING else None
            fsm.mic((rng.standard_normal(FRAME) * 20).astype(np.int16), out)
        elif r < 0.95:
            try:
                fsm.reply_ready()
            except BadTransition:
                assert fsm.state is not State.THINKING
        else:
            try:
                fsm.playback_done()
            except BadTransition:
                assert fsm.state is not State.AGENT_SPEAKING
    for prev, nxt in zip([State.LISTENING] + [t[1] for t in fsm.history], [t[1] for t in fsm.history]):
        assert nxt in TurnTaking.TRANSITIONS[prev]
