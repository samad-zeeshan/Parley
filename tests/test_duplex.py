"""Full duplex: the agent stops talking within a bounded time after the caller starts.

Two tests of the bound. The simulated one steps a shared 20 ms clock through the
mic and speaker streams, so the result is exact and repeatable. The threaded one
runs a real-time player and a real-time mic feeder on separate threads and
measures wall-clock time, which is what a caller would hear.
"""

import numpy as np
import pytest

from speech.audio import SAMPLE_RATE
from speech.duplex import BARGE_IN_BOUND_S, Playback, RealtimeDuplex, simulate

RNG = np.random.default_rng(1)


def speechlike(seconds, level=3000):
    # Noise bursts with 80 ms gaps every 400 ms, closer to syllables than one flat burst.
    n = int(seconds * SAMPLE_RATE)
    x = RNG.standard_normal(n) * level
    period, gap = int(0.4 * SAMPLE_RATE), int(0.08 * SAMPLE_RATE)
    for start in range(period - gap, n, period):
        x[start:start + gap] *= 0.01
    return x.astype(np.int16)


def silence(seconds):
    return (RNG.standard_normal(int(seconds * SAMPLE_RATE)) * 20).astype(np.int16)


AGENT = speechlike(4.0, level=2500)


def test_the_bound_is_what_the_readme_states():
    assert BARGE_IN_BOUND_S == 0.2


@pytest.mark.parametrize("offset", [0.3, 1.0, 2.37, 3.5])
def test_agent_stops_within_bound_on_the_simulated_clock(offset):
    caller = np.concatenate([silence(offset), speechlike(1.0)])
    r = simulate(AGENT, caller)
    assert r.barged_in
    assert r.latency_s <= 0.1                       # 3 onset frames + the frame it lands in
    assert r.stopped_at_s < len(AGENT) / SAMPLE_RATE


def test_no_barge_in_when_the_caller_is_silent():
    r = simulate(AGENT, silence(4.5))
    assert not r.barged_in and r.played_s == pytest.approx(len(AGENT) / SAMPLE_RATE, abs=0.02)


def test_own_echo_does_not_stop_the_agent():
    # The mic hears the agent at -12 dB (a laptop speaker next to its mic).
    r = simulate(AGENT, silence(4.5), echo_gain=0.25)
    assert not r.barged_in


def test_caller_over_echo_still_stops_the_agent():
    caller = np.concatenate([silence(1.5), speechlike(1.0, level=3000)])
    r = simulate(AGENT, caller, echo_gain=0.25)
    assert r.barged_in and r.latency_s <= 0.1


def test_playback_stop_is_immediate():
    p = Playback(AGENT)
    p.next_frame()
    p.stop()
    assert p.next_frame() is None and not p.playing


def test_agent_stops_within_bound_in_real_time():
    caller = np.concatenate([silence(0.8), speechlike(1.0)])
    r = RealtimeDuplex(AGENT[: int(2.5 * SAMPLE_RATE)], caller).run()
    assert r.barged_in
    assert r.latency_s <= BARGE_IN_BOUND_S, r
