"""Full-duplex playback: the mic is heard while the agent talks, and the agent stops when the caller starts.

Barge-in decisions come from the turn-taking machine in dialogue/turntaking.py.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from dialogue.turntaking import State, TurnTaking

from .audio import SAMPLE_RATE
from .vad import DEFAULT_COUPLING, FRAME, FRAME_MS, EnergyVAD, echo_extra_db, frame_db  # noqa: F401

# Onset needs 3 voiced frames (60 ms) and the stop lands on the next frame (20 ms): 80 ms on an exact
# clock. The 0.2 s bound is what the real-time test enforces, with room for thread scheduling.
BARGE_IN_BOUND_S = 0.2


class Playback:
    """Agent speech played out one frame at a time. stop() takes effect on the next frame."""

    def __init__(self, pcm: np.ndarray):
        self.pcm = pcm
        self.pos = 0
        self._stopped = threading.Event()

    @property
    def playing(self) -> bool:
        return not self._stopped.is_set() and self.pos < len(self.pcm)

    def next_frame(self) -> np.ndarray | None:
        if not self.playing:
            return None
        f = self.pcm[self.pos:self.pos + FRAME]
        self.pos += FRAME
        if len(f) < FRAME:
            f = np.pad(f, (0, FRAME - len(f)))
        return f

    def stop(self) -> None:
        self._stopped.set()

    @property
    def played_s(self) -> float:
        return min(self.pos, len(self.pcm)) / SAMPLE_RATE


def _frame(pcm: np.ndarray, k: int) -> np.ndarray:
    f = pcm[k * FRAME:(k + 1) * FRAME]
    return np.pad(f, (0, FRAME - len(f))) if len(f) < FRAME else f


def caller_onset_s(caller: np.ndarray, min_db: float = 40.0) -> float | None:
    """Ground truth: first frame of the caller track (alone, no echo) above the VAD floor."""
    for k in range(len(caller) // FRAME):
        if frame_db(_frame(caller, k)) > min_db:
            return k * FRAME_MS / 1000
    return None


@dataclass
class BargeInResult:
    barged_in: bool
    onset_s: float | None      # when the caller started, from the caller track alone
    stopped_at_s: float | None  # when the agent's audio stopped
    played_s: float
    false_stop: bool = False   # stopped before the caller said anything

    @property
    def latency_s(self) -> float | None:
        if self.stopped_at_s is None or self.onset_s is None:
            return None
        return self.stopped_at_s - self.onset_s


def simulate(agent: np.ndarray, caller: np.ndarray, echo_gain: float | None = None,
             coupling: float = DEFAULT_COUPLING, vad: EnergyVAD | None = None) -> BargeInResult:
    """Step both streams on one exact 20 ms clock."""
    vad = vad or EnergyVAD()
    play = Playback(agent)
    fsm = TurnTaking(vad, State.AGENT_SPEAKING, coupling, on_barge_in=play.stop)
    onset = caller_onset_s(caller, vad.min_db)
    stopped_at = None
    n = max(len(agent), len(caller)) // FRAME + 1
    for k in range(n):
        out = play.next_frame()
        if out is None and fsm.state is State.AGENT_SPEAKING:
            fsm.playback_done()
        mic = _frame(caller, k).astype(np.float64)
        if out is not None and echo_gain:
            mic = mic + echo_gain * out
        mic = np.clip(mic, -32768, 32767).astype(np.int16)
        fsm.mic(mic, out)
        if fsm.barge_ins:
            stopped_at = (k + 1) * FRAME_MS / 1000
            break
    false_stop = stopped_at is not None and (onset is None or stopped_at < onset)
    return BargeInResult(stopped_at is not None, onset, stopped_at, play.played_s, false_stop)


class RealtimeDuplex:
    """A real-time player thread and a real-time mic thread, 20 ms pacing each.

    Measures wall-clock time from the moment the caller's first voiced frame is
    fed to the moment the player skips its next frame.
    """

    def __init__(self, agent: np.ndarray, caller: np.ndarray, echo_gain: float | None = None,
                 coupling: float = DEFAULT_COUPLING):
        self.play = Playback(agent)
        self.caller = caller
        self.echo_gain = echo_gain
        self.coupling = coupling
        self.vad = EnergyVAD()
        self.fsm = TurnTaking(self.vad, State.AGENT_SPEAKING, coupling)
        self._last_out: np.ndarray | None = None
        self._lock = threading.Lock()
        self.onset_wall: float | None = None
        self.stop_wall: float | None = None
        self.detect_wall: float | None = None

    def _player(self, t0: float) -> None:
        k = 0
        while True:
            target = t0 + k * FRAME_MS / 1000
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            was_playing = self.play.playing
            out = self.play.next_frame()
            with self._lock:
                self._last_out = out
            if out is None:
                if was_playing is False and self.stop_wall is None and self.detect_wall is not None:
                    self.stop_wall = time.perf_counter()
                return
            k += 1

    def _mic(self, t0: float) -> None:
        onset_frame = None
        onset = caller_onset_s(self.caller, self.vad.min_db)
        if onset is not None:
            onset_frame = round(onset * 1000 / FRAME_MS)
        n = len(self.caller) // FRAME
        for k in range(n):
            target = t0 + k * FRAME_MS / 1000 + 0.005  # mic runs 5 ms behind the speaker tick
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            if k == onset_frame:
                self.onset_wall = time.perf_counter()
            with self._lock:
                out = self._last_out
            mic = _frame(self.caller, k).astype(np.float64)
            if out is not None and self.echo_gain:
                mic = mic + self.echo_gain * out
            mic = np.clip(mic, -32768, 32767).astype(np.int16)
            if out is None and self.fsm.state is State.AGENT_SPEAKING and not self.play.playing:
                self.fsm.playback_done()
            self.fsm.mic(mic, out)
            if self.fsm.barge_ins and self.play.playing:
                self.detect_wall = time.perf_counter()
                self.play.stop()
                return

    def run(self) -> BargeInResult:
        t0 = time.perf_counter() + 0.05
        threads = [threading.Thread(target=self._player, args=(t0,)), threading.Thread(target=self._mic, args=(t0,))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        barged = self.detect_wall is not None
        onset_s = (self.onset_wall - t0) if self.onset_wall else None
        stop_s = (self.stop_wall - t0) if self.stop_wall else None
        return BargeInResult(barged, onset_s, stop_s, self.play.played_s,
                             barged and (onset_s is None or (stop_s or 0) < onset_s))
