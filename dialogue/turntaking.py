"""Turn taking as an explicit finite-state machine, driven by 20 ms mic frames.

The states and legal moves are in TRANSITIONS and in docs/diagrams/turn-taking.html.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from speech.vad import DEFAULT_COUPLING, EnergyVAD, echo_extra_db


class State(Enum):
    LISTENING = "listening"
    CALLER_SPEAKING = "caller_speaking"
    ENDPOINTING = "endpointing"
    THINKING = "thinking"
    AGENT_SPEAKING = "agent_speaking"


class BadTransition(Exception):
    pass


class TurnTaking:
    # Kept separate from the semantics of the call, after arXiv 2609.03321: this machine only knows
    # who holds the floor. What was said is the dialogue policy's business.
    TRANSITIONS = {
        State.LISTENING: {State.CALLER_SPEAKING},
        State.CALLER_SPEAKING: {State.ENDPOINTING},
        State.ENDPOINTING: {State.CALLER_SPEAKING, State.THINKING},
        # A caller who starts again while the reply is being computed keeps the floor.
        State.THINKING: {State.AGENT_SPEAKING, State.CALLER_SPEAKING},
        State.AGENT_SPEAKING: {State.LISTENING, State.CALLER_SPEAKING},
    }

    def __init__(self, vad: EnergyVAD | None = None, state: State = State.LISTENING,
                 coupling: float = DEFAULT_COUPLING, on_barge_in=None):
        self.vad = vad or EnergyVAD()
        self.state = state
        self.coupling = coupling
        self.on_barge_in = on_barge_in
        self.frame = 0
        self.history: list[tuple[int, State]] = []
        self.barge_ins = 0
        self.turns_closed = 0

    def _go(self, new: State) -> None:
        if new not in self.TRANSITIONS[self.state]:
            raise BadTransition(f"{self.state.value} -> {new.value}")
        self.state = new
        self.history.append((self.frame, new))

    def mic(self, frame: np.ndarray, out: np.ndarray | None = None) -> State:
        """Feed one mic frame. `out` is the agent frame going to the speaker at the same time, if any."""
        self.frame += 1
        extra = echo_extra_db(self.vad, out, self.coupling) if self.state is State.AGENT_SPEAKING else 0.0
        ev = self.vad.push(frame, extra_db=extra)
        if ev == "start":
            if self.state is State.AGENT_SPEAKING:
                self.barge_ins += 1
                self._go(State.CALLER_SPEAKING)
                if self.on_barge_in:
                    self.on_barge_in()
            elif self.state in (State.LISTENING, State.THINKING):
                self._go(State.CALLER_SPEAKING)
        elif ev == "end" and self.state in (State.CALLER_SPEAKING, State.ENDPOINTING):
            if self.state is State.CALLER_SPEAKING:
                self._go(State.ENDPOINTING)
            self._go(State.THINKING)
            self.turns_closed += 1
        elif self.state is State.CALLER_SPEAKING and self.vad._quiet > 0:
            # The first quiet frame opens the hangover. The final ASR decode can start here, so up to
            # the whole hangover of decode time is hidden before the turn officially closes.
            self._go(State.ENDPOINTING)
        elif self.state is State.ENDPOINTING and self.vad._quiet == 0:
            self._go(State.CALLER_SPEAKING)
        return self.state

    def reply_ready(self) -> None:
        if self.state is not State.THINKING:
            raise BadTransition(f"reply ready in {self.state.value}")
        self._go(State.AGENT_SPEAKING)

    def playback_done(self) -> None:
        if self.state is not State.AGENT_SPEAKING:
            raise BadTransition(f"playback done in {self.state.value}")
        self._go(State.LISTENING)
