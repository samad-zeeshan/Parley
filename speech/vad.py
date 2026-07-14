"""Frame-energy voice activity detection with an adaptive noise floor.

Deliberately simple: 20 ms frames, a noise floor that follows the quietest
recent frames, onset after a few loud frames in a row, end after a hangover of
quiet frames. It is enough for endpointing and barge-in on clean synthetic
audio; a neural VAD (Silero) would be the upgrade for real phone audio.
"""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE

FRAME_MS = 20
FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples


def frame_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) if len(frame) else 0.0
    return 20.0 * np.log10(rms + 1.0)


class EnergyVAD:
    def __init__(self, margin_db: float = 15.0, min_db: float = 40.0, onset_frames: int = 3,
                 hangover_frames: int = 25):
        self.margin_db = margin_db
        self.min_db = min_db
        self.onset_frames = onset_frames
        self.hangover_frames = hangover_frames
        self.noise_db = min_db - margin_db
        self.in_speech = False
        self._loud = 0
        self._quiet = 0

    @property
    def threshold_db(self) -> float:
        return max(self.min_db, self.noise_db + self.margin_db)

    def is_voiced(self, frame: np.ndarray, extra_db: float = 0.0) -> bool:
        return frame_db(frame) > self.threshold_db + extra_db

    def push(self, frame: np.ndarray, extra_db: float = 0.0) -> str | None:
        """Feed one frame. Returns "start", "end" or None.

        extra_db raises the bar for this frame; the duplex loop uses it while the agent
        is talking so its own echo does not count as the caller.
        """
        db = frame_db(frame)
        voiced = db > self.threshold_db + extra_db
        if not voiced and not self.in_speech:
            # Follow the floor down fast and up slowly.
            rate = 0.3 if db < self.noise_db else 0.02
            self.noise_db += rate * (db - self.noise_db)
        if not self.in_speech:
            self._loud = self._loud + 1 if voiced else 0
            if self._loud >= self.onset_frames:
                self.in_speech, self._quiet = True, 0
                return "start"
            return None
        self._quiet = 0 if voiced else self._quiet + 1
        if self._quiet >= self.hangover_frames:
            self.in_speech, self._loud = False, 0
            return "end"
        return None

    def reset(self) -> None:
        self.in_speech, self._loud, self._quiet = False, 0, 0
