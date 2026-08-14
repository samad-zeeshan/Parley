"""Acoustic stress arms for TRACE: noise at a stated SNR, reverberation at a stated RT60, a competing talker."""

import numpy as np
import pytest

from eval.acoustics import ARMS, add_noise, reverb, rir, snr_db, stress
from speech.audio import SAMPLE_RATE

RNG = np.random.default_rng(9)


def speech(seconds=1.5):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return (np.sin(2 * np.pi * 220 * t) * env * 6000).astype(np.int16)


def test_arm_list_matches_the_protocol():
    kinds = {a.split("@")[0] for a in ARMS}
    assert kinds == {"clean", "white", "babble", "reverb", "competing"}
    assert sum(a.startswith("white") for a in ARMS) == 3
    assert sum(a.startswith("babble") for a in ARMS) == 3
    assert sum(a.startswith("reverb") for a in ARMS) == 2


@pytest.mark.parametrize("target", [20, 10, 5])
def test_white_noise_lands_on_the_target_snr(target):
    clean = speech()
    noisy = add_noise(clean, RNG.standard_normal(len(clean)), target)
    assert snr_db(clean, noisy.astype(np.float64) - clean) == pytest.approx(target, abs=0.5)


@pytest.mark.parametrize("rt60", [0.3, 0.8])
def test_room_impulse_decays_60_db_in_rt60(rt60):
    h = rir(rt60, np.random.default_rng(1))
    energy = np.cumsum((h ** 2)[::-1])[::-1]
    db = 10 * np.log10(energy / energy[0] + 1e-12)
    t60 = np.argmax(db <= -60) / SAMPLE_RATE
    assert t60 == pytest.approx(rt60, rel=0.2)


def test_reverb_keeps_length_and_level_bounded():
    x = speech()
    y = reverb(x, 0.8, np.random.default_rng(2))
    assert len(y) == len(x) and y.dtype == np.int16
    assert np.abs(y).max() <= 32767


def test_stress_is_deterministic_per_key():
    x = speech()
    a = stress(x, "white@10", key="call/1/0")
    b = stress(x, "white@10", key="call/1/0")
    c = stress(x, "white@10", key="call/2/0")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_clean_arm_is_the_identity():
    x = speech()
    assert np.array_equal(stress(x, "clean", key="k"), x)


def test_babble_and_competing_use_the_given_talkers():
    x = speech()
    talkers = [speech(2.0) // 2, speech(1.0) // 3]
    y = stress(x, "babble@10", key="k", talkers=talkers)
    z = stress(x, "competing", key="k", talkers=talkers)
    assert len(y) == len(x) == len(z)
    assert not np.array_equal(y, x) and not np.array_equal(z, x)
