"""Code-switch scoring and the local mitigations: English-span WER, the mixed prompt and the pause segmenter."""

import numpy as np
import pytest

from eval.scoring import span_errors
from eval.scripts import CALLS
from speech.asr import MIXED_PROMPT, split_at_pauses
from speech.audio import SAMPLE_RATE
from speech.backends import CONFIGS, BackendUnavailable, load
from speech.textnorm import normalize_orthography

RNG = np.random.default_rng(5)


def test_english_span_errors_are_counted_apart_from_arabic():
    segs = [("en", "Hi,"), ("ar", "أبي أحجز"), ("en", "viewing"), ("ar", "في"), ("en", "Dubai Marina")]
    out = span_errors(segs, "هاي أبي أحجز دي دبي مارينا")
    assert out["en"].ref_words == 4 and out["ar"].ref_words == 3
    assert out["en"].errors == 4          # all four English words lost or transliterated
    assert out["ar"].errors == 1          # "في" dropped


def test_perfect_hypothesis_has_no_span_errors():
    segs = [("en", "Sunday"), ("ar", "عقب الظهر")]
    out = span_errors(segs, "Sunday عقب الظهر.")
    assert out["en"].errors == 0 and out["ar"].errors == 0
    assert out["latin_kept"] == 1.0


def test_transliterated_english_counts_as_an_error_and_as_not_latin():
    out = span_errors([("en", "Sunday"), ("ar", "عقب الظهر")], "سندي عقب الظهر")
    assert out["en"].substitutions == 1
    assert out["latin_kept"] == 0.0


def test_insertions_go_to_the_span_they_follow():
    out = span_errors([("ar", "تمام"), ("en", "confirm the booking")], "تمام confirm the booking now")
    assert out["en"].insertions == 1 and out["ar"].insertions == 0


def test_mixed_prompt_shares_no_three_word_run_with_any_scripted_line():
    """The prompt biases decoding toward keeping English in Latin script. It must not leak test lines."""
    def grams(text):
        w = normalize_orthography(text).split()
        return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}

    prompt = grams(MIXED_PROMPT)
    for cid, call in CALLS.items():
        for t in call["turns"]:
            assert not prompt & grams(t["text"]), (cid, t["text"])


def _burst(seconds):
    return (RNG.standard_normal(int(seconds * SAMPLE_RATE)) * 3000).astype(np.int16)


def _gap(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.int16)


def test_split_at_pauses_finds_the_segments():
    pcm = np.concatenate([_gap(0.2), _burst(0.6), _gap(0.25), _burst(0.4), _gap(0.05), _burst(0.3), _gap(0.2)])
    chunks = split_at_pauses(pcm, min_gap_s=0.15)
    assert len(chunks) == 2                       # the 50 ms gap is inside a word, not a switch point
    assert all(len(c) > 0.3 * SAMPLE_RATE for c in chunks)


def test_split_at_pauses_on_silence_returns_nothing():
    assert split_at_pauses(_gap(1.0)) == []


def test_speech_configs_include_the_local_variants_and_hosted():
    assert {"local", "local-v1", "local-2pass", "local-base", "local-tiny", "hosted"} <= set(CONFIGS)
    assert CONFIGS["local"].asr_model == "small"
    assert CONFIGS["local-v1"].prompt != CONFIGS["local"].prompt


def test_hosted_is_not_available_offline():
    with pytest.raises(BackendUnavailable):
        load("hosted")
