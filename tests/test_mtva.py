"""MTVA protocol pieces: controlled transcription errors and the split-turn caller."""

import pytest

from eval.harness import script_ok, split_turn
from eval.mtva import CONDITIONS, group_lines, inject, realized_wer
from eval.scripts import CALLS

LINES = [t["text"] for c in CALLS.values() for t in c["turns"]]


def test_zero_rate_leaves_the_text_alone():
    assert all(inject(t, 0.0, key=str(i)) == t for i, t in enumerate(LINES))


def test_injection_is_deterministic_per_key():
    t = "أبي شقة غرفتين في البرشاء والميزانية مية وعشرين ألف"
    assert inject(t, 0.2, key="a") == inject(t, 0.2, key="a")
    assert any(inject(t, 0.3, key=str(k)) != inject(t, 0.3, key="a") for k in range(5))


@pytest.mark.parametrize("rate", [0.05, 0.1, 0.2, 0.3])
def test_realized_word_error_rate_is_near_the_nominal_rate(rate):
    got = realized_wer(LINES, rate)
    assert abs(got - rate) < max(0.03, rate * 0.25), got


def test_english_inside_arabic_can_come_back_in_arabic_script():
    """The failure v1 saw: "Sunday" written as Arabic letters. The injector reproduces it."""
    outs = {inject("Sunday عقب الظهر", 0.9, key=str(k)).split()[0] for k in range(40)}
    assert any(not w.isascii() for w in outs)


def test_split_turn_cuts_long_lines_only():
    assert split_turn("yes") == ["yes"]
    parts = split_turn("My number is zero five zero one two three")
    assert len(parts) == 2 and " ".join(parts) == "My number is zero five zero one two three"


def test_script_check():
    assert script_ok("في أي منطقة تبحث؟", "ar")
    assert not script_ok("في أي منطقة تبحث؟ Dubai Marina", "ar")
    assert script_ok("Which area are you looking in?", "en")
    assert not script_ok("Which area? دبي", "en")


def test_conditions_cover_the_protocol():
    assert {"reference", "asr:local", "asr:hosted", "split"} <= set(CONDITIONS)
    assert sum(c.startswith("noise@") for c in CONDITIONS) >= 4


def test_group_lines_joins_fragments_of_one_caller_turn():
    turns = [{"line": 0, "attempt": 0, "fragment": 0}, {"line": 0, "attempt": 0, "fragment": 1},
             {"line": 1, "attempt": 0, "fragment": 0}]
    assert [len(g) for g in group_lines(turns)] == [2, 1]
