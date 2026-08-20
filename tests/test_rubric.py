"""Gulf dialect rubric: scoring, judge calibration on anchors, and the reply set."""

import pytest

from eval.rubric import ANCHORS, CRITERIA, PENALTIES, calibrate, fit_line, score, template_replies


def test_criteria_and_penalties_follow_the_adapted_rubric():
    assert len(CRITERIA) == 5
    assert {"hallucination", "ambiguous_framing", "register_flattening"} <= set(PENALTIES)


def test_score_is_criteria_met_minus_penalties_over_the_maximum():
    verdict = {"criteria": {c: True for c in CRITERIA}, "penalties": []}
    assert score(verdict) == 1.0
    verdict = {"criteria": {c: i < 3 for i, c in enumerate(CRITERIA)}, "penalties": ["hallucination"]}
    assert score(verdict) == pytest.approx((3 - PENALTIES["hallucination"]) / 5)


def test_score_never_drops_below_minus_one():
    verdict = {"criteria": {c: False for c in CRITERIA}, "penalties": list(PENALTIES)}
    assert score(verdict) == -1.0


def test_anchor_set_spans_good_and_bad_replies():
    refs = [score(a["reference"]) for a in ANCHORS]
    assert len(ANCHORS) >= 10
    assert max(refs) == 1.0 and min(refs) < 0.2
    assert all(set(a["reference"]["criteria"]) == set(CRITERIA) for a in ANCHORS)


def test_fit_line_recovers_a_linear_map():
    a, b = fit_line([0.0, 0.5, 1.0], [0.1, 0.35, 0.6])
    assert a == pytest.approx(0.5) and b == pytest.approx(0.1)


def test_calibrate_reports_agreement_and_the_map():
    judged = [a["reference"] for a in ANCHORS]          # a perfect judge
    cal = calibrate(judged)
    assert cal["criterion_agreement"] == 1.0
    assert cal["slope"] == pytest.approx(1.0) and cal["intercept"] == pytest.approx(0.0, abs=1e-9)


def test_template_replies_are_arabic_and_cover_every_spoken_action():
    replies = template_replies()
    actions = {r["action"] for r in replies}
    assert {"ask_slot", "offer", "ask_confirm", "confirmed", "no_results", "unavailable", "listen"} <= actions
    assert all(any("؀" <= ch <= "ۿ" for ch in r["text"]) for r in replies)
