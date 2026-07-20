"""Scoring for the evaluation harness.

Digit, date and time error rates follow "The Hidden Cost of Digits": run the
same normalizer over reference and hypothesis, pull out the entity tokens of
one kind in order, and compute the error rate over that token sequence. A phone
number contributes one token per digit, so one wrong digit is one error in ten.
An amount, a date, or a time is one token each; a time window is two.
"""

from __future__ import annotations

from datetime import date

from speech.normalize import normalize

from .wer import ErrorCounts, count_errors

DIGIT_KINDS = {"phone", "number"}
DATE_KINDS = {"date"}
TIME_KINDS = {"time", "time_window"}


def entity_tokens(text: str, kinds: set[str], today: date) -> list[str]:
    out: list[str] = []
    for e in normalize(text, today).entities:
        if e.kind not in kinds:
            continue
        if e.kind == "phone":
            out.extend(e.value)
        elif e.kind == "time_window":
            out.extend(e.value)
        else:
            out.append(str(e.value))
    return out


def entity_errors(ref: str, hyp: str, kinds: set[str], today: date) -> ErrorCounts:
    return count_errors(" ".join(entity_tokens(ref, kinds, today)), " ".join(entity_tokens(hyp, kinds, today)))


def slot_counts(gold: dict, pred: dict) -> tuple[int, int, int]:
    """True positives, false positives, false negatives over (slot, value) pairs."""
    g = {(k, _hashable(v)) for k, v in gold.items() if v is not None}
    p = {(k, _hashable(v)) for k, v in pred.items() if v is not None}
    return len(g & p), len(p - g), len(g - p)


def _hashable(v):
    return tuple(v) if isinstance(v, list) else v


def f1(tp: int, fp: int, fn: int) -> float | None:
    if tp + fp + fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    xs = sorted(values)
    if not xs:
        raise ValueError("no values")
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
