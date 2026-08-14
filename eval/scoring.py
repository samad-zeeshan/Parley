"""Scoring for the evaluation harness: entity error rates, per-language span errors, slot F1, percentiles.

Entity error rates follow arXiv 2609.21084: one token per phone digit, one per amount, date or time.
"""

from __future__ import annotations

from datetime import date

import re

from speech.normalize import normalize
from speech.textnorm import normalize_orthography

from .wer import ZERO, ErrorCounts, align_indices, count_errors

_LATIN = re.compile(r"[a-z]")

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


def span_errors(segments: list[tuple[str, str]], hyp: str) -> dict:
    """Errors on the English and on the Arabic words of one reference line, from one alignment.

    Scored apart as in arXiv 2605.19069, because a single WER hides that the English inside an
    Arabic sentence is what gets lost. English written in Arabic script counts as an error.
    """
    ref, tags = [], []
    for lang, text in segments:
        words = normalize_orthography(text).split()
        ref += words
        tags += [lang] * len(words)
    h = normalize_orthography(hyp).split()
    counts = {"en": [0, 0, 0], "ar": [0, 0, 0]}  # substitutions, deletions, insertions
    latin, last = 0, None
    for i, j in align_indices(ref, h):
        if i is None:
            # An inserted word belongs to the span it follows, or to the first span if nothing came before.
            counts[last or (tags[0] if tags else "ar")][2] += 1
            continue
        last = tags[i]
        if j is None:
            counts[last][1] += 1
        else:
            counts[last][0] += ref[i] != h[j]
            latin += last == "en" and bool(_LATIN.search(h[j]))
    n = {k: tags.count(k) for k in counts}
    out = {k: ErrorCounts(*v, n[k]) if n[k] or any(v) else ZERO for k, v in counts.items()}
    out["en_latin_kept"] = latin
    out["latin_kept"] = latin / n["en"] if n["en"] else None
    return out


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
