"""Word error rate by Levenshtein alignment over whitespace tokens."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.ref_words + other.ref_words,
        )

    @property
    def wer(self) -> float:
        if self.ref_words == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.ref_words


ZERO = ErrorCounts(0, 0, 0, 0)


def align(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    """Return the aligned pairs (ref_token, hyp_token); None marks a gap."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    pairs: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((ref[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hyp[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def count_errors(ref: str, hyp: str) -> ErrorCounts:
    r, h = ref.split(), hyp.split()
    s = dl = ins = 0
    for a, b in align(r, h):
        if a is None:
            ins += 1
        elif b is None:
            dl += 1
        elif a != b:
            s += 1
    return ErrorCounts(s, dl, ins, len(r))


def wer(ref: str, hyp: str) -> float:
    return count_errors(ref, hyp).wer
