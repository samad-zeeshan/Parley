"""Arabic to Latin script for the demo's second transcript line.

Uses the Arabizi conventions Gulf speakers type in chat (3 for ع, 7 for ح).
Letter by letter, no vowels restored, so "دبي مارينا" becomes "dby maryna".
It is a reading aid, not a transcription standard.
"""

from __future__ import annotations

import re

_MAP = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "aa", "ٱ": "a", "ء": "'", "ؤ": "'", "ئ": "'",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "7", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "3", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ي": "y", "ى": "a", "ة": "a", "پ": "p",
    "چ": "ch", "گ": "g", "ڤ": "v", "،": ",", "؟": "?", "؛": ";",
    **{chr(0x0660 + i): str(i) for i in range(10)},
}
_DROP = re.compile(r"[ً-ْٰـ]")


def to_latin(text: str) -> str:
    return "".join(_MAP.get(ch, ch) for ch in _DROP.sub("", text))
