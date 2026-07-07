"""Orthographic normalization used before scoring ASR output.

This is not the number normalizer (see speech/normalize.py). It only removes
differences that no listener would call an error: case, punctuation, Arabic
diacritics, tatweel, and the usual alef, ta marbuta and alef maqsura variants.
"""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # harakat, dagger alef, tatweel
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")

_ARABIC_MAP = str.maketrans({
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0622": "\u0627",  # آ -> ا
    "\u0671": "\u0627",  # ٱ -> ا
    "\u0629": "\u0647",  # ة -> ه
    "\u0649": "\u064A",  # ى -> ي
    "\u060C": " ",       # Arabic comma
    "\u061F": " ",       # Arabic question mark
    "\u061B": " ",       # Arabic semicolon
})


def normalize_orthography(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _DIACRITICS.sub("", text)
    text = text.translate(_ARABIC_MAP)
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
