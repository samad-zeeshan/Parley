"""Per-utterance language and dialect identification.

Labels follow the categories we can actually test: en, ar-gulf, ar-msa, and
mixed (Arabic and English in one utterance). NADI labels dialects by country;
Gulf here means Emirati and wider Gulf wording, and nothing finer.

Language comes from script: Arabic letters versus Latin letters per token.
Dialect comes from lexical markers. A Gulf marker is a word MSA would not use
(أبي for "I want", وايد, الحين, بكرة, عقب). An MSA marker is a word a Gulf
speaker would rarely say in a phone call (أريد, هل, سوف, غدا). Arabic with no
marker on either side is labelled MSA, which is the conservative guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import _norm_word

_AR = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[^\W\d_]+")

GULF_MARKERS = {_norm_word(w) for w in [
    "أبي", "ابي", "أبغي", "ابغي", "أبغى", "أبا", "ابا", "نبي", "تبي", "وايد", "الحين", "بكرة", "بكره", "باكر",
    "عقب", "زين", "شلون", "خلاص", "عطني", "عطنا", "هني", "هناك", "ويا", "جي", "ياي", "الياي", "الجاي", "شو",
    "ليش", "مب", "مو", "حق", "يبا", "يبغي", "تمام", "مية", "ميتين", "طعش", "المسا", "الصبح", "ما أبي", "ثنين",
]}
MSA_MARKERS = {_norm_word(w) for w in [
    "أريد", "اريد", "هل", "سوف", "غدا", "غداً", "الآن", "لماذا", "كيف", "يوجد", "توجد", "أود", "اود",
    "نعم", "حسنا", "حسناً", "من فضلك", "لو سمحت", "صباحا", "مساء", "مئة", "مائة", "مئتا", "سنويا",
    "أبحث", "ابحث", "أؤكد", "اؤكد", "رقم هاتفي", "الخيار", "ميزانيتي",
]}


# Place names and fillers carry no language signal for choosing the reply language:
# "Dubai Marina" inside an Arabic sentence is still an Arabic sentence.
_NEUTRAL_EN = {"ok", "okay", "dubai", "marina", "abu", "dhabi", "jlt", "jvc", "downtown", "business", "bay",
               "al", "barsha", "reem", "island", "khalifa", "city", "raha", "beach", "saadiyat", "jumeirah",
               "lake", "towers", "village", "circle", "aed"}


@dataclass(frozen=True)
class LangID:
    label: str          # en | ar-gulf | ar-msa | mixed
    ar_tokens: int
    en_tokens: int
    gulf_score: int
    msa_score: int
    en_content: int = 0  # English words that are not place names or fillers


def identify(text: str) -> LangID:
    words = _WORD.findall(text)
    ar = [w for w in words if _AR.search(w)]
    en = [w for w in words if _LATIN.search(w)]
    norm = [_norm_word(w) for w in ar]
    bigrams = [f"{a} {b}" for a, b in zip(norm, norm[1:])]
    gulf = sum(1 for w in norm + bigrams if w in GULF_MARKERS) + sum(1 for w in norm if w.endswith("طعش"))
    msa = sum(1 for w in norm + bigrams if w in MSA_MARKERS)
    en_content = sum(1 for w in en if w.lower() not in _NEUTRAL_EN)
    if ar and en_content:
        label = "mixed"
    elif ar:
        label = "ar-gulf" if gulf > msa else "ar-msa"
    else:
        label = "en"
    return LangID(label, len(ar), len(en), gulf, msa, en_content)


def reply_language(lid: LangID) -> str:
    """Reply in the caller's language. For a mixed utterance, the matrix language:
    whichever script carried more words, Arabic on a tie."""
    if lid.label == "en":
        return "en"
    if lid.label == "mixed":
        return "en" if lid.en_content > lid.ar_tokens else "ar"
    return "ar"
