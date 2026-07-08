"""Number, date, time and phone normalization for English, Gulf Arabic and MSA.

Runs on every ASR transcript before the intent layer, and on reference and
hypothesis text before scoring. The output is the transcript with each entity
replaced by one canonical token (120000, 0501234567, 16:00, 16:00-18:00,
2026-10-01) plus the list of entities found.

The scanner tries, at each token: phone, date, time, time window, number. The
order matters: "الساعة خمسة ونص" is a time, not the number 5.5, and
"zero five zero ..." is a phone, not ten numbers.

Viewing hours are 10:00 to 19:30, so a bare hour from 1 to 7 with no am or pm
marker is read as afternoon. That is a domain rule, not a language rule.
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Orthography and tokens
# ---------------------------------------------------------------------------

_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_TOKEN = re.compile(r"\d+(?:[:./]\d+)*|[^\W\d_]+")


def _norm_word(w: str) -> str:
    return _DIACRITICS.sub("", w).translate(_AR_MAP).lower()


def _prepare(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_DIGITS)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)                 # 120,000
    text = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", text, flags=re.I)  # p.m. -> pm
    return text


def tokenize(text: str) -> list[str]:
    toks = [_norm_word(t) for t in _TOKEN.findall(_prepare(text))]
    out: list[str] = []
    for t in toks:
        # Arabic attaches "and" to the next word: وعشرين, وستة, ونص.
        if len(t) > 2 and t.startswith("و") and _is_ar_numberish(t[1:]):
            out.extend(["و", t[1:]])
        else:
            out.append(t)
    return out


def _lex(d: dict) -> dict:
    return {_norm_word(k): v for k, v in d.items()}


# ---------------------------------------------------------------------------
# Lexicons (keys are normalized with _norm_word at import)
# ---------------------------------------------------------------------------

EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
           "eighty": 80, "ninety": 90}
EN_SCALES = {"thousand": 1_000, "k": 1_000, "grand": 1_000, "million": 1_000_000, "m": 1_000_000}

AR_UNITS = _lex({
    "صفر": 0, "واحد": 1, "واحدة": 1, "وحدة": 1, "اثنين": 2, "اثنان": 2, "ثنين": 2, "اثنتين": 2, "اثنتان": 2,
    "ثلاث": 3, "ثلاثة": 3, "تلات": 3, "تلاتة": 3, "اربع": 4, "اربعة": 4, "خمس": 5, "خمسة": 5,
    "ست": 6, "ستة": 6, "سته": 6, "سبع": 7, "سبعة": 7, "ثمان": 8, "ثماني": 8, "ثمانية": 8, "ثمن": 8,
    "ثمنية": 8, "تسع": 9, "تسعة": 9, "عشر": 10, "عشرة": 10,
    "احدعش": 11, "حدعش": 11, "احدى عشر": 11, "اثنعش": 12, "ثنعش": 12, "اثنا عشر": 12, "اثني عشر": 12,
    "ثلطعش": 13, "ثلاثطعش": 13, "اربعطعش": 14, "خمسطعش": 15, "ستطعش": 16, "سبعطعش": 17,
    "ثمنطعش": 18, "ثمانطعش": 18, "تسعطعش": 19, "اثنا": 2, "اثني": 2, "احدى": 1,
})
_AR_TEEN_TEN = set(_lex({"عشر": 1, "عشرة": 1}))
AR_TENS = _lex({
    "عشرين": 20, "عشرون": 20, "ثلاثين": 30, "ثلاثون": 30, "اربعين": 40, "اربعون": 40, "خمسين": 50,
    "خمسون": 50, "ستين": 60, "ستون": 60, "سبعين": 70, "سبعون": 70, "ثمانين": 80, "ثمانون": 80,
    "تسعين": 90, "تسعون": 90,
})
AR_HUNDREDS = _lex({
    "مية": 100, "مئة": 100, "مائة": 100, "ميه": 100, "ميتين": 200, "مئتين": 200, "مئتا": 200,
    "مائتين": 200, "مائتا": 200, "مئتان": 200, "ثلاثمية": 300, "ثلاثمئة": 300, "ثلاثمائة": 300,
    "اربعمية": 400, "اربعمئة": 400, "اربعمائة": 400, "خمسمية": 500, "خمسمئة": 500, "خمسمائة": 500,
    "ستمية": 600, "ستمئة": 600, "ستمائة": 600, "سبعمية": 700, "سبعمئة": 700, "سبعمائة": 700,
    "ثمنمية": 800, "ثمانمئة": 800, "ثمانمائة": 800, "تسعمية": 900, "تسعمئة": 900, "تسعمائة": 900,
})
# (value, is_dual): a dual scale word carries its own "two".
AR_SCALES = _lex({
    "الف": (1_000, False), "الاف": (1_000, False), "الفين": (2_000, True), "الفان": (2_000, True),
    "مليون": (1_000_000, False), "ملايين": (1_000_000, False), "مليونين": (2_000_000, True),
})
AR_HALF = _lex({"نص": 0.5, "نصف": 0.5})
AR_QUARTER = _lex({"ربع": 0.25})

AR_ORDINAL_HOURS = _lex({
    "الواحدة": 1, "الثانية": 2, "الثالثة": 3, "الرابعة": 4, "الخامسة": 5, "السادسة": 6, "السابعة": 7,
    "الثامنة": 8, "التاسعة": 9, "العاشرة": 10, "الحادية عشرة": 11, "الثانية عشرة": 12,
})

EN_WEEKDAYS = {d.lower(): i for i, d in enumerate(calendar.day_name)}
AR_WEEKDAYS = _lex({"الاثنين": 0, "الثلاثاء": 1, "الاربعاء": 2, "الخميس": 3, "الجمعة": 4, "السبت": 5,
                    "الاحد": 6})
EN_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
EN_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
AR_MONTHS = _lex({"يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5, "يونيو": 6, "يوليو": 7,
                  "اغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12})
EN_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
               "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
               "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
               "nineteenth": 19, "twentieth": 20, "thirtieth": 30}

NEXT_WORDS = {"next", "coming"} | set(_lex({"الجاي": 1, "القادم": 1, "الياي": 1, "الجاية": 1}))
TODAY_WORDS = {"today"} | set(_lex({"اليوم": 1}))
TOMORROW_WORDS = {"tomorrow"} | set(_lex({"بكرة": 1, "بكره": 1, "باكر": 1, "غدا": 1, "الغد": 1, "بكرا": 1}))
DAY_WORDS = {"on"} | set(_lex({"يوم": 1}))

# Periods of day: (am/pm hint for a spoken hour, default window when said alone)
PERIODS = {
    ("morning",): ("am", ("09:00", "12:00")),
    ("afternoon",): ("pm", ("12:00", "17:00")),
    ("evening",): ("pm", ("17:00", "21:00")),
    ("night",): ("pm", ("17:00", "21:00")),
    ("tonight",): ("pm", ("17:00", "21:00")),
    ("am",): ("am", None),
    ("pm",): ("pm", None),
}
_AR_PERIODS = {
    ("الصبح",): ("am", ("09:00", "12:00")), ("الصباح",): ("am", ("09:00", "12:00")),
    ("صباحا",): ("am", ("09:00", "12:00")), ("صباح",): ("am", ("09:00", "12:00")),
    ("الظهر",): ("noon", ("12:00", "15:00")), ("ظهرا",): ("noon", ("12:00", "15:00")),
    ("عقب", "الظهر"): ("pm", ("12:00", "21:00")), ("بعد", "الظهر"): ("pm", ("12:00", "21:00")),
    ("العصر",): ("pm", ("15:00", "18:00")), ("عصرا",): ("pm", ("15:00", "18:00")),
    ("المغرب",): ("pm", ("17:00", "21:00")),
    ("المسا",): ("pm", ("17:00", "21:00")), ("المساء",): ("pm", ("17:00", "21:00")),
    ("مساء",): ("pm", ("17:00", "21:00")), ("مساءا",): ("pm", ("17:00", "21:00")),
    ("بالليل",): ("pm", ("17:00", "21:00")), ("الليل",): ("pm", ("17:00", "21:00")),
    ("ليلا",): ("pm", ("17:00", "21:00")),
}
PERIODS.update({tuple(_norm_word(w) for w in k): v for k, v in _AR_PERIODS.items()})
_PERIOD_FILLERS = {"in", "the", "at", "this"} | set(_lex({"في": 1}))

AR_AND = "و"
AR_HOUR_WORDS = set(_lex({"الساعة": 1, "ساعة": 1, "الساعه": 1}))
AR_BETWEEN = set(_lex({"بين": 1, "مابين": 1}))
AR_AFTER = set(_lex({"بعد": 1, "عقب": 1}))
AR_EXCEPT = set(_lex({"الا": 1}))

_PRONOUN_ONE_BEFORE = {"the", "this", "that", "which", "first", "second", "last", "no", "any", "every",
                       "some", "each", "other", "another", "someone", "anyone"}
_PRONOUN_ONE_AFTER = {"is", "was", "of", "please"}


def _is_ar_numberish(w: str) -> bool:
    return (w in AR_UNITS or w in AR_TENS or w in AR_HUNDREDS or w in AR_SCALES or w in AR_HALF
            or w in AR_QUARTER)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    kind: str       # number | phone | date | time | time_window
    value: object   # int | str | (start, end)
    start: int      # token span in Normalized.tokens
    end: int

    @property
    def canonical(self) -> str:
        if self.kind == "time_window":
            return f"{self.value[0]}-{self.value[1]}"
        return str(self.value)


@dataclass
class Normalized:
    text: str
    tokens: list[str]
    entities: list[Entity] = field(default_factory=list)

    def of(self, kind: str) -> list[Entity]:
        return [e for e in self.entities if e.kind == kind]


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

_PHONE = re.compile(r"^0(5\d{8}|[234679]\d{7})$")


def valid_phone(digits: str) -> str | None:
    if digits.startswith("00971"):
        digits = "0" + digits[5:]
    elif digits.startswith("971"):
        digits = "0" + digits[3:]
    return digits if _PHONE.match(digits) else None


class _Scanner:
    def __init__(self, tokens: list[str], today: date):
        self.t = tokens
        self.today = today

    def at(self, i: int) -> str:
        return self.t[i] if 0 <= i < len(self.t) else ""

    # ---- digits and numbers ------------------------------------------------

    def digit_word(self, i: int) -> str | None:
        w = self.at(i)
        if w in ("oh", "o"):
            return "0"
        if w in EN_UNITS and EN_UNITS[w] < 10:
            return str(EN_UNITS[w])
        if w in AR_UNITS and AR_UNITS[w] < 10:
            return str(AR_UNITS[w])
        return None

    def phone(self, i: int) -> tuple[str, int] | None:
        digits: list[tuple[str, int]] = []  # cumulative (digits, next index)
        acc, j = "", i
        while j < len(self.t) and len(acc) <= 13:
            w = self.at(j)
            if w.isdigit():
                acc += w
                j += 1
            elif w in ("double", "triple") and self.digit_word(j + 1):
                acc += self.digit_word(j + 1) * (2 if w == "double" else 3)
                j += 2
            elif self.digit_word(j) is not None:
                acc += self.digit_word(j)
                j += 1
            else:
                break
            digits.append((acc, j))
        for acc, j in reversed(digits):
            if j - i >= 2 or len(acc) >= 9:
                ok = valid_phone(acc)
                if ok:
                    return ok, j
        return None

    def number(self, i: int) -> tuple[float, int] | None:
        w = self.at(i)
        if re.fullmatch(r"\d+(\.\d+)?", w):
            val, j = float(w), i + 1
            nxt = self.at(j)
            if nxt in EN_SCALES:
                return val * EN_SCALES[nxt], j + 1
            if nxt in AR_SCALES:
                return val * AR_SCALES[nxt][0], j + 1
            return val, j
        if w in EN_UNITS or w in EN_TENS or (w == "a" and self.at(i + 1) in ("hundred", "thousand", "million")):
            return self._english_number(i)
        if _is_ar_numberish(w) and w not in AR_HALF and w not in AR_QUARTER:
            return self._arabic_number(i)
        return None

    def _english_number(self, i: int) -> tuple[float, int] | None:
        total, cur, j, seen = 0.0, 0.0, i, False
        while j < len(self.t):
            w = self.at(j)
            if w == "a" and self.at(j + 1) in ("hundred", "thousand", "million"):
                cur, j = max(cur, 1), j + 1
            elif w in EN_UNITS:
                cur, j, seen = cur + EN_UNITS[w], j + 1, True
            elif w in EN_TENS:
                cur, j, seen = cur + EN_TENS[w], j + 1, True
            elif w == "hundred" and seen:
                cur, j = max(cur, 1) * 100, j + 1
            elif w in EN_SCALES and seen and w not in ("m",):
                total, cur, j = total + max(cur, 1) * EN_SCALES[w], 0.0, j + 1
            elif w == "and" and self.at(j + 1) == "a" and self.at(j + 2) == "half" and seen:
                cur, j = cur + 0.5, j + 3
            elif w == "and" and seen and (self.at(j + 1) in EN_UNITS or self.at(j + 1) in EN_TENS):
                j += 1
            else:
                break
        if not seen:
            return None
        return total + cur, j

    def _arabic_number(self, i: int) -> tuple[float, int] | None:
        total, cur, j, seen, last_scale = 0.0, 0.0, i, False, 1
        while j < len(self.t):
            w = self.at(j)
            if w == AR_AND and seen and (_is_ar_numberish(self.at(j + 1)) or self.at(j + 1).isdigit()):
                j += 1
                continue
            if w in AR_UNITS and AR_UNITS[w] < 10 and self.at(j + 1) in _AR_TEEN_TEN:
                cur += AR_UNITS[w] + 10  # MSA teens: ثلاثة عشر
                j += 1
            elif w in AR_UNITS:
                cur += AR_UNITS[w]
            elif w.isdigit() and seen and cur % 100 == 0 and int(w) < 100 and (
                    self.at(j + 1) in AR_SCALES or self.at(j - 1) == AR_AND):
                cur += int(w)  # ASR mixes forms: "مية 20 ألف"
            elif w in AR_TENS:
                cur += AR_TENS[w]
            elif w in AR_HUNDREDS:
                v = AR_HUNDREDS[w]
                cur = cur * v if (v == 100 and 0 < cur < 10) else cur + v
            elif w in AR_SCALES:
                v, dual = AR_SCALES[w]
                if dual:
                    total += v * (cur if cur else 1)
                    last_scale = v // 2
                else:
                    total += (cur if cur else 1) * v
                    last_scale = v
                cur = 0.0
            elif w in AR_HALF and seen and self.at(j - 1) == AR_AND:
                total += 0.5 * last_scale
            elif w in AR_QUARTER and seen and self.at(j - 1) == AR_AND:
                total += 0.25 * last_scale
            else:
                break
            seen, j = True, j + 1
        if not seen:
            return None
        return total + cur, j

    # ---- dates ------------------------------------------------------------

    def _roll(self, d: date) -> date:
        return d if d >= self.today else date(d.year + 1, d.month, d.day)

    def _day_of_month(self, i: int) -> tuple[int, int] | None:
        w = self.at(i)
        if w.isdigit() and 1 <= int(w) <= 31:
            j = i + 1
            if self.at(j) in ("st", "nd", "rd", "th"):
                j += 1
            return int(w), j
        if w in EN_ORDINALS:
            return EN_ORDINALS[w], i + 1
        if w in ("twenty", "thirty") and self.at(i + 1) in EN_ORDINALS:
            return EN_TENS[w] + EN_ORDINALS[self.at(i + 1)], i + 2
        n = self.number(i)
        if n and float(n[0]).is_integer() and 1 <= n[0] <= 31 and not re.fullmatch(r"\d+(\.\d+)?", w):
            return int(n[0]), n[1]
        return None

    def _month(self, i: int) -> int | None:
        return EN_MONTHS.get(self.at(i)) or AR_MONTHS.get(self.at(i))

    def date(self, i: int) -> tuple[str, int] | None:
        j = i
        if self.at(j) in DAY_WORDS:
            j += 1  # "on Thursday", "يوم الخميس"
        w = self.at(j)
        if w == "the" and self.at(j + 1) == "day" and self.at(j + 2) == "after" and self.at(j + 3) == "tomorrow":
            return (self.today + timedelta(days=2)).isoformat(), j + 4
        if w == "day" and self.at(j + 1) == "after" and self.at(j + 2) == "tomorrow":
            return (self.today + timedelta(days=2)).isoformat(), j + 3
        if w in AR_AFTER and self.at(j + 1) in TOMORROW_WORDS:
            return (self.today + timedelta(days=2)).isoformat(), j + 2
        if w in TOMORROW_WORDS:
            return (self.today + timedelta(days=1)).isoformat(), j + 1
        if w in TODAY_WORDS:
            return self.today.isoformat(), j + 1
        nxt = False
        if w in ("next", "coming", "this"):
            nxt, j, w = w != "this", j + 1, self.at(j + 1)
        wd = EN_WEEKDAYS.get(w, AR_WEEKDAYS.get(w))
        if wd is not None:
            j += 1
            if self.at(j) in NEXT_WORDS:
                nxt, j = True, j + 1
            ahead = (wd - self.today.weekday()) % 7
            if nxt and ahead == 0:
                ahead = 7
            return (self.today + timedelta(days=ahead)).isoformat(), j
        # d/m, as written in the UAE
        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})", w)
        if m:
            d, mo = int(m.group(1)), int(m.group(2))
            try:
                return self._roll(date(self.today.year, mo, d)).isoformat(), j + 1
            except ValueError:
                return None
        # "the 3rd of October", "3 October", "٣ أكتوبر"
        k = j + 1 if w == "the" else j
        dm = self._day_of_month(k)
        if dm:
            d, k2 = dm
            if self.at(k2) == "of":
                k2 += 1
            mo = self._month(k2)
            if mo:
                try:
                    return self._roll(date(self.today.year, mo, d)).isoformat(), k2 + 1
                except ValueError:
                    return None
        # "October 12"
        mo = self._month(j)
        if mo:
            dm = self._day_of_month(j + 1)
            if dm:
                try:
                    return self._roll(date(self.today.year, mo, dm[0])).isoformat(), dm[1]
                except ValueError:
                    return None
        return None

    # ---- times ------------------------------------------------------------

    def _period(self, i: int) -> tuple[str, tuple | None, int] | None:
        j = i
        while self.at(j) in _PERIOD_FILLERS and j - i < 2:
            j += 1
        for n in (2, 1):
            key = tuple(self.at(k) for k in range(j, j + n))
            if key in PERIODS:
                hint, window = PERIODS[key]
                return hint, window, j + n
        return None

    def _hour(self, i: int) -> tuple[int, int, int] | None:
        """Return (hour, minute, next index) for a spoken or written clock hour."""
        w = self.at(i)
        m = re.fullmatch(r"(\d{1,2})[:.](\d{2})", w)
        if m and int(m.group(1)) <= 24 and int(m.group(2)) < 60:
            return int(m.group(1)), int(m.group(2)), i + 1
        two = f"{w} {self.at(i + 1)}"
        if two in AR_ORDINAL_HOURS:
            return AR_ORDINAL_HOURS[two], 0, i + 2
        if w in AR_ORDINAL_HOURS:
            return AR_ORDINAL_HOURS[w], 0, i + 1
        if w.isdigit() and 0 < int(w) <= 24:
            return int(w), 0, i + 1
        if w in EN_UNITS and 0 < EN_UNITS[w] <= 12:
            h, j = EN_UNITS[w], i + 1
            nxt = self.at(j)
            if nxt in ("fifteen", "thirty"):
                return h, EN_UNITS.get(nxt) or EN_TENS[nxt], j + 1
            if nxt == "forty" and self.at(j + 1) == "five":
                return h, 45, j + 2
            return h, 0, j
        if w in AR_UNITS and 0 < AR_UNITS[w] <= 12:
            return AR_UNITS[w], 0, i + 1
        return None

    def _minutes_suffix(self, i: int) -> tuple[int, int]:
        """Arabic 'و نص', 'و ربع', 'الا ربع' after an hour."""
        if self.at(i) == AR_AND and self.at(i + 1) in AR_HALF:
            return 30, i + 2
        if self.at(i) == AR_AND and self.at(i + 1) in AR_QUARTER:
            return 15, i + 2
        if self.at(i) in AR_EXCEPT and self.at(i + 1) in AR_QUARTER:
            return -15, i + 2
        if self.at(i) in ("o",) and self.at(i + 1) == "clock":
            return 0, i + 2
        return 0, i

    @staticmethod
    def _to24(h: int, hint: str | None) -> int:
        if hint == "am":
            return 0 if h == 12 else h
        if hint == "pm":
            return h + 12 if h < 12 else h
        if hint == "noon":
            return h if h == 12 or h >= 13 else (h + 12 if h <= 5 else h)
        if 1 <= h <= 7:
            return h + 12
        return h

    def _clock(self, i: int) -> tuple[str, int, str | None] | None:
        """An hour plus optional minutes and period. Returns (HH:MM, next, hint)."""
        hr = self._hour(i)
        if not hr:
            return None
        h, mnt, j = hr
        add, j = self._minutes_suffix(j)
        per = self._period(j)
        hint = None
        if per:
            hint, _, j = per
        total = (self._to24(h, hint) * 60 + mnt + add) % (24 * 60)
        return f"{total // 60:02d}:{total % 60:02d}", j, hint

    def time(self, i: int) -> tuple[str, int] | None:
        w = self.at(i)
        if w in ("noon", "midday"):
            return "12:00", i + 1
        if w in ("half", "quarter") and self.at(i + 1) in ("past", "to"):
            hr = self._hour(i + 2)
            if hr:
                h, _, j = hr
                per = self._period(j)
                hint = per[0] if per else None
                j = per[2] if per else j
                delta = (30 if w == "half" else 15) * (1 if self.at(i + 1) == "past" else -1)
                total = self._to24(h, hint) * 60 + delta
                return f"{total // 60:02d}:{total % 60:02d}", j
        if w in ("at", "around") or w in AR_HOUR_WORDS:
            c = self._clock(i + 1)
            if c and not self._followed_by_scale(i + 1):
                return c[0], c[1]
            return None
        if re.fullmatch(r"\d{1,2}:\d{2}", w):
            c = self._clock(i)
            return (c[0], c[1]) if c else None
        c = self._clock(i)
        if c and (c[2] is not None or self.at(c[1] - 1) == "clock"):
            return c[0], c[1]
        return None

    def _followed_by_scale(self, i: int) -> bool:
        n = self.number(i)
        return bool(n and n[0] > 24)

    def window(self, i: int) -> tuple[tuple[str, str], int] | None:
        w = self.at(i)
        if w in ("between", "from") or w in AR_BETWEEN:
            j = i + 1
            if self.at(j) in AR_HOUR_WORDS:
                j += 1
            a = self._hour(j)
            if a:
                h1, m1, j = a
                if self.at(j) in ("and", "to", "till", "until") or self.at(j) == AR_AND:
                    j += 1
                    if self.at(j) in AR_HOUR_WORDS:
                        j += 1
                    b = self._hour(j)
                    if b:
                        h2, m2, j = b
                        per = self._period(j)
                        hint = per[0] if per else None
                        j = per[2] if per else j
                        s, e = self._to24(h1, hint), self._to24(h2, hint)
                        return (f"{s:02d}:{m1:02d}", f"{e:02d}:{m2:02d}"), j
        if w in ("after",) or (w in AR_AFTER and self.at(i + 1) in AR_HOUR_WORDS):
            j = i + 1 + (1 if self.at(i + 1) in AR_HOUR_WORDS else 0)
            c = self._clock(j)
            if c:
                return (c[0], "21:00"), c[1]
        per = self._period(i)
        if per and per[1]:
            return per[1], per[2]
        return None


def normalize(text: str, today: date) -> Normalized:
    tokens = tokenize(text)
    sc = _Scanner(tokens, today)
    out: list[str] = []
    entities: list[Entity] = []
    i = 0
    while i < len(tokens):
        hit = None
        p = sc.phone(i)
        if p:
            hit = ("phone", p[0], p[1])
        if not hit:
            d = sc.date(i)
            if d:
                hit = ("date", d[0], d[1])
        if not hit:
            t = sc.time(i)
            if t:
                hit = ("time", t[0], t[1])
        if not hit:
            w = sc.window(i)
            if w:
                hit = ("time_window", w[0], w[1])
        if not hit:
            n = sc.number(i)
            if n and not _pronoun_one(tokens, i, n[1]):
                v = n[0]
                hit = ("number", int(v) if float(v).is_integer() else v, n[1])
        if hit:
            kind, value, j = hit
            e = Entity(kind, value, i, j)
            entities.append(e)
            out.append(e.canonical)
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return Normalized(" ".join(out), tokens, entities)


def _pronoun_one(tokens: list[str], i: int, j: int) -> bool:
    if j - i != 1 or tokens[i] != "one":
        return False
    before = tokens[i - 1] if i > 0 else ""
    after = tokens[j] if j < len(tokens) else ""
    return before in _PRONOUN_ONE_BEFORE or after in _PRONOUN_ONE_AFTER
