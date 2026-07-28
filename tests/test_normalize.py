"""Number, date, time and phone normalization in English, Gulf Arabic and MSA.

The reference day is Thursday 2026-10-01 in every case.
"""

from datetime import date

import pytest

from speech.normalize import normalize

TODAY = date(2026, 10, 1)


def ents(text, kind=None):
    out = normalize(text, TODAY).entities
    return [e.value for e in out if kind is None or e.kind == kind]


# ---- English numbers ------------------------------------------------------

@pytest.mark.parametrize("text,value", [
    ("my budget is one hundred and twenty thousand dirhams a year", 120_000),
    ("120,000 AED", 120_000),
    ("around 95000 dirhams", 95_000),
    ("one and a half million", 1_500_000),
    ("1.5 million", 1_500_000),
    ("two hundred k", 200_000),
    ("150k", 150_000),
    ("eighty five thousand", 85_000),
])
def test_english_amounts(text, value):
    assert value in ents(text, "number")


def test_small_counts_stay_numbers():
    assert ents("a two bedroom flat", "number") == [2]


def test_canonical_text_replaces_words_with_digits():
    assert normalize("budget one hundred and twenty thousand", TODAY).text == "budget 120000"


# ---- Arabic numbers -------------------------------------------------------

@pytest.mark.parametrize("text,value", [
    ("الميزانية مية وعشرين ألف درهم", 120_000),       # Gulf
    ("ميزانيتي مئتا ألف درهم سنويا", 200_000),        # MSA
    ("مليون ونص", 1_500_000),                          # Gulf
    ("خمسة وثمانين ألف", 85_000),
    ("ثلاث مية ألف", 300_000),
    ("ثلاثمائة ألف درهم", 300_000),
    ("ميتين ألف", 200_000),
    ("عشرة آلاف", 10_000),
    ("٢٥٠٠٠٠ درهم", 250_000),                           # Arabic-Indic digits
    ("١٢٠ ألف", 120_000),
    ("مية 20 ألف درهم", 120_000),                       # ASR output mixing words and digits
    ("مية و20 ألف", 120_000),
])
def test_arabic_amounts(text, value):
    assert value in ents(text, "number")


def test_arabic_counts():
    assert ents("ثلاث غرف نوم", "number") == [3]


# ---- phones ---------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "my number is zero five zero one two three four five six seven",
    "my number is 050 123 4567",
    "call me on +971 50 123 4567",
    "0501234567",
    "oh five oh one two three four five six seven",
    "رقمي صفر خمسة صفر واحد اثنين ثلاثة أربعة خمسة ستة سبعة",
    "رقمي ٠٥٠١٢٣٤٥٦٧",
])
def test_phones(text):
    assert ents(text, "phone") == ["0501234567"]


def test_double_digit_words_in_phone():
    assert ents("zero five five double nine eight seven six five four", "phone") == ["0559987654"]


# ---- dates ----------------------------------------------------------------

@pytest.mark.parametrize("text,iso", [
    ("on Thursday", "2026-10-01"),
    ("tomorrow", "2026-10-02"),
    ("the day after tomorrow", "2026-10-03"),
    ("next Sunday", "2026-10-04"),
    ("Saturday", "2026-10-03"),
    ("today", "2026-10-01"),
    ("the 3rd of October", "2026-10-03"),
    ("October 12", "2026-10-12"),
    ("12/10", "2026-10-12"),
    ("يوم الخميس", "2026-10-01"),
    ("بكرة", "2026-10-02"),
    ("باكر", "2026-10-02"),
    ("غدا", "2026-10-02"),
    ("بعد بكرة", "2026-10-03"),
    ("الأحد الجاي", "2026-10-04"),
    ("يوم الأحد", "2026-10-04"),
    ("٣ أكتوبر", "2026-10-03"),
    ("اثنعش أكتوبر", "2026-10-12"),
])
def test_dates(text, iso):
    assert ents(text, "date") == [iso]


# ---- times ----------------------------------------------------------------

@pytest.mark.parametrize("text,hhmm", [
    ("at 4 pm", "16:00"),
    ("at 4pm", "16:00"),
    ("at four p.m.", "16:00"),
    ("at 16:00", "16:00"),
    ("4:30 pm", "16:30"),
    ("at 10 in the morning", "10:00"),
    ("half past five", "17:30"),
    ("at five", "17:00"),              # viewings run 10:00 to 19:30, so a bare 1 to 7 means pm
    ("at eleven", "11:00"),
    ("الساعة أربعة العصر", "16:00"),
    ("الساعة عشر الصبح", "10:00"),
    ("الساعة خمسة ونص المسا", "17:30"),
    ("الساعة العاشرة صباحا", "10:00"),
    ("الساعة الرابعة مساء", "16:00"),
    ("الساعة ثلاث وربع", "15:15"),
    ("الساعة ٤", "16:00"),
])
def test_times(text, hhmm):
    assert ents(text, "time") == [hhmm]


@pytest.mark.parametrize("text,window", [
    ("between four and six pm", ("16:00", "18:00")),
    ("between 4 and 6", ("16:00", "18:00")),
    ("in the afternoon", ("12:00", "17:00")),
    ("in the morning", ("09:00", "12:00")),
    ("in the evening", ("17:00", "21:00")),
    ("after 5", ("17:00", "21:00")),
    ("بين الساعة أربعة وستة", ("16:00", "18:00")),
    ("العصر", ("15:00", "18:00")),
    ("عقب الظهر", ("12:00", "21:00")),
    ("الصبح", ("09:00", "12:00")),
    ("في المساء", ("17:00", "21:00")),
])
def test_time_windows(text, window):
    assert ents(text, "time_window") == [window]


# ---- whole utterances and code switching -----------------------------------

def test_full_english_booking_sentence():
    n = normalize("Thursday between four and six pm, budget 150k, number 050 123 4567", TODAY)
    kinds = {e.kind: e.value for e in n.entities}
    assert kinds == {"date": "2026-10-01", "time_window": ("16:00", "18:00"),
                     "number": 150_000, "phone": "0501234567"}


def test_full_gulf_sentence():
    n = normalize("أبي أحجز يوم الخميس الساعة أربعة العصر والميزانية مية وعشرين ألف", TODAY)
    kinds = {e.kind: e.value for e in n.entities}
    assert kinds == {"date": "2026-10-01", "time": "16:00", "number": 120_000}


def test_code_switched_sentence():
    n = normalize("أبي شقة two bedroom في Dubai Marina يوم Thursday", TODAY)
    kinds = {e.kind: e.value for e in n.entities}
    assert kinds == {"number": 2, "date": "2026-10-01"}


def test_text_without_numbers_is_unchanged_apart_from_orthography():
    assert normalize("Yes please, confirm the booking.", TODAY).text == "yes please confirm the booking"


def test_one_as_pronoun_is_not_a_number():
    assert ents("the first one is fine", "number") == []


def test_phone_is_written_digit_by_digit_in_the_canonical_text():
    assert normalize("call 0501234567 please", TODAY).text == "call 0 5 0 1 2 3 4 5 6 7 please"
