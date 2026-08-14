"""Thirty-two more scripted callers, eight per language class, built from phrase banks with a fixed seed.

Every line was written for this project. Call parameters are drawn against the seeded listings so a correct agent can book.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from api.clock import DUBAI, FixedClock
from api.db import connect
from api.seed import area_names, seed
from api.service import BookingService

from .v1 import T

ANCHOR = datetime(2026, 10, 1, 9, 0, tzinfo=DUBAI)
AR_AREA = area_names()

EN_DAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AR_DAY = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]

# Window per spoken period, as speech/normalize.py defines them.
EN_PERIOD = {"morning": ("09:00", "12:00"), "afternoon": ("12:00", "17:00"), "evening": ("17:00", "21:00")}
GULF_PERIOD = {"الصبح": ("09:00", "12:00"), "عقب الظهر": ("12:00", "21:00"), "العصر": ("15:00", "18:00"),
               "المسا": ("17:00", "21:00")}
MSA_PERIOD = {"صباحا": ("09:00", "12:00"), "بعد الظهر": ("12:00", "21:00"), "في المساء": ("17:00", "21:00"),
              "عصرا": ("15:00", "18:00")}

EN_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
AR_DIGITS = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
EN_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen"]
EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
GULF_TENS = ["", "عشرة", "عشرين", "ثلاثين", "أربعين", "خمسين", "ستين", "سبعين", "ثمانين", "تسعين"]
MSA_TENS = ["", "عشرة", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون"]
GULF_HUNDREDS = ["", "مية", "ميتين", "ثلاثمية"]
MSA_HUNDREDS = ["", "مئة", "مئتا", "ثلاثمئة"]
AR_UNITS = ["", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
GULF_TEENS = ["عشرة", "احدعش", "اثنعش", "ثلطعش", "اربعطعش", "خمسطعش", "ستطعش", "سبعطعش", "ثمنطعش", "تسعطعش"]
MSA_TEENS = ["عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر", "ستة عشر", "سبعة عشر",
             "ثمانية عشر", "تسعة عشر"]
EN_ORD = ["", "first", "second", "third"]
AR_ORD = ["", "الأول", "الثاني", "الثالث"]


def en_thousands(n: int) -> str:
    k = n // 1000
    h, rest = divmod(k, 100)
    parts = [f"{EN_UNITS[h]} hundred"] if h else []
    if rest:
        if rest < 10:
            words = EN_UNITS[rest]
        elif rest < 20:
            words = EN_TEENS[rest - 10]
        else:
            words = EN_TENS[rest // 10] + (f" {EN_UNITS[rest % 10]}" if rest % 10 else "")
        parts.append(("and " if h else "") + words)
    return " ".join(parts) + " thousand"


def ar_thousands(n: int, msa: bool) -> str:
    """Arabic reads units before tens: 125 thousand is "hundred and five and twenty thousand"."""
    k = n // 1000
    h, rest = divmod(k, 100)
    tens, units = divmod(rest, 10)
    hundreds = (MSA_HUNDREDS if msa else GULF_HUNDREDS)[h]
    if msa and h == 2 and rest:
        hundreds = "مئتان"  # the construct form مئتا only stands directly before the noun
    parts = [hundreds] if h else []
    if tens == 1:
        parts.append((MSA_TEENS if msa else GULF_TEENS)[units])
    else:
        if units:
            parts.append(AR_UNITS[units])
        if tens:
            parts.append((MSA_TENS if msa else GULF_TENS)[tens])
    return " و".join(parts) + (" آلاف" if 3 <= rest <= 10 else " ألف")


def en_digits(phone: str) -> str:
    return " ".join(EN_UNITS[int(c)] for c in phone)


def ar_digits(phone: str) -> str:
    return " ".join(AR_DIGITS[int(c)] for c in phone)


def en_beds(n: int) -> str:
    return "studio" if n == 0 else f"{EN_UNITS[n]} bedroom"


def gulf_beds(n: int) -> str:
    return {0: "استوديو", 1: "غرفة وحدة", 2: "غرفتين", 3: "ثلاث غرف"}[n]


def msa_beds(n: int) -> str:
    return {0: "استوديو", 1: "غرفة واحدة", 2: "غرفتين", 3: "ثلاث غرف"}[n]


def _feasible(rng: random.Random, svc: BookingService, periods: dict) -> dict:
    """Pick area, bedrooms, day and period until the listings offer at least the chosen option."""
    areas = sorted(AR_AREA)
    while True:
        area = rng.choice(areas)
        beds = rng.choice([0, 1, 2, 2, 3])
        offset = rng.randint(1, 6)  # Friday to Wednesday; Thursday is today and reads ambiguously
        day = ANCHOR.date() + timedelta(days=offset)
        period = rng.choice(list(periods) + [None])
        window = periods[period] if period else None
        slots = svc.list_slots(area=area, bedrooms=beds, date=day.isoformat(), window=window, limit=5)[:3]
        if not slots:
            continue
        choice = rng.randint(1, len(slots))
        top = max(s["annual_rent_aed"] for s in slots)
        budget = (top // 5000 + rng.randint(1, 3)) * 5000
        return {"area": area, "bedrooms": beds, "date": day, "period": period, "window": window,
                "choice": choice, "budget": budget, "tomorrow": offset == 1 and rng.random() < 0.5}


def _phone(rng: random.Random) -> str:
    return "05" + str(rng.choice([0, 2, 5, 6, 8])) + "".join(str(rng.randint(0, 9)) for _ in range(7))


def _slots(p: dict, *keys) -> dict:
    out = {}
    for k in keys:
        if k == "date":
            out["date"] = p["date"].isoformat()
        elif k == "time_window":
            if p["window"]:
                out["time_window"] = list(p["window"])
        else:
            out[k] = p[k]
    return out


# ---------------------------------------------------------------------------
# One builder per language class. Each returns eight lines in call order.
# The third line is volunteered by the caller (a side question or a filler),
# after the in-the-wild turn types in arXiv 2605.16364: not every turn is a request.
# ---------------------------------------------------------------------------

def english(rng: random.Random, p: dict, phone: str) -> list[dict]:
    area, n = p["area"], p["bedrooms"]
    with_beds = rng.random() < 0.4
    greet = T(rng.choice(["Hello, good morning.", "Hi there.", "Good afternoon.", "Hello?"]), "en", "greet")
    if with_beds:
        opener = T(rng.choice([f"I want to view a {en_beds(n)} in {area}.",
                               f"Do you have a {en_beds(n)} flat in {area} I could see?"]),
                   "en", "book_viewing", _slots(p, "area", "bedrooms"))
    else:
        opener = T(rng.choice([f"I would like to book a viewing in {area}.",
                               f"I am calling about apartments in {area}.",
                               f"Can I see a flat in {area} please?"]), "en", "book_viewing", _slots(p, "area"))
    side, side_intent = rng.choice([("Is there parking in the building?", "out_of_scope"),
                                    ("What is the weather like this week?", "out_of_scope"),
                                    ("Sorry, one second.", "unclear"), ("Are pets allowed?", "out_of_scope")])
    volunteer = T(side, "en", side_intent)
    volunteer["volunteer"] = True
    budget_words = en_thousands(p["budget"])
    if with_beds:
        beds = T(f"My budget is {budget_words} dirhams.", "en", "provide_details", _slots(p, "budget"))
        beds["volunteer"] = True
    elif rng.random() < 0.5:
        beds = T(f"{en_beds(n).capitalize()}, and my budget is {budget_words} dirhams.", "en", "provide_details",
                 _slots(p, "bedrooms", "budget"))
    else:
        beds = T(rng.choice([f"A {en_beds(n)} please.", f"{en_beds(n).capitalize()}."]), "en", "provide_details",
                 _slots(p, "bedrooms"))
    day = "Tomorrow" if p["tomorrow"] else EN_DAY[p["date"].weekday()]
    if p["period"]:
        on_day = "tomorrow" if p["tomorrow"] else f"on {day}"
        when = rng.choice([f"{day} in the {p['period']}.", f"Could I come {on_day} in the {p['period']}?"])
    else:
        when = rng.choice([f"{day} is fine, any time.", f"{day} works for me."])
    date_line = T(when, "en", "provide_details", _slots(p, "date", "time_window"))
    k = p["choice"]
    choice = T(rng.choice([f"The {EN_ORD[k]} one please.", f"I'll take the {EN_ORD[k]} one.", f"Option {EN_UNITS[k]}."]),
               "en", "choose_option", choice=k)
    ph = T(rng.choice([f"My number is {en_digits(phone)}.", f"It's {en_digits(phone)}.",
                       f"You can reach me on {en_digits(phone)}."]), "en", "provide_details", {"phone": phone})
    confirm = T(rng.choice(["Yes, please book it.", "Yes, confirm it.", "That's right, go ahead.", "Sure, book it."]),
                "en", "confirm")
    return [greet, opener, volunteer, beds, date_line, choice, ph, confirm]


def gulf(rng: random.Random, p: dict, phone: str) -> list[dict]:
    area, n = AR_AREA[p["area"]], p["bedrooms"]
    with_beds = rng.random() < 0.4
    G = "ar-gulf"
    greet = T(rng.choice(["السلام عليكم", "مرحبا", "هلا والله", "صباح الخير"]), G, "greet")
    if with_beds:
        opener = T(rng.choice([f"أبي شقة {gulf_beds(n)} في {area}", f"أبغي أشوف {gulf_beds(n)} في {area}"]), G,
                   "book_viewing", _slots(p, "area", "bedrooms"))
    else:
        opener = T(rng.choice([f"أبي أشوف شقة في {area}", f"عندكم شقق في {area}؟ أبي أحجز معاينة",
                               f"أبغي أحجز معاينة في {area}", f"ودي أشوف شقة في {area}"]), G, "book_viewing",
                   _slots(p, "area"))
    side, side_intent = rng.choice([("الجو حار وايد هالأيام، صح؟", "out_of_scope"),
                                    ("في مواقف سيارات في البناية؟", "out_of_scope"),
                                    ("لحظة الله يخليك", "unclear"), ("عادي أجيب قطوتي معاي؟", "out_of_scope")])
    volunteer = T(side, G, side_intent)
    volunteer["volunteer"] = True
    budget_words = ar_thousands(p["budget"], msa=False)
    if with_beds:
        beds = T(f"الميزانية {budget_words} درهم", G, "provide_details", _slots(p, "budget"))
        beds["volunteer"] = True
    elif rng.random() < 0.5:
        beds = T(f"{gulf_beds(n)}، والميزانية {budget_words}", G, "provide_details", _slots(p, "bedrooms", "budget"))
    else:
        beds = T(rng.choice([f"{gulf_beds(n)}", f"أبي {gulf_beds(n)}"]), G, "provide_details", _slots(p, "bedrooms"))
    day = "بكرة" if p["tomorrow"] else f"يوم {AR_DAY[p['date'].weekday()]}"
    when = f"{day} {p['period']}" if p["period"] else rng.choice([f"{day} أي وقت", f"{day} يناسبني"])
    date_line = T(when, G, "provide_details", _slots(p, "date", "time_window"))
    k = p["choice"]
    choice = T(rng.choice([f"{AR_ORD[k]} زين", f"أبي {AR_ORD[k]}", f"خلني آخذ {AR_ORD[k]}"]), G, "choose_option",
               choice=k)
    ph = T(rng.choice([f"رقمي {ar_digits(phone)}", f"{ar_digits(phone)}", f"سجل رقمي {ar_digits(phone)}"]), G,
           "provide_details", {"phone": phone})
    confirm = T(rng.choice(["أكد الحجز الله يعافيك", "زين، أكد", "تمام أكد", "إي، احجزه"]), G, "confirm")
    return [greet, opener, volunteer, beds, date_line, choice, ph, confirm]


def msa(rng: random.Random, p: dict, phone: str) -> list[dict]:
    area, n = AR_AREA[p["area"]], p["bedrooms"]
    with_beds = rng.random() < 0.4
    M = "ar-msa"
    greet = T(rng.choice(["السلام عليكم", "مرحبا", "مساء الخير", "أهلا"]), M, "greet")
    if with_beds:
        opener = T(f"أريد معاينة شقة من {msa_beds(n)} في {area}", M, "book_viewing", _slots(p, "area", "bedrooms"))
    else:
        opener = T(rng.choice([f"أريد حجز موعد لمعاينة شقة في {area}", f"أبحث عن شقة للإيجار في {area}",
                               f"هل لديكم شقق في {area}؟ أود حجز معاينة", f"أرغب في معاينة شقة في {area}"]), M,
                   "book_viewing", _slots(p, "area"))
    side, side_intent = rng.choice([("هل الطقس مناسب هذا الأسبوع؟", "out_of_scope"),
                                    ("هل يوجد موقف للسيارات؟", "out_of_scope"),
                                    ("لحظة من فضلك", "unclear"), ("هل يسمح بالحيوانات الأليفة؟", "out_of_scope")])
    volunteer = T(side, M, side_intent)
    volunteer["volunteer"] = True
    budget_words = ar_thousands(p["budget"], msa=True)
    if with_beds:
        beds = T(f"ميزانيتي {budget_words} درهم سنويا", M, "provide_details", _slots(p, "budget"))
        beds["volunteer"] = True
    elif rng.random() < 0.5:
        beds = T(f"{msa_beds(n)}، وميزانيتي {budget_words} درهم", M, "provide_details",
                 _slots(p, "bedrooms", "budget"))
    else:
        beds = T(rng.choice([f"{msa_beds(n)}", f"أحتاج {msa_beds(n)}"]), M, "provide_details", _slots(p, "bedrooms"))
    day = "غدا" if p["tomorrow"] else f"يوم {AR_DAY[p['date'].weekday()]}"
    when = f"{day} {p['period']}" if p["period"] else rng.choice([f"{day} في أي وقت", f"هل يمكن {day}؟"])
    date_line = T(when, M, "provide_details", _slots(p, "date", "time_window"))
    k = p["choice"]
    choice = T(rng.choice([f"الخيار {AR_ORD[k]} من فضلك", f"أختار {AR_ORD[k]}", f"{AR_ORD[k]} لو سمحت"]), M,
               "choose_option", choice=k)
    ph = T(rng.choice([f"رقم هاتفي {ar_digits(phone)}", f"رقمي هو {ar_digits(phone)}"]), M, "provide_details",
           {"phone": phone})
    confirm = T(rng.choice(["نعم أؤكد الحجز", "نعم من فضلك", "موافق، أكد الحجز", "نعم"]), M, "confirm")
    return [greet, opener, volunteer, beds, date_line, choice, ph, confirm]


def switched(rng: random.Random, p: dict, phone: str) -> list[dict]:
    """Gulf Arabic with English inserted, and English with Arabic inserted, inside one line."""
    area_en, area_ar, n = p["area"], AR_AREA[p["area"]], p["bedrooms"]
    X = "mixed"
    greet = T(rng.choice([[("en", "Hi,"), ("ar", "السلام عليكم")], [("ar", "هلا"), ("en", "good morning")]]), X, "greet")
    opener = T(rng.choice([[("ar", "أبي أحجز"), ("en", "viewing"), ("ar", f"في {area_ar}")],
                           [("en", "Hello, I want to see a flat"), ("ar", f"في {area_ar}")],
                           [("ar", "عندكم"), ("en", f"apartments in {area_en}")],
                           [("ar", "أبغي"), ("en", f"a flat in {area_en}")]]), X, "book_viewing", _slots(p, "area"))
    side, side_intent = rng.choice([([("ar", "في"), ("en", "parking"), ("ar", "ولا لا؟")], "out_of_scope"),
                                    ([("en", "sorry,"), ("ar", "لحظة")], "unclear"),
                                    ([("en", "by the way, how is the weather"), ("ar", "هالأيام؟")], "out_of_scope")])
    volunteer = T(side, X, side_intent)
    volunteer["volunteer"] = True
    if rng.random() < 0.5:
        beds = T([("en", en_beds(n)), ("ar", f"والميزانية {ar_thousands(p['budget'], msa=False)}")], X,
                 "provide_details", _slots(p, "bedrooms", "budget"))
    else:
        beds = T(rng.choice([[("ar", "أبي"), ("en", en_beds(n))], [("en", en_beds(n)), ("ar", "لو سمحت")]]), X,
                 "provide_details", _slots(p, "bedrooms"))
    if p["tomorrow"]:
        day_ar, day_en = "بكرة", "tomorrow"
    else:
        day_ar, day_en = f"يوم {AR_DAY[p['date'].weekday()]}", EN_DAY[p["date"].weekday()]
    # The period is spoken in the other language from the day, so each date line really switches.
    en_period = {"الصبح": "morning", "عقب الظهر": "afternoon", "العصر": "afternoon", "المسا": "evening"}
    if p["period"] and rng.random() < 0.5:
        segs = [("en", day_en), ("ar", p["period"])]
        window = GULF_PERIOD[p["period"]]
    elif p["period"]:
        word = en_period[p["period"]]
        segs = [("ar", day_ar), ("en", f"in the {word}")]
        window = EN_PERIOD[word]
    else:
        segs = [("en", day_en), ("ar", "أي وقت")]
        window = None
    slots = {"date": p["date"].isoformat(), **({"time_window": list(window)} if window else {})}
    date_line = T(segs, X, "provide_details", slots)
    k = p["choice"]
    choice = T(rng.choice([[("en", "OK,"), ("ar", AR_ORD[k])], [("ar", AR_ORD[k]), ("en", "please")],
                           [("en", f"the {EN_ORD[k]} one"), ("ar", "لو سمحت")]]), X, "choose_option", choice=k)
    ph = T([("en", "my number is"), ("ar", ar_digits(phone))], X, "provide_details", {"phone": phone})
    confirm = T(rng.choice([[("ar", "تمام"), ("en", "confirm the booking")], [("en", "yes"), ("ar", "أكد")],
                            [("en", "OK go ahead"), ("ar", "احجزه")]]), X, "confirm")
    return [greet, opener, volunteer, beds, date_line, choice, ph, confirm]


KINDS = {
    "en": (english, EN_PERIOD, ["piper", "sapi", "sapi-david"]),
    "gulf": (gulf, GULF_PERIOD, ["piper", "piper-slow"]),
    "msa": (msa, MSA_PERIOD, ["piper", "piper-slow"]),
    "switch": (switched, GULF_PERIOD, ["piper"]),
}
LANGUAGE = {"en": "en", "gulf": "ar", "msa": "ar", "switch": "mixed"}


def generate(per_kind: int = 8, rng_seed: int = 2026) -> dict:
    conn = connect(":memory:")
    seed(conn, anchor=ANCHOR.date(), rng_seed=7)
    svc = BookingService(conn, FixedClock(ANCHOR))
    rng = random.Random(rng_seed)
    calls = {}
    for kind, (build, periods, voices) in KINDS.items():
        for i in range(per_kind):
            p = _feasible(rng, svc, periods)
            phone = _phone(rng)
            calls[f"{kind}-gen-{i + 1:02d}"] = {"language": LANGUAGE[kind], "voice": voices[i % len(voices)],
                                                "phone": phone, "turns": build(rng, p, phone)}
    conn.close()
    return calls
