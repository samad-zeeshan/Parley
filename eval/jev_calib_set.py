"""Held-out labelled items for the Jev heads: temperature fitting, threshold choice, and the
model smoke test in ADR 0004. None of these lines is in eval/scripts.py, so the evaluation
never scores a head on the data it was calibrated on. Synthetic only.
"""

from __future__ import annotations

import random
from datetime import date

from api.clock import DUBAI  # noqa: F401 -- keeps the seed's time zone convention explicit
from api.db import connect
from api.seed import seed
from api.service import BookingService

# (text, context the agent last asked about, gold intent)
INTENT = [
    ("I'd like to see a flat in Business Bay", "", "book_viewing"),
    ("can I book a viewing for Saturday", "", "book_viewing"),
    ("أبغي أشوف شقة في الخليج التجاري", "", "book_viewing"),
    ("أرغب في معاينة شقة في شاطئ الراحة", "", "book_viewing"),
    ("أبي أحجز viewing in JLT", "", "book_viewing"),
    ("my budget is ninety thousand", "budget", "provide_details"),
    ("three bedrooms", "bedrooms", "provide_details"),
    ("يوم الثلاثاء الصبح", "date", "provide_details"),
    ("رقمي صفر خمسة اثنين ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة", "phone", "provide_details"),
    ("الأربعاء بعد الساعة خمسة", "date", "provide_details"),
    ("the third one", "offer", "choose_option"),
    ("I'll take option two", "offer", "choose_option"),
    ("الثالث لو سمحت", "offer", "choose_option"),
    ("أختار الخيار الثاني", "offer", "choose_option"),
    ("sure, sounds good", "ask_confirm", "confirm"),
    ("yes that's right", "ask_confirm", "confirm"),
    ("ايه تمام", "ask_confirm", "confirm"),
    ("نعم موافق", "ask_confirm", "confirm"),
    ("no, none of those work", "offer", "deny"),
    ("لا ما يناسبني", "offer", "deny"),
    ("لا شكرا", "offer", "deny"),
    ("nope", "offer", "deny"),
    ("please cancel my viewing", "", "cancel_booking"),
    ("I need to cancel the appointment", "", "cancel_booking"),
    ("أبي ألغي الموعد", "", "cancel_booking"),
    ("أريد إلغاء الحجز", "", "cancel_booking"),
    ("sorry, can you say that again", "date", "repeat"),
    ("عيد مرة ثانية", "date", "repeat"),
    ("what did you say", "phone", "repeat"),
    ("hello there", "", "greet"),
    ("السلام عليكم", "", "greet"),
    ("مرحبا", "", "greet"),
    ("thanks, bye", "", "goodbye"),
    ("مع السلامة", "", "goodbye"),
    ("that's all, goodbye", "", "goodbye"),
    ("what's the weather in Dubai today", "", "out_of_scope"),
    ("who won the football match", "", "out_of_scope"),
    ("شو أخبار الطقس", "", "out_of_scope"),
    ("mmm", "date", "unclear"),
    ("uh", "phone", "unclear"),
    ("اه", "area", "unclear"),
]

# (caller answer to "Shall I confirm the booking?", yes/no)
YES_NO = [
    ("yes please", "yes"), ("go ahead", "yes"), ("sure", "yes"), ("okay do it", "yes"),
    ("ايه أكد", "yes"), ("نعم", "yes"), ("تمام", "yes"), ("أكيد", "yes"), ("yeah why not", "yes"),
    ("no", "no"), ("no thanks", "no"), ("not now", "no"), ("لا", "no"), ("لا ما أبي", "no"),
    ("لا شكرا", "no"), ("wait, no", "no"), ("مب الحين", "no"), ("I changed my mind, no", "no"),
]

# (utterance, dialect option)
DIALECT = [
    ("Is there parking in the building?", "english"), ("I can do Monday after five.", "english"),
    ("How much is the rent per year?", "english"), ("Please call me back tomorrow.", "english"),
    ("وايد زين، عطني الموعد الأول", "gulf"), ("شلون الإيجار؟", "gulf"), ("الحين أبي أشوف الشقة", "gulf"),
    ("ما أبي هذا، أبغي شي ثاني", "gulf"), ("بكرة عقب الظهر يناسبني", "gulf"),
    ("هل يمكنني معاينة الشقة غدا؟", "msa"), ("أود معرفة قيمة الإيجار السنوي", "msa"),
    ("سوف أحضر في الموعد المحدد", "msa"), ("لماذا لا يوجد موعد صباحا؟", "msa"),
    ("أريد شقة قريبة من المترو", "msa"),
    ("أبي studio في Business Bay", "switch"), ("the budget هو مية ألف", "switch"),
    ("OK خلاص book it", "switch"), ("عندك anything on Friday?", "switch"),
    ("yalla أكد الحجز please", "switch"),
]


def judge_items(n: int = 30, rng_seed: int = 11) -> list[tuple[str, list[dict], str]]:
    """(reply, api slots, grounded|not_grounded). Grounded replies restate the slot; the others change
    one fact: the day, the time, the rent, the tower, or the area."""
    from dialogue.phrasing import fmt_date

    conn = connect(":memory:")
    anchor = date(2026, 10, 1)
    seed(conn, anchor=anchor, rng_seed=7)

    class _Clock:
        def now(self):
            from datetime import datetime

            return datetime(2026, 10, 1, 9, 0, tzinfo=DUBAI)

    svc = BookingService(conn, _Clock())
    slots = svc.list_slots(limit=200)
    rng = random.Random(rng_seed)
    out = []
    for k in range(n):
        s = rng.choice(slots)
        lang = "ar" if k % 2 else "en"
        when, hhmm, rent = fmt_date(s["starts_at"], lang), s["starts_at"][11:16], s["annual_rent_aed"]
        area, addr = (s["area_ar"], s["address_ar"]) if lang == "ar" else (s["area"], s["address"])
        fault = k % 3  # a third grounded, two thirds with one changed fact
        if fault == 1:
            hhmm = f"{(int(hhmm[:2]) + rng.choice([1, 2, -1])) % 24:02d}:30"
        elif fault == 2:
            rent = rent + rng.choice([5000, 10000, -5000, 20000])
        if lang == "en":
            reply = f"I have {when} at {hhmm} at {addr}, AED {rent:,} a year."
        else:
            reply = f"عندي موعد {when} الساعة {hhmm} في {addr}، الإيجار {rent:,} درهم في السنة."
        out.append((reply, [s], "grounded" if fault == 0 else "not_grounded"))
    return out
