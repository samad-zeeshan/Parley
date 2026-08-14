"""The eight hand-written v1 callers, kept word for word so v2 numbers can sit next to v1.

Reference day is Thursday 2026-10-01, 09:00 Dubai time. Labels: en, ar-gulf, ar-msa, mixed.
"""

from __future__ import annotations


def T(segments, dialect, intent, slots=None, choice=None):
    if isinstance(segments, str):
        segments = [("ar" if any("؀" <= c <= "ۿ" for c in segments) else "en", segments)]
    return {
        "segments": segments,
        "text": " ".join(t for _, t in segments),
        "dialect": dialect,
        "intent": intent,
        "slots": slots or {},
        "choice": choice,
    }


V1_CALLS = {
    "en-booking": {
        "language": "en", "voice": "piper", "phone": "0501234567",
        "turns": [
            T("Hi, I would like to book a viewing for a two bedroom flat in Dubai Marina.", "en", "book_viewing",
              {"area": "Dubai Marina", "bedrooms": 2}),
            T("My budget is two hundred thousand dirhams a year.", "en", "provide_details", {"budget": 200000}),
            T("Thursday between four and six in the afternoon.", "en", "provide_details",
              {"date": "2026-10-01", "time_window": ["16:00", "18:00"]}),
            T("The first one please.", "en", "choose_option", choice=1),
            T("My number is zero five zero one two three four five six seven.", "en", "provide_details",
              {"phone": "0501234567"}),
            T("Yes, please confirm the booking.", "en", "confirm"),
        ],
    },
    "en-booking-2": {
        "language": "en", "voice": "sapi", "phone": "0523344556",
        "turns": [
            T("Hello, I am looking for a two bedroom in JVC.", "en", "book_viewing",
              {"area": "Jumeirah Village Circle", "bedrooms": 2}),
            T("Saturday morning.", "en", "provide_details",
              {"date": "2026-10-03", "time_window": ["09:00", "12:00"]}),
            T("The second one.", "en", "choose_option", choice=2),
            T("It is zero five two three three four four five five six.", "en", "provide_details",
              {"phone": "0523344556"}),
            T("Yes, go ahead.", "en", "confirm"),
        ],
    },
    "gulf-booking": {
        "language": "ar", "voice": "piper", "phone": "0559876543",
        "turns": [
            T("السلام عليكم، أبي أحجز معاينة لشقة غرفتين في البرشاء", "ar-gulf", "book_viewing",
              {"area": "Al Barsha", "bedrooms": 2}),
            T("الميزانية مية وعشرين ألف درهم", "ar-gulf", "provide_details", {"budget": 120000}),
            T("يوم السبت عقب الظهر", "ar-gulf", "provide_details",
              {"date": "2026-10-03", "time_window": ["12:00", "21:00"]}),
            T("الأول زين", "ar-gulf", "choose_option", choice=1),
            T("رقمي صفر خمسة خمسة تسعة ثمانية سبعة ستة خمسة أربعة ثلاثة", "ar-gulf", "provide_details",
              {"phone": "0559876543"}),
            T("خلاص تمام أكد الحجز", "ar-gulf", "confirm"),
        ],
    },
    "gulf-booking-2": {
        "language": "ar", "voice": "piper-slow", "phone": "0561122334",
        "turns": [
            T("هلا، أبغي شقة غرفتين في جزيرة الريم", "ar-gulf", "book_viewing",
              {"area": "Al Reem Island", "bedrooms": 2}),
            T("يوم الأحد الصبح", "ar-gulf", "provide_details",
              {"date": "2026-10-04", "time_window": ["09:00", "12:00"]}),
            T("لا ما أبي هذا، الثاني", "ar-gulf", "choose_option", choice=2),
            T("صفر خمسة ستة واحد واحد اثنين اثنين ثلاثة ثلاثة أربعة", "ar-gulf", "provide_details",
              {"phone": "0561122334"}),
            T("زين أكد", "ar-gulf", "confirm"),
        ],
    },
    "msa-booking": {
        "language": "ar", "voice": "piper", "phone": "0507654321",
        "turns": [
            T("أريد حجز موعد لمعاينة شقة من ثلاث غرف نوم في جزيرة الريم", "ar-msa", "book_viewing",
              {"area": "Al Reem Island", "bedrooms": 3}),
            T("ميزانيتي مئتا ألف درهم سنويا", "ar-msa", "provide_details", {"budget": 200000}),
            T("يوم الخميس في الساعة العاشرة صباحا", "ar-msa", "provide_details",
              {"date": "2026-10-01", "time_window": ["10:00", "11:00"]}),
            T("الخيار الأول من فضلك", "ar-msa", "choose_option", choice=1),
            T("رقم هاتفي صفر خمسة صفر سبعة ستة خمسة أربعة ثلاثة اثنين واحد", "ar-msa", "provide_details",
              {"phone": "0507654321"}),
            T("نعم أؤكد الحجز", "ar-msa", "confirm"),
        ],
    },
    "msa-booking-2": {
        "language": "ar", "voice": "piper-slow", "phone": "0581239876",
        "turns": [
            T("أبحث عن شقة من غرفتين في مدينة خليفة", "ar-msa", "book_viewing",
              {"area": "Khalifa City", "bedrooms": 2}),
            T("هل يوجد موعد يوم الاثنين في المساء", "ar-msa", "provide_details",
              {"date": "2026-10-05", "time_window": ["17:00", "21:00"]}),
            T("الأول من فضلك", "ar-msa", "choose_option", choice=1),
            T("صفر خمسة ثمانية واحد اثنين ثلاثة تسعة ثمانية سبعة ستة", "ar-msa", "provide_details",
              {"phone": "0581239876"}),
            T("نعم", "ar-msa", "confirm"),
        ],
    },
    "switch-booking": {
        "language": "mixed", "voice": "piper", "phone": "0501112233",
        "turns": [
            T([("en", "Hi,"), ("ar", "أبي أحجز"), ("en", "viewing"), ("ar", "في"), ("en", "Dubai Marina")],
              "mixed", "book_viewing", {"area": "Dubai Marina"}),
            T([("en", "two bedroom"), ("ar", "والميزانية مية وسبعين ألف")], "mixed", "provide_details",
              {"bedrooms": 2, "budget": 170000}),
            T([("ar", "بكرة"), ("en", "in the evening")], "mixed", "provide_details",
              {"date": "2026-10-02", "time_window": ["17:00", "21:00"]}),
            T([("en", "OK,"), ("ar", "الأول")], "mixed", "choose_option", choice=1),
            T("My number is zero five zero one one one two two three three.", "en", "provide_details",
              {"phone": "0501112233"}),
            T([("ar", "تمام"), ("en", "confirm the booking")], "mixed", "confirm"),
        ],
    },
    "switch-booking-2": {
        "language": "mixed", "voice": "piper", "phone": "0544455667",
        "turns": [
            T([("ar", "مرحبا، أبغي"), ("en", "one bedroom apartment"), ("ar", "في البرشاء")], "mixed",
              "book_viewing", {"area": "Al Barsha", "bedrooms": 1}),
            T([("en", "Sunday"), ("ar", "عقب الظهر")], "mixed", "provide_details",
              {"date": "2026-10-04", "time_window": ["12:00", "21:00"]}),
            T([("ar", "الثاني"), ("en", "please")], "mixed", "choose_option", choice=2),
            T("صفر خمسة أربعة أربعة أربعة خمسة خمسة ستة ستة سبعة", "ar-gulf", "provide_details",
              {"phone": "0544455667"}),
            T([("en", "yes"), ("ar", "أكد الحجز")], "mixed", "confirm"),
        ],
    },
}
