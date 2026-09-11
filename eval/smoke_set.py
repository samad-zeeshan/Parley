"""The 20-utterance smoke set used to choose the ASR model. Synthetic only.

Each item is a list of (language, text) segments so a code-switched line is voiced per segment.
"""

SMOKE_SET = [
    # English, Piper lessac and Windows SAPI Zira alternate.
    {"id": "en-01", "dialect": "en", "voice": "piper", "segments": [("en", "Hi, I would like to book a viewing for a two bedroom flat in Dubai Marina.")]},
    {"id": "en-02", "dialect": "en", "voice": "sapi", "segments": [("en", "My budget is one hundred and twenty thousand dirhams a year.")]},
    {"id": "en-03", "dialect": "en", "voice": "piper", "segments": [("en", "Is there anything on Thursday between four and six in the afternoon?")]},
    {"id": "en-04", "dialect": "en", "voice": "sapi", "segments": [("en", "My number is zero five zero one two three four five six seven.")]},
    {"id": "en-05", "dialect": "en", "voice": "piper", "segments": [("en", "Yes please, confirm the booking.")]},
    {"id": "en-06", "dialect": "en", "voice": "sapi", "segments": [("en", "Actually, cancel that booking please.")]},
    # Gulf Arabic wording (Emirati lexicon). Voice is the Jordanian Piper speaker.
    {"id": "gulf-01", "dialect": "ar-gulf", "voice": "piper", "segments": [("ar", "أبي أحجز معاينة لشقة في دبي مارينا")]},
    {"id": "gulf-02", "dialect": "ar-gulf", "voice": "piper", "segments": [("ar", "أبغي شقة غرفتين في البرشاء والميزانية مية وعشرين ألف درهم")]},
    {"id": "gulf-03", "dialect": "ar-gulf", "voice": "piper", "segments": [("ar", "يوم الخميس الساعة أربعة العصر زين")]},
    {"id": "gulf-04", "dialect": "ar-gulf", "voice": "piper", "segments": [("ar", "رقمي صفر خمسة صفر واحد اثنين ثلاثة أربعة خمسة ستة سبعة")]},
    {"id": "gulf-05", "dialect": "ar-gulf", "voice": "piper", "segments": [("ar", "لا ما أبي هذا الموعد عطني وقت ثاني عقب الظهر")]},
    {"id": "gulf-06", "dialect": "ar-gulf", "voice": "piper", "segments": [("ar", "خلاص تمام أكد الحجز")]},
    # Modern Standard Arabic.
    {"id": "msa-01", "dialect": "ar-msa", "voice": "piper", "segments": [("ar", "أريد حجز موعد لمعاينة شقة في أبوظبي")]},
    {"id": "msa-02", "dialect": "ar-msa", "voice": "piper", "segments": [("ar", "أبحث عن شقة من ثلاث غرف نوم في جزيرة الريم")]},
    {"id": "msa-03", "dialect": "ar-msa", "voice": "piper", "segments": [("ar", "ميزانيتي مئتا ألف درهم سنويا")]},
    {"id": "msa-04", "dialect": "ar-msa", "voice": "piper", "segments": [("ar", "هل يوجد موعد يوم الأحد في الساعة العاشرة صباحا")]},
    {"id": "msa-05", "dialect": "ar-msa", "voice": "piper", "segments": [("ar", "نعم أؤكد الحجز من فضلك")]},
    {"id": "msa-06", "dialect": "ar-msa", "voice": "piper", "segments": [("ar", "رقم هاتفي صفر خمسة خمسة تسعة ثمانية سبعة ستة خمسة أربعة ثلاثة")]},
    # Code switching, each segment voiced in its own language.
    {"id": "mix-01", "dialect": "mixed", "voice": "piper", "segments": [("ar", "أبي شقة"), ("en", "two bedroom"), ("ar", "في"), ("en", "Dubai Marina")]},
    {"id": "mix-02", "dialect": "mixed", "voice": "piper", "segments": [("en", "OK"), ("ar", "خلاص"), ("en", "confirm the booking"), ("ar", "يوم الخميس")]},
]


def reference_text(item: dict) -> str:
    return " ".join(text for _, text in item["segments"])
