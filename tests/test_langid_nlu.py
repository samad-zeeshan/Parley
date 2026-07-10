"""Per-utterance language and dialect identification, and the rule-based NLU that
backs the LLM (and runs alone in the hermetic suite)."""

from datetime import date

import pytest
from jsonschema import validate

from dialogue.nlu import RuleNLU
from dialogue.schema import NLU_SCHEMA
from speech.langid import identify, reply_language

TODAY = date(2026, 10, 1)


@pytest.mark.parametrize("text,label", [
    ("I want to book a viewing in Dubai Marina", "en"),
    ("أبي أحجز معاينة لشقة في دبي مارينا", "ar-gulf"),
    ("أبغي شقة غرفتين في البرشاء", "ar-gulf"),
    ("لا ما أبي هذا الموعد عطني وقت ثاني", "ar-gulf"),
    ("أريد حجز موعد لمعاينة شقة في أبوظبي", "ar-msa"),
    ("هل يوجد موعد يوم الأحد في الساعة العاشرة صباحا", "ar-msa"),
    ("أبي شقة two bedroom في Dubai Marina", "mixed"),
    ("OK خلاص confirm the booking", "mixed"),
])
def test_identify(text, label):
    assert identify(text).label == label


def test_reply_language_follows_the_caller():
    assert reply_language(identify("hello")) == "en"
    assert reply_language(identify("أبي أحجز")) == "ar"
    assert reply_language(identify("أبي شقة two bedroom في Dubai Marina")) == "ar"  # Arabic matrix
    assert reply_language(identify("OK confirm the booking يوم الخميس please")) == "en"


nlu = RuleNLU()


@pytest.mark.parametrize("text,intent,slots", [
    ("I'd like to book a viewing for a two bedroom flat in Dubai Marina", "book_viewing",
     {"area": "Dubai Marina", "bedrooms": 2}),
    ("my budget is one hundred and fifty thousand dirhams", "provide_details", {"budget": 150000}),
    ("Thursday between four and six pm", "provide_details",
     {"date": "2026-10-01", "time_window": ["16:00", "18:00"]}),
    ("my number is 050 123 4567", "provide_details", {"phone": "0501234567"}),
    ("yes please confirm", "confirm", {}),
    ("no, not that one", "deny", {}),
    ("the second one", "choose_option", {}),
    ("cancel my booking", "cancel_booking", {}),
    ("a studio in JLT", "book_viewing", {"area": "Jumeirah Lake Towers", "bedrooms": 0}),
    ("أبي أحجز معاينة لشقة غرفتين في دبي مارينا", "book_viewing", {"area": "Dubai Marina", "bedrooms": 2}),
    ("أبحث عن شقة من ثلاث غرف نوم في جزيرة الريم", "book_viewing", {"area": "Al Reem Island", "bedrooms": 3}),
    ("الميزانية مية وعشرين ألف درهم", "provide_details", {"budget": 120000}),
    ("يوم الخميس الساعة أربعة العصر", "provide_details", {"date": "2026-10-01", "time_window": ["16:00", "17:00"]}),
    ("خلاص تمام أكد الحجز", "confirm", {}),
    ("نعم أؤكد الحجز من فضلك", "confirm", {}),
    ("لا ما أبي هذا الموعد", "deny", {}),
    ("الأول", "choose_option", {}),
    ("ألغ الحجز", "cancel_booking", {}),
    ("رقمي صفر خمسة صفر واحد اثنين ثلاثة أربعة خمسة ستة سبعة", "provide_details", {"phone": "0501234567"}),
    ("أبي شقة two bedroom في Dubai Marina يوم Thursday", "book_viewing",
     {"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"}),
    ("what's the weather tomorrow", "out_of_scope", {"date": "2026-10-02"}),
])
def test_rule_nlu(text, intent, slots):
    out = nlu.parse(text, TODAY)
    assert out.intent == intent
    assert out.slots == slots
    validate(out.to_json(), NLU_SCHEMA)


def test_choice_numbers():
    assert nlu.parse("the second one", TODAY).choice == 2
    assert nlu.parse("option 3", TODAY).choice == 3
    assert nlu.parse("الأول", TODAY).choice == 1
    assert nlu.parse("الثاني", TODAY).choice == 2


# ---- the model path, with the model faked ---------------------------------

class FakeLLMNLU:
    """LLMNLU with the HTTP call replaced by a canned answer."""

    def __new__(cls, answer):
        from dialogue.nlu import LLMNLU

        n = LLMNLU(base_url="http://127.0.0.1:9")  # nothing listens here
        n.raw = lambda norm, today, context="": answer
        return n


def test_llm_output_is_schema_checked_and_falls_back_to_rules():
    n = FakeLLMNLU('{"intent": "book_a_yacht", "slots": {}}')
    out = n.parse("I want a two bedroom in Dubai Marina", TODAY)
    assert out.source == "rules-fallback" and out.intent == "book_viewing"
    assert n.rejections == 1


def test_llm_slots_without_support_in_the_utterance_are_dropped():
    # Seen in the smoke test: the model filled a phone number and a time nobody said.
    n = FakeLLMNLU('{"intent": "book_viewing", "slots": {"area": "Dubai Marina", "bedrooms": 2, '
                   '"phone": "0501234567", "time_window": ["14:00", "15:00"]}}')
    out = n.parse("I want a two bedroom in Dubai Marina", TODAY)
    assert out.source == "llm"
    assert out.slots == {"area": "Dubai Marina", "bedrooms": 2}
    assert set(out.dropped) == {"phone", "time_window"}


def test_llm_supported_slots_are_kept():
    n = FakeLLMNLU('{"intent": "provide_details", "slots": {"phone": "0501234567", "budget": 120000}}')
    out = n.parse("budget 120000, number 050 123 4567", TODAY)
    assert out.slots == {"phone": "0501234567", "budget": 120000}


def test_unreachable_model_falls_back():
    from dialogue.nlu import LLMNLU

    n = LLMNLU(base_url="http://127.0.0.1:9", timeout=0.5)
    assert n.parse("yes please confirm", TODAY).source == "rules-fallback"
