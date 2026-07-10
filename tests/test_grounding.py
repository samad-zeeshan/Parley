"""The agent never states a slot, price or address the API did not return.

The LLM only phrases. Its reply is checked against the facts from this turn's
API results; any time, date, number, area, tower or slot id that is not among
those facts rejects the reply, and the deterministic template is spoken instead.
"""

from datetime import date

import pytest

from dialogue.grounding import Facts, ungrounded
from dialogue.phrasing import Phraser
from dialogue.policy import Action

TODAY = date(2026, 10, 1)

SLOT = {
    "slot_id": "s-p001-100116", "starts_at": "2026-10-01T16:00", "ends_at": "2026-10-01T16:30",
    "property_ref": "MJ-001", "emirate": "Dubai", "area": "Dubai Marina", "area_ar": "دبي مارينا",
    "address": "Unit 1204, Tower C3, Dubai Marina, Dubai", "address_ar": "شقة 1204، برج ت3، دبي مارينا، دبي",
    "bedrooms": 2, "annual_rent_aed": 155000, "agent_name": "Omar", "agent_name_ar": "عمر",
}
FACTS = Facts.from_slots([SLOT])


@pytest.mark.parametrize("reply", [
    "I have Thursday at 16:00 in Dubai Marina, a two bedroom for AED 155,000 a year.",
    "Your viewing is at Unit 1204, Tower C3, Dubai Marina on Thursday 1 October at 16:00.",
    "عندي موعد يوم الخميس الساعة 16:00 في دبي مارينا، غرفتين، الإيجار 155000 درهم في السنة.",
    "Which option do you want? Say one.",
])
def test_grounded_replies_pass(reply):
    assert ungrounded(reply, FACTS, TODAY) == []


@pytest.mark.parametrize("reply,what", [
    ("I also have Friday at 14:30 in Dubai Marina.", "date"),                  # invented slot day
    ("I have Thursday at 14:30 in Dubai Marina.", "time"),                     # invented slot time
    ("It is AED 120,000 a year.", "number"),                                   # invented price
    ("The flat is in Tower Z9, Dubai Marina.", "number"),                      # invented address
    ("There is also one in Downtown Dubai.", "area"),                          # area the API did not return
    ("عندي موعد يوم الجمعة الساعة 16:00", "date"),                            # Arabic, invented day
    ("الإيجار مية ألف درهم", "number"),                                        # Arabic, invented price
    ("Slot s-p009-100511 is free.", "slot_id"),                                # invented id
])
def test_ungrounded_replies_are_caught(reply, what):
    kinds = {v.kind for v in ungrounded(reply, FACTS, TODAY)}
    assert what in kinds


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def phrase(self, action, facts_json, lang):
        self.calls += 1
        return self.reply


def test_injected_hallucinated_slot_is_rejected_and_template_is_spoken():
    llm = FakeLLM("Great news, I have Thursday at 16:00 and also Friday at 14:30 in Dubai Marina.")
    phraser = Phraser(llm=llm, today=TODAY)
    out = phraser.render(Action("offer", {"slots": [SLOT]}), lang="en")
    assert llm.calls == 1
    assert out.source == "template"
    assert out.rejected and any(v.kind == "date" for v in out.rejected)
    assert "14:30" not in out.text and "Friday" not in out.text
    assert "16:00" in out.text


def test_grounded_llm_reply_is_used():
    llm = FakeLLM("I found a two bedroom in Dubai Marina, Thursday at 16:00, AED 155,000 a year. Shall I hold it?")
    out = Phraser(llm=llm, today=TODAY).render(Action("offer", {"slots": [SLOT]}), lang="en")
    assert out.source == "llm" and out.rejected == []


def test_templates_are_grounded_in_both_languages():
    phraser = Phraser(llm=None, today=TODAY)
    for lang in ("en", "ar"):
        for act in (Action("offer", {"slots": [SLOT]}), Action("ask_confirm", {"slot": SLOT}),
                    Action("confirmed", {"slot": SLOT, "booking_id": "b-0123456789ab"})):
            out = phraser.render(act, lang=lang)
            assert out.source == "template"
            assert ungrounded(out.text, Facts.from_slots([SLOT]), TODAY) == [], (lang, act.name, out.text)
