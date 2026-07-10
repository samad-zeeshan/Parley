"""The deterministic dialogue policy: slot state in, next action out. No model involved."""

import pytest

from dialogue.policy import DialogueState, decide
from dialogue.schema import NLUResult

SLOT_A = {"slot_id": "s-p001-100116", "starts_at": "2026-10-01T16:00"}
SLOT_B = {"slot_id": "s-p002-100117", "starts_at": "2026-10-01T17:00"}


def nlu(intent, **slots):
    choice = slots.pop("choice", None)
    return NLUResult(intent=intent, slots=slots, choice=choice)


def state(**kw):
    s = DialogueState()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_greeting_with_nothing_known_asks_for_area():
    s = DialogueState()
    s.merge(nlu("greet"))
    assert decide(s).name == "ask_slot" and decide(s).args["slot"] == "area"


@pytest.mark.parametrize("known,missing", [
    ({}, "area"),
    ({"area": "Dubai Marina"}, "bedrooms"),
    ({"area": "Dubai Marina", "bedrooms": 2}, "date"),
])
def test_asks_required_slots_in_fixed_order(known, missing):
    s = DialogueState()
    s.merge(nlu("book_viewing", **known))
    a = decide(s)
    assert a.name == "ask_slot" and a.args["slot"] == missing


def test_searches_once_required_slots_are_known():
    s = DialogueState()
    s.merge(nlu("book_viewing", area="Dubai Marina", bedrooms=2, date="2026-10-01"))
    a = decide(s)
    assert a.name == "search"
    assert a.args == {"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"}


def test_search_carries_budget_and_window():
    s = DialogueState()
    s.merge(nlu("book_viewing", area="Dubai Marina", bedrooms=2, date="2026-10-01",
                budget=150000, time_window=["16:00", "18:00"]))
    assert decide(s).args == {"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01",
                              "max_rent": 150000, "start": "16:00", "end": "18:00"}


def test_offers_results_after_search():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"})
    s.record_search([SLOT_A, SLOT_B])
    a = decide(s)
    assert a.name == "offer" and a.args["slots"] == [SLOT_A, SLOT_B]


def test_no_results_asks_to_change_something():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"})
    s.record_search([])
    assert decide(s).name == "no_results"


def test_changing_a_slot_after_search_searches_again():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"})
    s.record_search([SLOT_A])
    s.merge(nlu("provide_details", date="2026-10-02"))
    assert decide(s).name == "search"


def test_choosing_an_option_holds_that_slot():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"})
    s.record_search([SLOT_A, SLOT_B])
    s.merge(nlu("choose_option", choice=2))
    a = decide(s)
    assert a.name == "hold" and a.args == {"slot_id": SLOT_B["slot_id"]}


def test_choice_out_of_range_reoffers():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"})
    s.record_search([SLOT_A])
    s.merge(nlu("choose_option", choice=3))
    assert decide(s).name == "offer"


def test_yes_to_a_single_offer_holds_it():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"})
    s.record_search([SLOT_A])
    s.merge(nlu("confirm"))
    assert decide(s).name == "hold"


def test_yes_to_several_offers_asks_which():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"})
    s.record_search([SLOT_A, SLOT_B])
    s.merge(nlu("confirm"))
    assert decide(s).name == "offer"


def test_asks_phone_after_hold():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"})
    s.record_search([SLOT_A])
    s.record_hold({"hold_id": "h-0123456789ab", "slot_id": SLOT_A["slot_id"]}, SLOT_A)
    a = decide(s)
    assert a.name == "ask_slot" and a.args["slot"] == "phone"


def test_asks_confirmation_when_hold_and_phone_are_known():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01", "phone": "0501234567"})
    s.record_search([SLOT_A])
    s.record_hold({"hold_id": "h-0123456789ab", "slot_id": SLOT_A["slot_id"]}, SLOT_A)
    a = decide(s)
    assert a.name == "ask_confirm" and a.args["slot"] == SLOT_A


def test_confirms_only_after_an_explicit_yes_to_ask_confirm():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01", "phone": "0501234567"})
    s.record_search([SLOT_A])
    s.record_hold({"hold_id": "h-0123456789ab", "slot_id": SLOT_A["slot_id"]}, SLOT_A)
    s.record_spoken(decide(s))          # agent asked "shall I confirm?"
    s.merge(nlu("confirm"))
    a = decide(s)
    assert a.name == "confirm" and a.args == {"hold_id": "h-0123456789ab", "phone": "0501234567"}


def test_no_at_ask_confirm_releases_and_reoffers():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01", "phone": "0501234567"})
    s.record_search([SLOT_A, SLOT_B])
    s.record_hold({"hold_id": "h-0123456789ab", "slot_id": SLOT_A["slot_id"]}, SLOT_A)
    s.record_spoken(decide(s))
    s.merge(nlu("deny"))
    assert decide(s).name == "release"


def test_done_after_booking():
    s = state(slots={"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01", "phone": "0501234567"})
    s.record_booking({"booking_id": "b-0123456789ab", "slot_id": SLOT_A["slot_id"]}, SLOT_A)
    assert decide(s).name == "confirmed"


def test_cancel_after_booking():
    s = state(slots={"area": "Dubai Marina"})
    s.record_booking({"booking_id": "b-0123456789ab", "slot_id": SLOT_A["slot_id"]}, SLOT_A)
    s.merge(nlu("cancel_booking"))
    a = decide(s)
    assert a.name == "cancel" and a.args == {"booking_id": "b-0123456789ab"}


def test_goodbye_ends():
    s = DialogueState()
    s.merge(nlu("goodbye"))
    assert decide(s).name == "goodbye"


def test_out_of_scope_gets_a_redirect_not_a_guess():
    s = state(slots={"area": "Dubai Marina"})
    s.merge(nlu("out_of_scope"))
    assert decide(s).name == "redirect"
