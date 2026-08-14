"""The evaluation harness: the scripted caller's next line, wrong-action scoring, and a text-only call."""

from eval.harness import gold_constraints, next_line, reference_source, run_call, wrong_actions
from eval.scripts import CALLS
from eval.scripts.v1 import T

CALL = {"phone": "0501234567", "turns": [
    T("Hello.", "en", "greet"),
    T("I want to view a flat in Dubai Marina.", "en", "book_viewing", {"area": "Dubai Marina"}),
    {**T("Is there parking?", "en", "out_of_scope"), "volunteer": True},
    T("Two bedrooms.", "en", "provide_details", {"bedrooms": 2}),
    T("Saturday morning.", "en", "provide_details", {"date": "2026-10-03", "time_window": ["09:00", "12:00"]}),
    T("The second one.", "en", "choose_option", choice=2),
    T("zero five zero one two three four five six seven", "en", "provide_details", {"phone": "0501234567"}),
    T("Yes.", "en", "confirm"),
]}


def test_volunteered_line_comes_right_after_the_line_before_it():
    assert next_line(CALL, "ask_slot", {"slot": "bedrooms"}, 1, [0, 1]) == 2


def test_asked_slot_picks_the_line_that_carries_it():
    assert next_line(CALL, "ask_slot", {"slot": "bedrooms"}, 2, [0, 1, 2]) == 3
    assert next_line(CALL, "ask_slot", {"slot": "date"}, 3, [0, 1, 2, 3]) == 4


def test_asked_again_repeats_the_same_line():
    assert next_line(CALL, "ask_slot", {"slot": "date"}, 4, [0, 1, 2, 3, 4]) == 4


def test_offer_confirm_and_end():
    assert next_line(CALL, "offer", {}, 4, [0, 1, 2, 3, 4]) == 5
    assert next_line(CALL, "ask_confirm", {}, 6, [0, 1, 2, 3, 4, 5, 6]) == 7
    assert next_line(CALL, "confirmed", {}, 7, list(range(8))) is None


def test_redirect_answers_the_question_it_carries():
    assert next_line(CALL, "redirect", {"next": "bedrooms"}, 2, [0, 1, 2]) == 3


SLOT = {"slot_id": "s-p001-100310", "area": "Dubai Marina", "bedrooms": 2, "starts_at": "2026-10-03T10:00",
        "annual_rent_aed": 150000}
OTHER = {**SLOT, "slot_id": "s-p001-100311", "starts_at": "2026-10-03T11:00"}


def test_gold_constraints_collect_every_line():
    g = gold_constraints(CALL)
    assert g == {"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-03", "time_window": ["09:00", "12:00"],
                 "phone": "0501234567", "choice": 2}


def test_right_hold_and_confirm_are_not_wrong():
    events = [{"kind": "hold", "slot": OTHER, "offered": [SLOT["slot_id"], OTHER["slot_id"]], "consistent": True},
              {"kind": "confirm", "slot": OTHER, "phone": "0501234567"}]
    assert wrong_actions(CALL, events) == []


def test_wrong_index_wrong_area_and_wrong_phone_are_counted():
    events = [{"kind": "hold", "slot": SLOT, "offered": [SLOT["slot_id"], OTHER["slot_id"]], "consistent": True},
              {"kind": "hold", "slot": {**OTHER, "area": "Al Barsha"}, "offered": [], "consistent": False},
              {"kind": "confirm", "slot": OTHER, "phone": "0509999999"}]
    assert [w["why"] for w in wrong_actions(CALL, events)] == ["not the option chosen", "area", "phone"]


def test_v1_english_call_completes_on_reference_text():
    out = run_call("en-booking", CALLS["en-booking"], reference_source)
    assert out["completed"] and out["wrong_actions"] == []
    assert out["spoken_ungrounded"] == 0
