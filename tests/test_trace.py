"""TRACE scoring: completion, wrong actions, recovery and user effort per arm and language."""

from eval.trace import aggregate_arm, had_trouble, plot_svg


def turn(intent="provide_details", gold="provide_details", slots=None, gold_slots=None, action="ask_slot",
         args=None):
    return {"intent": intent, "gold_intent": gold, "slots": slots or {}, "gold_slots": gold_slots or {},
            "action": action, "action_args": args or {}}


def call(cid, completed, turns, wrong=()):
    return {"call_id": cid, "kind": cid.split("-")[0], "completed": completed, "turns": turns,
            "wrong_actions": list(wrong)}


CLEAN = [call("en-a", True, [turn()] * 6), call("en-b", True, [turn()] * 5)]


def test_trouble_means_a_misheard_turn_or_a_re_ask():
    assert not had_trouble(CLEAN[0])
    assert had_trouble(call("en-x", True, [turn(intent="unclear")]))
    assert had_trouble(call("en-x", True, [turn(slots={"date": "2026-10-03"}, gold_slots={"date": "2026-10-04"})]))
    assert had_trouble(call("en-x", True, [turn(args={"reason": "not_understood"})]))


def test_aggregate_counts_recovery_and_extra_turns():
    stressed = [call("en-a", True, [turn(intent="unclear")] + [turn()] * 7),
                call("en-b", False, [turn(intent="unclear")] * 16, wrong=[{"kind": "hold", "why": "date"}])]
    row = aggregate_arm(stressed, CLEAN)["en"]
    assert row["completed"] == 1 and row["calls"] == 2
    assert row["wrong_actions"] == 1
    assert row["recovered"] == 1 and row["troubled"] == 2
    assert row["turns_per_completed_call"] == 8.0
    assert row["extra_turns_vs_clean"] == 2.0


def test_plot_is_one_svg_with_a_panel_per_language():
    rows = {"clean": {"en": {"completed": 2, "calls": 2}, "gulf": {"completed": 1, "calls": 2},
                      "msa": {"completed": 0, "calls": 2}, "switch": {"completed": 2, "calls": 2}},
            "white@5": {"en": {"completed": 1, "calls": 2}, "gulf": {"completed": 0, "calls": 2},
                        "msa": {"completed": 0, "calls": 2}, "switch": {"completed": 0, "calls": 2}}}
    svg = plot_svg(rows)
    assert svg.startswith("<svg") and svg.count("<g class=\"panel\"") == 4
    assert "white@5" in svg and "Gulf Arabic" in svg
