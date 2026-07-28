"""Every tool call is validated against its JSON schema, and against the session's
facts, before it reaches the booking API."""

import pytest

from api.audit import entries, verify_chain
from dialogue.tools import ToolExecutor, ToolRejected


@pytest.fixture
def tools(svc, conn):
    return ToolExecutor(svc, conn, session_id="sess-1")


def test_valid_search_runs(tools):
    out = tools.call("list_slots", {"area": "Dubai Marina", "bedrooms": 2, "date": "2026-10-01"})
    assert isinstance(out, list) and out
    assert all(s["area"] == "Dubai Marina" for s in out)


@pytest.mark.parametrize("args", [
    {"area": "Atlantis", "bedrooms": 2},              # not an area we list
    {"area": "Dubai Marina", "bedrooms": "two"},      # wrong type
    {"area": "Dubai Marina", "bedrooms": 9},          # out of range
    {"area": "Dubai Marina", "date": "Thursday"},     # not ISO
    {"area": "Dubai Marina", "start": "4pm"},         # not HH:MM
    {"area": "Dubai Marina", "sql": "drop table"},    # unknown field
])
def test_bad_search_arguments_are_rejected(tools, args):
    with pytest.raises(ToolRejected):
        tools.call("list_slots", args)


def test_unknown_tool_is_rejected(tools):
    with pytest.raises(ToolRejected):
        tools.call("transfer_money", {})


def test_hold_on_a_slot_the_api_never_offered_is_rejected(tools, svc):
    # A well-formed id that exists in the database but was never returned to this session.
    real = svc.list_slots(limit=1)[0]["slot_id"]
    with pytest.raises(ToolRejected, match="not offered"):
        tools.call("hold_slot", {"slot_id": real})


def test_hold_on_an_offered_slot_runs(tools):
    offered = tools.call("list_slots", {"area": "Al Barsha"})
    hold = tools.call("hold_slot", {"slot_id": offered[0]["slot_id"]})
    assert hold["slot_id"] == offered[0]["slot_id"]


def test_confirm_needs_a_hold_this_session_made(tools):
    with pytest.raises(ToolRejected, match="hold"):
        tools.call("confirm_booking", {"hold_id": "h-0123456789ab", "phone": "0501234567"})


def test_confirm_rejects_malformed_phone_before_the_api(tools):
    offered = tools.call("list_slots", {"area": "Al Barsha"})
    hold = tools.call("hold_slot", {"slot_id": offered[0]["slot_id"]})
    with pytest.raises(ToolRejected):
        tools.call("confirm_booking", {"hold_id": hold["hold_id"], "phone": "call me"})


def test_confirm_is_idempotent_within_a_session(tools, conn):
    offered = tools.call("list_slots", {"area": "Al Barsha"})
    hold = tools.call("hold_slot", {"slot_id": offered[0]["slot_id"]})
    b1 = tools.call("confirm_booking", {"hold_id": hold["hold_id"], "phone": "0501234567"})
    b2 = tools.call("confirm_booking", {"hold_id": hold["hold_id"], "phone": "0501234567"})
    assert b1["booking_id"] == b2["booking_id"]
    assert conn.execute("select count(*) from bookings").fetchone()[0] == 1


def test_rejections_and_calls_are_in_the_audit_chain(tools, conn):
    with pytest.raises(ToolRejected):
        tools.call("hold_slot", {"slot_id": "s-p000-100110"})
    offered = tools.call("list_slots", {"area": "Al Barsha"})
    tools.call("hold_slot", {"slot_id": offered[0]["slot_id"]})
    actions = [e["action"] for e in entries(conn)]
    assert actions[0] == "tool_rejected"
    assert "hold" in actions
    assert all(e["payload"].get("session_id") == "sess-1" for e in entries(conn) if e["action"].startswith("tool"))
    assert verify_chain(conn)


def test_agent_booking_actions_reach_the_metrics(svc, conn):
    from api.metrics import Metrics

    m = Metrics()
    tools = ToolExecutor(svc, conn, session_id="m", metrics=m)
    offered = tools.call("list_slots", {"area": "Al Barsha"})
    hold = tools.call("hold_slot", {"slot_id": offered[0]["slot_id"]})
    tools.call("confirm_booking", {"hold_id": hold["hold_id"], "phone": "0501234567"})
    tools.call("confirm_booking", {"hold_id": hold["hold_id"], "phone": "0501234567"})
    get = m.registry.get_sample_value
    assert get("parley_booking_actions_total", {"action": "confirm", "outcome": "ok"}) == 1
    assert get("parley_booking_actions_total", {"action": "confirm", "outcome": "replay"}) == 1
