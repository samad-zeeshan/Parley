"""Tool noise and outage: the agent recovers safely and never states a slot the database does not hold."""

import random

import pytest

from api.audit import entries
from api.service import ServiceUnavailable
from dialogue.agent import Agent
from eval.toolnoise import MODES, NoisyService, false_facts

OPENING = ["Hi, a two bedroom flat in Dubai Marina.", "Thursday between four and six in the afternoon."]


class Flaky:
    """A booking service whose chosen tool fails the first `fails` times."""

    def __init__(self, svc, tool, fails):
        self.svc, self.tool, self.fails = svc, tool, fails

    def __getattr__(self, name):
        attr = getattr(self.svc, name)
        if name != self.tool:
            return attr

        def wrapped(*a, **kw):
            if self.fails > 0:
                self.fails -= 1
                raise ServiceUnavailable("down")
            return attr(*a, **kw)
        return wrapped


def test_list_slots_down_gives_a_safe_reply(svc, conn, clock):
    agent = Agent(Flaky(svc, "list_slots", 99), conn, clock, session_id="down")
    agent.turn(OPENING[0])
    r = agent.turn(OPENING[1])
    assert r.action == "unavailable"
    assert r.ungrounded == [] and "16:00" not in r.text


def test_one_transient_failure_is_retried(svc, conn, clock):
    agent = Agent(Flaky(svc, "list_slots", 1), conn, clock, session_id="flaky")
    agent.turn(OPENING[0])
    assert agent.turn(OPENING[1]).action == "offer"


def test_callback_is_recorded_when_the_phone_is_known(svc, conn, clock):
    agent = Agent(Flaky(svc, "confirm_booking", 0), conn, clock, session_id="cb")
    for t in OPENING + ["The first one please.", "My number is zero five zero one two three four five six seven."]:
        agent.turn(t)
    agent.svc = Flaky(svc, "confirm", 99)
    agent.tools.svc = agent.svc
    r = agent.turn("Yes, confirm it.")
    assert r.action == "unavailable" and "0501234567" in r.text
    assert "booked" not in r.text.lower()
    assert [e["action"] for e in entries(conn)].count("callback_requested") == 1
    assert conn.execute("select count(*) from bookings").fetchone()[0] == 0


def test_contradictory_slots_are_not_offered(svc, conn, clock):
    agent = Agent(NoisyService(svc, "contradictory", random.Random(1)), conn, clock, session_id="contra")
    agent.turn(OPENING[0])
    r = agent.turn(OPENING[1])
    assert r.action == "unavailable"
    assert false_facts(r.action, agent.state, conn) == []


def test_contradictory_confirm_is_not_announced_as_booked(svc, conn, clock):
    noisy = NoisyService(svc, "contradictory-confirm", random.Random(1))
    agent = Agent(noisy, conn, clock, session_id="contra2")
    for t in OPENING + ["The first one please.", "My number is zero five zero one two three four five six seven."]:
        agent.turn(t)
    r = agent.turn("Yes, confirm it.")
    assert r.action == "unavailable"
    assert false_facts(r.action, agent.state, conn) == []


def test_empty_results_say_so(svc, conn, clock):
    agent = Agent(NoisyService(svc, "empty", random.Random(1)), conn, clock, session_id="empty")
    agent.turn(OPENING[0])
    assert agent.turn(OPENING[1]).action == "no_results"


def test_down_service_raises_on_every_tool(svc):
    noisy = NoisyService(svc, "down", random.Random(0))
    with pytest.raises(ServiceUnavailable):
        noisy.list_slots(area="Dubai Marina")


def test_slow_service_can_time_out_after_doing_the_work(svc):
    noisy = NoisyService(svc, "slow", random.Random(3))
    outcomes = []
    for _ in range(20):
        try:
            noisy.list_slots(area="Dubai Marina")
            outcomes.append("ok")
        except TimeoutError:
            outcomes.append("timeout")
    assert "ok" in outcomes and "timeout" in outcomes


def test_false_facts_catches_an_offered_slot_that_differs_from_the_database(svc, conn, clock):
    agent = Agent(svc, conn, clock, session_id="truth")
    agent.turn(OPENING[0])
    agent.turn(OPENING[1])
    assert false_facts("offer", agent.state, conn) == []
    agent.state.offered[0] = {**agent.state.offered[0], "starts_at": "2026-10-09T16:00"}
    assert false_facts("offer", agent.state, conn) == [("offer", agent.state.offered[0]["slot_id"])]


def test_modes_cover_the_protocol():
    assert {"clean", "slow", "empty", "contradictory", "down"} <= set(MODES)
