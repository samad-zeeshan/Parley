"""Scripted calls through the whole dialogue layer (text in, text out).

The same scripts are played as synthetic audio in eval/; here they run on the
transcripts so the suite stays hermetic.
"""

import pytest

from api.audit import entries, verify_chain
from dialogue.agent import Agent
from eval.scripts import CALLS


@pytest.mark.parametrize("call_id", ["en-booking", "gulf-booking", "switch-booking"])
def test_scripted_call_ends_in_a_booking_with_valid_audit(call_id, svc, conn, clock):
    call = CALLS[call_id]
    agent = Agent(svc, conn, clock, session_id=call_id)
    replies = [agent.turn(t["text"]) for t in call["turns"]]
    assert replies[-1].action == "confirmed", [r.action for r in replies]
    rows = conn.execute("select status, phone from bookings").fetchall()
    assert rows == [("confirmed", call["phone"])]
    assert verify_chain(conn)
    actions = [e["action"] for e in entries(conn)]
    assert actions.count("confirm") == 1 and "hold" in actions


def test_reply_language_follows_the_last_utterance(svc, conn, clock):
    agent = Agent(svc, conn, clock, session_id="s")
    assert agent.turn("I want to book a viewing").lang == "en"
    assert agent.turn("في دبي مارينا").lang == "ar"
    assert agent.turn("two bedrooms please").lang == "en"


def test_every_reply_in_the_scripts_is_grounded(svc, conn, clock):
    for call_id, call in CALLS.items():
        agent = Agent(svc, conn, clock, session_id=call_id)
        for t in call["turns"]:
            r = agent.turn(t["text"])
            assert r.ungrounded == [], (call_id, r.text)


def test_cancel_after_booking(svc, conn, clock):
    call = CALLS["en-booking"]
    agent = Agent(svc, conn, clock, session_id="c")
    for t in call["turns"]:
        agent.turn(t["text"])
    r = agent.turn("actually, cancel my booking")
    assert r.action == "cancelled"
    assert conn.execute("select status from bookings").fetchone()[0] == "cancelled"
    assert verify_chain(conn)
