"""A caller turn split across two messages: the agent waits for the rest instead of answering half of it."""

import pytest

from dialogue.agent import Agent, open_ended

HEAD = ["Hi, a two bedroom flat in Dubai Marina.", "Thursday between four and six in the afternoon.",
        "The first one please."]


@pytest.mark.parametrize("text,expected", [
    ("My number is zero five zero", True),
    ("سجل رقمي صفر خمسة خمسة تسعة", True),
    ("Thursday between four and", True),
    ("يوم السبت بين الساعة ثلاثة و", True),
    ("My number is zero five zero one two three four five six seven.", False),
    ("The first one please.", False),
    ("Two bedrooms.", False),
    ("", False),
])
def test_open_ended(text, expected):
    assert open_ended(text) is expected


def test_split_phone_is_joined(svc, conn, clock):
    agent = Agent(svc, conn, clock, session_id="split")
    for t in HEAD:
        agent.turn(t)
    first = agent.turn("My number is zero five zero")
    assert first.action == "listen" and first.text == "Go on."
    second = agent.turn("one two three four five six seven.")
    assert agent.state.slots["phone"] == "0501234567"
    assert second.action == "ask_confirm"


def test_split_time_window_is_joined(svc, conn, clock):
    agent = Agent(svc, conn, clock, session_id="split-time")
    agent.turn("Hi, a two bedroom flat in Dubai Marina.")
    assert agent.turn("Thursday between four and").action == "listen"
    agent.turn("six in the afternoon.")
    assert agent.state.slots["time_window"] == ["16:00", "18:00"]


def test_arabic_wait_reply_is_in_arabic(svc, conn, clock):
    agent = Agent(svc, conn, clock, session_id="split-ar")
    agent.turn("أبي شقة غرفتين في البرشاء")
    r = agent.turn("رقمي صفر خمسة")
    assert r.action == "listen" and r.lang == "ar" and r.ungrounded == []


def test_a_caller_who_never_finishes_is_not_held_forever(svc, conn, clock):
    agent = Agent(svc, conn, clock, session_id="split-stuck")
    agent.turn("Hi, a two bedroom flat in Dubai Marina.")
    actions = [agent.turn("zero five zero").action for _ in range(3)]
    assert actions[:2] == ["listen", "listen"] and actions[2] != "listen"
