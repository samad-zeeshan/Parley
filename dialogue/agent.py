"""One call session: transcript in, reply out.

    transcript -> language/dialect ID -> normalizer -> NLU (schema-checked)
      -> policy -> validated tool calls -> policy ... -> phrasing (grounding-checked)

The agent holds no facts of its own. Slots it offers come from list_slots in
this session; holds and bookings come from the API's answers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from api.service import BookingError, HoldExpired, SlotUnavailable
from speech.langid import identify, reply_language
from speech.normalize import Normalized, normalize

from .grounding import Facts, Violation, ungrounded
from .nlu import RuleNLU
from .phrasing import Phraser, action_slots
from .policy import Action, DialogueState, decide
from .schema import NLUResult
from .tools import ToolExecutor, ToolRejected

MAX_STEPS = 6


@dataclass
class TurnResult:
    text: str
    lang: str
    action: str
    dialect: str
    nlu: NLUResult
    normalized: Normalized
    source: str = "template"
    rejected: list[Violation] = field(default_factory=list)   # LLM reply violations, if any
    ungrounded: list[Violation] = field(default_factory=list)  # violations in the spoken text: must be empty
    tools: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


class Agent:
    def __init__(self, service, conn, clock, session_id: str, nlu=None, phraser: Phraser | None = None,
                 metrics=None):
        self.svc = service
        self.clock = clock
        self.session_id = session_id
        self.nlu = nlu or RuleNLU()
        self.phraser = phraser or Phraser(clock=clock)
        self.metrics = metrics
        self.tools = ToolExecutor(service, conn, session_id, metrics=metrics)
        self.state = DialogueState()
        self.lang = "en"

    def turn(self, text: str) -> TurnResult:
        t0 = time.perf_counter()
        today = self.clock.now().date()
        lid = identify(text)
        if lid.ar_tokens + lid.en_tokens > 0:
            self.lang = reply_language(lid)   # reply in the language the caller used last
        norm = normalize(text, today)
        t1 = time.perf_counter()
        if isinstance(self.nlu, RuleNLU):
            nlu = self.nlu.parse(text, today, norm)
        else:
            nlu = self.nlu.parse(text, today, norm, context=self.state.last_asked or "")
        t2 = time.perf_counter()
        self.state.merge(nlu)

        called: list[str] = []
        action = decide(self.state)
        note = None
        for _ in range(MAX_STEPS):
            if not action.is_tool:
                break
            called.append(action.name)
            try:
                note = self._execute(action)
            except ToolRejected:
                action = Action("error")
                break
            action = decide(self.state)
            if note and action.name == "offer":
                action = Action("offer", {**action.args, "note": note})
        t3 = time.perf_counter()

        rendered = self.phraser.render(action, self.lang, caller=self.state.slots)
        self.state.record_spoken(action)
        facts = Facts.from_slots(action_slots(action), caller=self.state.slots)
        final_violations = ungrounded(rendered.text, facts, today)
        t4 = time.perf_counter()

        if self.metrics:
            self.metrics.turns.labels(lid.label, action.name).inc()
            if rendered.rejected:
                self.metrics.grounding_rejections.inc()
            for stage, secs in (("nlu", t2 - t1), ("policy_tools", t3 - t2), ("phrasing", t4 - t3)):
                self.metrics.stage_seconds.labels(stage).observe(secs)

        return TurnResult(
            text=rendered.text, lang=self.lang, action=action.name, dialect=lid.label, nlu=nlu, normalized=norm,
            source=rendered.source, rejected=rendered.rejected, ungrounded=final_violations, tools=called,
            timings={"normalize": t1 - t0, "nlu": t2 - t1, "policy_tools": t3 - t2, "phrasing": t4 - t3},
        )

    def _execute(self, action: Action) -> str | None:
        s = self.state
        if action.name == "search":
            s.record_search(self.tools.call("list_slots", action.args))
        elif action.name == "hold":
            slot = next(x for x in s.offered if x["slot_id"] == action.args["slot_id"])
            try:
                s.record_hold(self.tools.call("hold_slot", action.args), slot)
            except SlotUnavailable:
                s.offered = [x for x in s.offered if x["slot_id"] != slot["slot_id"]]
                s.rejected_slot_ids.add(slot["slot_id"])
                s.intent = None
                return "taken"
        elif action.name == "confirm":
            try:
                booking = self.tools.call("confirm_booking", action.args)
            except HoldExpired:
                s.record_release()
                return "taken"
            s.record_booking(booking, s.held_slot)
        elif action.name == "release":
            try:
                self.tools.call("release_hold", action.args)
            except BookingError:
                pass
            s.record_release()
        elif action.name == "cancel":
            self.tools.call("cancel_booking", action.args)
            s.record_cancel()
        return None
