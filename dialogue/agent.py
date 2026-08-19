"""One call session: transcript in, reply out.

    transcript -> language/dialect ID -> normalizer -> NLU (schema-checked)
      -> policy -> validated tool calls -> policy ... -> phrasing (grounding-checked)

The agent holds no facts of its own. Slots it offers come from list_slots in
this session; holds and bookings come from the API's answers.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date

from api.service import BookingError, HoldExpired, SlotUnavailable
from speech.langid import identify, reply_language
from speech.normalize import Normalized, _norm_word, normalize

from .grounding import Facts, Violation, ungrounded
from .nlu import RuleNLU
from .phrasing import Phraser, action_slots, template
from .policy import Action, DialogueState, decide
from .schema import NLUResult
from .tools import ToolExecutor, ToolFailed, ToolRejected

MAX_STEPS = 6
MAX_CARRY = 2

_DIGIT_WORDS = {_norm_word(w) for w in [
    "zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "صفر", "واحد", "اثنين", "ثنين", "ثلاثة", "ثلاث", "أربعة", "اربع", "خمسة", "خمس", "ستة", "ست", "سبعة", "سبع",
    "ثمانية", "ثمان", "تسعة", "تسع"]}
_CONNECTORS = {_norm_word(w) for w in ["and", "between", "to", "from", "or", "و", "بين", "إلى", "الى", "من", "أو"]}


def open_ended(text: str, today: date | None = None) -> bool:
    """The message stops mid-thought: on a connector, or inside a run of spoken digits with no full phone yet.

    MTVA (arXiv 2609.20152) splits caller turns across messages; answering the first half on its own
    loses phone numbers and time windows.
    """
    words = [_norm_word(w) for w in re.findall(r"[^\W_]+", text)]
    if not words:
        return False
    if words[-1] in _CONNECTORS:
        return True
    run = 0
    for w in reversed(words):
        if w in _DIGIT_WORDS or (w.isdigit() and len(w) == 1):
            run += 1
        else:
            break
    if run < 2:
        return False
    return not any(e.kind == "phone" for e in normalize(text, today or date(2026, 1, 1)).entities)


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
    action_args: dict = field(default_factory=dict)     # scalar arguments only: slot, next, reason, note
    rejected_text: str | None = None


class Agent:
    def __init__(self, service, conn, clock, session_id: str, nlu=None, phraser: Phraser | None = None,
                 metrics=None, carry: bool = True):
        self.svc = service
        self.clock = clock
        self.session_id = session_id
        self.nlu = nlu or RuleNLU()
        self.phraser = phraser or Phraser(clock=clock)
        self.metrics = metrics
        self.tools = ToolExecutor(service, conn, session_id, metrics=metrics)
        self.state = DialogueState()
        self.lang = "en"
        self.pending: str | None = None
        self.carry = carry
        self.carried = 0
        self.callback_requested = False

    def turn(self, text: str) -> TurnResult:
        t0 = time.perf_counter()
        today = self.clock.now().date()
        if self.pending:
            text, self.pending = f"{self.pending} {text}", None
        lid = identify(text)
        if lid.ar_tokens + lid.en_tokens > 0:
            self.lang = reply_language(lid)   # reply in the language the caller used last
        norm = normalize(text, today)
        if self.carry and self.carried < MAX_CARRY and open_ended(text, today):
            self.carried += 1
            self.pending = text
            reply = template(Action("listen"), self.lang)
            return TurnResult(text=reply, lang=self.lang, action="listen", dialect=lid.label,
                              nlu=NLUResult("unclear", source="carry"), normalized=norm,
                              ungrounded=ungrounded(reply, Facts(), today))
        self.carried = 0
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
            except ToolFailed as e:
                # Safe recovery after arXiv 2606.31307: say the system failed, promise nothing the
                # database has not confirmed, and queue a callback when the phone number is known.
                self.state.tool_error = e.reason
                action = decide(self.state)
                if self.state.slots.get("phone") and not self.callback_requested:
                    self.tools.request_callback(self.state.slots["phone"], f"{e.tool}: {e.reason}")
                    self.callback_requested = True
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
            action_args={k: v for k, v in action.args.items() if isinstance(v, (str, int))},
            rejected_text=rendered.rejected_text,
        )

    def _execute(self, action: Action) -> str | None:
        s = self.state
        if action.name == "search":
            slots = self.tools.call("list_slots", action.args)
            fitting = [x for x in slots if _fits(x, action.args)]
            if slots and not fitting:
                raise ToolFailed("list_slots", "mismatch")
            s.record_search(fitting)
        elif action.name == "hold":
            slot = next(x for x in s.offered if x["slot_id"] == action.args["slot_id"])
            try:
                hold = self.tools.call("hold_slot", action.args)
                if hold.get("slot_id") != slot["slot_id"]:
                    raise ToolFailed("hold_slot", "mismatch")
                s.record_hold(hold, slot)
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
            if booking.get("slot_id") != s.held_slot["slot_id"]:
                raise ToolFailed("confirm_booking", "mismatch")
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


def _fits(slot: dict, query: dict) -> bool:
    """A slot the API returned for this query must match it. One that does not is never read out."""
    if "area" in query and slot["area"].lower() != query["area"].lower():
        return False
    if "bedrooms" in query and slot["bedrooms"] != query["bedrooms"]:
        return False
    if "date" in query and slot["starts_at"][:10] != query["date"]:
        return False
    if "max_rent" in query and slot["annual_rent_aed"] > query["max_rent"]:
        return False
    if "start" in query and not (query["start"] <= slot["starts_at"][11:16] < query["end"]):
        return False
    return True
