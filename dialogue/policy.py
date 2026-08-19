"""Deterministic dialogue policy. Slot state in, next action out.

The model never chooses what happens next. It only fills slots (NLU) and, at
most, rewords the reply this policy picked (phrasing). Tool actions (search,
hold, confirm, release, cancel) are executed by the agent through the validated
tool layer, and the policy is asked again with the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import NLUResult

REQUIRED = ("area", "bedrooms", "date")
MAX_OFFERS = 3


@dataclass(frozen=True)
class Action:
    name: str
    args: dict = field(default_factory=dict)

    @property
    def is_tool(self) -> bool:
        return self.name in TOOL_ACTIONS


TOOL_ACTIONS = {"search", "hold", "confirm", "release", "cancel"}


@dataclass
class DialogueState:
    slots: dict = field(default_factory=dict)
    intent: str | None = None
    choice: int | None = None
    offered: list[dict] | None = None      # None: not searched for the current slots
    searched_with: dict | None = None
    rejected_slot_ids: set = field(default_factory=set)
    hold: dict | None = None
    held_slot: dict | None = None
    booking: dict | None = None
    booked_slot: dict | None = None
    cancelled: bool = False
    last_spoken: str | None = None
    last_asked: str | None = None
    tool_error: str | None = None

    # ---- updates ---------------------------------------------------------

    def merge(self, nlu: NLUResult) -> None:
        self.intent = nlu.intent
        self.choice = nlu.choice
        for k, v in nlu.slots.items():
            if v is not None:
                self.slots[k] = v
        self.tool_error = None

    def record_search(self, slots: list[dict]) -> None:
        fresh = [s for s in slots if s["slot_id"] not in self.rejected_slot_ids]
        self.offered = fresh[:MAX_OFFERS]
        self.searched_with = self.search_args()

    def record_hold(self, hold: dict, slot: dict) -> None:
        self.hold, self.held_slot = hold, slot
        self.choice = None
        self.intent = None

    def record_release(self) -> None:
        if self.held_slot:
            self.rejected_slot_ids.add(self.held_slot["slot_id"])
        self.hold = self.held_slot = None
        self.offered = None  # search again without the rejected slot
        self.intent = None

    def record_booking(self, booking: dict, slot: dict) -> None:
        self.booking, self.booked_slot = booking, slot
        self.hold = self.held_slot = None
        self.intent = None

    def record_cancel(self) -> None:
        self.cancelled = True
        self.intent = None

    def record_spoken(self, action: Action) -> None:
        self.last_spoken = action.name
        self.last_asked = action.args.get("slot") if action.name == "ask_slot" else action.name

    # ---- derived ---------------------------------------------------------

    def search_args(self) -> dict:
        args = {k: self.slots[k] for k in REQUIRED if k in self.slots}
        if "budget" in self.slots:
            args["max_rent"] = self.slots["budget"]
        if "time_window" in self.slots:
            args["start"], args["end"] = self.slots["time_window"]
        return args


def decide(s: DialogueState) -> Action:
    if s.intent == "goodbye":
        return Action("goodbye")
    if s.tool_error:
        return Action("unavailable", {"reason": s.tool_error, "phone": s.slots.get("phone")})

    if s.booking:
        if s.cancelled:
            return Action("cancelled", {"slot": s.booked_slot})
        if s.intent == "cancel_booking":
            return Action("cancel", {"booking_id": s.booking["booking_id"]})
        return Action("confirmed", {"slot": s.booked_slot, "booking_id": s.booking["booking_id"]})

    if s.intent == "out_of_scope":
        return Action("redirect", {"next": _next_question(s)})
    if s.intent == "unclear" and s.last_spoken == "ask_slot" and s.last_asked:
        return Action("ask_slot", {"slot": s.last_asked, "reason": "not_understood"})

    if s.hold:
        if s.last_spoken == "ask_confirm" and s.intent == "confirm" and "phone" in s.slots:
            return Action("confirm", {"hold_id": s.hold["hold_id"], "phone": s.slots["phone"]})
        if s.last_spoken == "ask_confirm" and s.intent == "deny":
            return Action("release", {"hold_id": s.hold["hold_id"]})
        if "phone" not in s.slots:
            return Action("ask_slot", {"slot": "phone"})
        return Action("ask_confirm", {"slot": s.held_slot, "phone": s.slots["phone"]})

    missing = [k for k in REQUIRED if k not in s.slots]
    if missing:
        return Action("ask_slot", {"slot": missing[0]})

    if s.offered is None or s.searched_with != s.search_args():
        return Action("search", s.search_args())

    if not s.offered:
        return Action("no_results", {"criteria": s.search_args()})

    if s.intent == "choose_option" and s.choice:
        if 1 <= s.choice <= len(s.offered):
            return Action("hold", {"slot_id": s.offered[s.choice - 1]["slot_id"]})
        return Action("offer", {"slots": s.offered, "note": "choice_out_of_range"})
    if s.intent == "confirm" and len(s.offered) == 1:
        return Action("hold", {"slot_id": s.offered[0]["slot_id"]})
    if s.intent == "deny" and s.last_spoken == "offer":
        return Action("ask_slot", {"slot": "date", "reason": "none_suitable"})
    return Action("offer", {"slots": s.offered})


def _next_question(s: DialogueState) -> str:
    missing = [k for k in REQUIRED if k not in s.slots]
    return missing[0] if missing else "choice"
