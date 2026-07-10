"""The strict JSON contracts: NLU output and every tool call's arguments."""

from __future__ import annotations

from dataclasses import dataclass, field

from api.seed import AREAS

INTENTS = [
    "book_viewing",     # wants a viewing (may carry slots)
    "provide_details",  # answers a question: budget, date, phone...
    "choose_option",    # picks one of the offered slots
    "confirm",          # yes
    "deny",             # no
    "cancel_booking",
    "repeat",
    "greet",
    "goodbye",
    "out_of_scope",
]

AREA_NAMES = [a for _, a, _, _ in AREAS]

_DATE = r"^\d{4}-\d{2}-\d{2}$"
_HHMM = r"^([01]\d|2[0-4]):[0-5]\d$"
_PHONE = r"^0(5\d{8}|[234679]\d{7})$"

SLOT_PROPERTIES = {
    "area": {"enum": AREA_NAMES},
    "bedrooms": {"type": "integer", "minimum": 0, "maximum": 5},
    "budget": {"type": "integer", "minimum": 10_000, "maximum": 5_000_000},
    "date": {"type": "string", "pattern": _DATE},
    "time_window": {"type": "array", "items": {"type": "string", "pattern": _HHMM}, "minItems": 2, "maxItems": 2},
    "phone": {"type": "string", "pattern": _PHONE},
}

NLU_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "slots"],
    "properties": {
        "intent": {"enum": INTENTS},
        "slots": {"type": "object", "additionalProperties": False, "properties": SLOT_PROPERTIES},
        "choice": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
    },
}

TOOL_SCHEMAS = {
    "list_slots": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "area": {"enum": AREA_NAMES},
            "bedrooms": SLOT_PROPERTIES["bedrooms"],
            "max_rent": SLOT_PROPERTIES["budget"],
            "date": SLOT_PROPERTIES["date"],
            "start": {"type": "string", "pattern": _HHMM},
            "end": {"type": "string", "pattern": _HHMM},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    },
    "hold_slot": {
        "type": "object",
        "additionalProperties": False,
        "required": ["slot_id"],
        "properties": {"slot_id": {"type": "string", "pattern": r"^s-p\d{3}-\d{6}$"}},
    },
    "release_hold": {
        "type": "object",
        "additionalProperties": False,
        "required": ["hold_id"],
        "properties": {"hold_id": {"type": "string", "pattern": r"^h-[0-9a-f]{12}$"}},
    },
    "confirm_booking": {
        "type": "object",
        "additionalProperties": False,
        "required": ["hold_id", "phone"],
        "properties": {
            "hold_id": {"type": "string", "pattern": r"^h-[0-9a-f]{12}$"},
            "phone": SLOT_PROPERTIES["phone"],
        },
    },
    "cancel_booking": {
        "type": "object",
        "additionalProperties": False,
        "required": ["booking_id"],
        "properties": {"booking_id": {"type": "string", "pattern": r"^b-[0-9a-f]{12}$"}},
    },
}


@dataclass
class NLUResult:
    intent: str
    slots: dict = field(default_factory=dict)
    choice: int | None = None
    source: str = "rules"
    dropped: dict = field(default_factory=dict)  # model slots with no support in the utterance

    def to_json(self) -> dict:
        out = {"intent": self.intent, "slots": dict(self.slots)}
        if self.choice is not None:
            out["choice"] = self.choice
        return out
