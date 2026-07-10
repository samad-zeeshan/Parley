"""Check that a reply states no fact the booking API did not return.

Touchstone's rule, applied to speech: the oracle (api/service.py) owns every
slot, price and address. A reply may repeat those facts and what the caller
said; any other date, time, number, phone, area, tower or id is a violation.
The check runs on the reply after the same normalization the ASR output gets,
so "Friday", "الجمعة", "14:30" and "مية ألف" are all caught.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from speech.normalize import normalize

_SLOT_ID = re.compile(r"\bs-p\d{3}-\d{6}\b")
_OTHER_ID = re.compile(r"\b[bh]-[0-9a-f]{12}\b")
_TOWER_EN = re.compile(r"\btower\s+([a-z])\s*(\d+)", re.I)
_TOWER_AR = re.compile(r"برج\s+(\S)(\d+)")


@dataclass(frozen=True)
class Violation:
    kind: str
    value: object


@dataclass
class Facts:
    dates: set = field(default_factory=set)
    times: set = field(default_factory=set)
    numbers: set = field(default_factory=set)
    phones: set = field(default_factory=set)
    areas: set = field(default_factory=set)
    towers: set = field(default_factory=set)
    ids: set = field(default_factory=set)

    @classmethod
    def from_slots(cls, slots: list[dict], caller: dict | None = None, ids: list[str] = ()) -> "Facts":
        f = cls()
        for i, s in enumerate(slots, start=1):
            f.numbers.add(i)  # "option 2"
            f.dates.add(s["starts_at"][:10])
            f.times.update({s["starts_at"][11:16], s["ends_at"][11:16]})
            f.numbers.update({s["annual_rent_aed"], s["bedrooms"]})
            for addr in (s["address"], s.get("address_ar", "")):
                f.numbers.update(int(d) for d in re.findall(r"\d+", addr))
                f.towers.update(_towers(addr))
            f.areas.add(s["area"])
            f.ids.add(s["slot_id"])
        f.ids.update(ids)
        if caller:
            f.add_caller(caller)
        return f

    def add_caller(self, slots: dict) -> None:
        """Repeating what the caller said is not inventing a fact."""
        if "area" in slots:
            self.areas.add(slots["area"])
        if "date" in slots:
            self.dates.add(slots["date"])
        if "time_window" in slots:
            self.times.update(slots["time_window"])
        for k in ("bedrooms", "budget"):
            if k in slots:
                self.numbers.add(slots[k])
        if "phone" in slots:
            self.phones.add(slots["phone"])


def _towers(text: str) -> set:
    out = {(a.lower(), int(b)) for a, b in _TOWER_EN.findall(text)}
    out |= {(a, int(b)) for a, b in _TOWER_AR.findall(text)}
    return out


def _areas_in(text: str) -> set[str]:
    from .nlu import AREA_ALIASES

    found = set()
    padded = f" {text} "
    for alias, area in AREA_ALIASES.items():
        if f" {alias} " in padded:
            found.add(area)
    return found


def ungrounded(reply: str, facts: Facts, today: date) -> list[Violation]:
    out: list[Violation] = []
    for sid in _SLOT_ID.findall(reply):
        if sid not in facts.ids:
            out.append(Violation("slot_id", sid))
    for oid in _OTHER_ID.findall(reply):
        if oid not in facts.ids:
            out.append(Violation("id", oid))
    for t in _towers(reply):
        if t not in facts.towers:
            out.append(Violation("tower", t))
    stripped = _OTHER_ID.sub(" ", _SLOT_ID.sub(" ", reply))
    norm = normalize(stripped, today)
    for e in norm.entities:
        if e.kind == "date" and e.value not in facts.dates:
            out.append(Violation("date", e.value))
        elif e.kind == "time" and e.value not in facts.times:
            out.append(Violation("time", e.value))
        elif e.kind == "time_window" and not set(e.value) <= facts.times:
            out.append(Violation("time", e.value))
        elif e.kind == "phone" and e.value not in facts.phones:
            out.append(Violation("phone", e.value))
        elif e.kind == "number" and e.value not in facts.numbers:
            out.append(Violation("number", e.value))
    for area in _areas_in(norm.text) - facts.areas:
        out.append(Violation("area", area))
    return out
