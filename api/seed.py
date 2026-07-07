"""Synthetic UAE-style listings. Nothing here is a real listing, building, agent or phone number.

Area names are real districts so that a caller can say them; everything inside an
area (tower letters, unit numbers, agents, rents) is generated from a fixed seed.

Run: uv run python -m api.seed majlis.db [YYYY-MM-DD]
"""

from __future__ import annotations

import random
import sqlite3
import sys
from datetime import date, datetime, timedelta

from .clock import DUBAI
from .db import connect

# (emirate, area, area_ar, base annual rent for a one-bedroom in AED)
AREAS = [
    ("Dubai", "Dubai Marina", "دبي مارينا", 110_000),
    ("Dubai", "Jumeirah Lake Towers", "أبراج بحيرات جميرا", 85_000),
    ("Dubai", "Downtown Dubai", "وسط مدينة دبي", 140_000),
    ("Dubai", "Business Bay", "الخليج التجاري", 100_000),
    ("Dubai", "Al Barsha", "البرشاء", 75_000),
    ("Dubai", "Jumeirah Village Circle", "قرية جميرا الدائرية", 65_000),
    ("Abu Dhabi", "Al Reem Island", "جزيرة الريم", 90_000),
    ("Abu Dhabi", "Khalifa City", "مدينة خليفة", 70_000),
    ("Abu Dhabi", "Al Raha Beach", "شاطئ الراحة", 105_000),
    ("Abu Dhabi", "Saadiyat Island", "جزيرة السعديات", 150_000),
]

# Rent multiplier by bedroom count (0 is a studio).
BEDROOM_FACTOR = {0: 0.7, 1: 1.0, 2: 1.45, 3: 1.95, 4: 2.6}

AGENTS = [
    ("agent-1", "Aisha", "عائشة"),
    ("agent-2", "Omar", "عمر"),
    ("agent-3", "Fatima", "فاطمة"),
    ("agent-4", "Rashid", "راشد"),
    ("agent-5", "Layla", "ليلى"),
    ("agent-6", "Khalid", "خالد"),
]

TOWER_LETTERS = "ABCDEFGH"
TOWER_LETTERS_AR = "أبتثجحخد"

DAYS = 14
FIRST_HOUR, LAST_HOUR = 10, 19


def _round_to(value: float, step: int = 5_000) -> int:
    return int(round(value / step) * step)


def seed(conn: sqlite3.Connection, anchor: date, rng_seed: int = 7) -> dict:
    rng = random.Random(rng_seed)
    for i, (aid, name, name_ar) in enumerate(AGENTS):
        conn.execute("insert into agents values (?,?,?,?)", (aid, name, name_ar, f"050000{i + 1:04d}"))

    n_props = n_slots = 0
    for a_idx, (emirate, area, area_ar, base) in enumerate(AREAS):
        bedroom_mix = [1, 2, 3, rng.choice([0, 2, 4])]
        for p_idx, beds in enumerate(bedroom_mix):
            pid = f"p{a_idx:02d}{p_idx}"
            t = rng.randrange(len(TOWER_LETTERS))
            tower_no = rng.randint(1, 9)
            unit = rng.randint(2, 45) * 100 + rng.randint(1, 12)
            address = f"Unit {unit}, Tower {TOWER_LETTERS[t]}{tower_no}, {area}, {emirate}"
            emirate_ar = "دبي" if emirate == "Dubai" else "أبوظبي"
            address_ar = f"شقة {unit}، برج {TOWER_LETTERS_AR[t]}{tower_no}، {area_ar}، {emirate_ar}"
            rent = _round_to(base * BEDROOM_FACTOR[beds] * rng.uniform(0.9, 1.15))
            agent = AGENTS[(a_idx + p_idx) % len(AGENTS)][0]
            conn.execute(
                "insert into properties values (?,?,?,?,?,?,?,?,?,?)",
                (pid, f"MJ-{a_idx:02d}{p_idx}", emirate, area, area_ar, address, address_ar, beds, rent, agent),
            )
            n_props += 1
            for d in range(DAYS):
                day = anchor + timedelta(days=d)
                hours = rng.sample(range(FIRST_HOUR, LAST_HOUR + 1), k=rng.randint(2, 4))
                for h in sorted(hours):
                    start = datetime(day.year, day.month, day.day, h, 0)
                    end = start + timedelta(minutes=30)
                    sid = f"s-{pid}-{start:%m%d%H}"
                    conn.execute(
                        "insert into viewing_slots(id, property_id, starts_at, ends_at) values (?,?,?,?)",
                        (sid, pid, start.strftime("%Y-%m-%dT%H:%M"), end.strftime("%Y-%m-%dT%H:%M")),
                    )
                    n_slots += 1
    return {"agents": len(AGENTS), "properties": n_props, "slots": n_slots}


def area_names() -> dict[str, str]:
    """English area name to Arabic area name."""
    return {a: ar for _, a, ar, _ in AREAS}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "majlis.db"
    anchor = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else datetime.now(DUBAI).date()
    print(seed(connect(path), anchor))
