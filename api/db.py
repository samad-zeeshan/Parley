"""SQLite schema for properties, agents, viewing slots, holds, bookings and the audit chain.

Times are stored as Dubai local ISO strings (YYYY-MM-DDTHH:MM), which sort correctly
as text and match what a caller says ("Thursday at 16:00").
"""

from __future__ import annotations

import sqlite3

SCHEMA = """
create table if not exists agents (
    id          text primary key,
    name        text not null,
    name_ar     text not null,
    phone       text not null
);

create table if not exists properties (
    id               text primary key,
    ref              text not null unique,
    emirate          text not null check (emirate in ('Dubai', 'Abu Dhabi')),
    area             text not null,
    area_ar          text not null,
    address          text not null,
    address_ar       text not null,
    bedrooms         integer not null check (bedrooms between 0 and 5),
    annual_rent_aed  integer not null check (annual_rent_aed > 0),
    agent_id         text not null references agents(id)
);

create table if not exists viewing_slots (
    id           text primary key,
    property_id  text not null references properties(id),
    starts_at    text not null,
    ends_at      text not null,
    status       text not null default 'open' check (status in ('open', 'booked'))
);
create index if not exists viewing_slots_start on viewing_slots(starts_at);

create table if not exists holds (
    id          text primary key,
    slot_id     text not null references viewing_slots(id),
    caller_id   text not null,
    created_at  text not null,
    expires_at  text not null,
    status      text not null check (status in ('active', 'confirmed', 'released'))
);

create table if not exists bookings (
    id               text primary key,
    slot_id          text not null references viewing_slots(id),
    hold_id          text not null references holds(id),
    phone            text not null,
    idempotency_key  text not null unique,
    status           text not null check (status in ('confirmed', 'cancelled')),
    created_at       text not null
);

create table if not exists audit (
    seq         integer primary key,
    ts          text not null,
    action      text not null,
    payload     text not null,
    prev_hash   text not null,
    entry_hash  text not null
);
"""


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("pragma foreign_keys = on")
    conn.executescript(SCHEMA)
    return conn
