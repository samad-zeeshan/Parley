"""Hash-chained audit log in SQLite, after Change-Gate's audit.py.

Each row commits to the previous row's hash. Editing any stored payload changes
its recomputed hash and verify_chain returns False from that row on.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

GENESIS_HASH = "0" * 64


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compute_entry_hash(prev_hash: str, seq: int, ts: str, action: str, payload_json: str) -> str:
    blob = f"{prev_hash}\n{seq}\n{ts}\n{action}\n{payload_json}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def append(conn: sqlite3.Connection, ts: str, action: str, payload: dict) -> dict:
    row = conn.execute("select seq, entry_hash from audit order by seq desc limit 1").fetchone()
    seq, prev = (row[0] + 1, row[1]) if row else (1, GENESIS_HASH)
    body = _canonical(payload)
    h = compute_entry_hash(prev, seq, ts, action, body)
    conn.execute("insert into audit values (?,?,?,?,?,?)", (seq, ts, action, body, prev, h))
    return {"seq": seq, "action": action, "entry_hash": h}


def entries(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("select seq, ts, action, payload, prev_hash, entry_hash from audit order by seq")
    return [
        {"seq": s, "ts": ts, "action": a, "payload": json.loads(p), "prev_hash": ph, "entry_hash": eh}
        for s, ts, a, p, ph, eh in rows
    ]


def verify_chain(conn: sqlite3.Connection) -> bool:
    prev = GENESIS_HASH
    expected_seq = 1
    for seq, ts, action, payload, prev_hash, entry_hash in conn.execute(
        "select seq, ts, action, payload, prev_hash, entry_hash from audit order by seq"
    ):
        if seq != expected_seq or prev_hash != prev:
            return False
        if compute_entry_hash(prev_hash, seq, ts, action, payload) != entry_hash:
            return False
        prev, expected_seq = entry_hash, seq + 1
    return True
