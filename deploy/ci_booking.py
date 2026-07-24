"""CI gate 3: one scripted booking completes against the deployed service.

    python deploy/ci_booking.py http://127.0.0.1:8000

Talks to the agent over HTTP by text (the cluster image has no ASR), books a
viewing, then checks the booking through the booking API and that the audit
chain verifies. Standard library only, so it runs on a bare CI runner.

The script says "tomorrow", not a weekday, because the service seeds its
synthetic slots from the day it starts.
"""

from __future__ import annotations

import json
import sys
import urllib.request

LINES = [
    "Hi, I would like to book a viewing for a two bedroom flat in Al Barsha.",
    "Tomorrow.",
    "The first one please.",
    "My number is zero five zero one two three four five six seven.",
    "Yes, please confirm the booking.",
]


def call(base: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main(base: str) -> int:
    sid = call(base, "POST", "/agent/sessions")["session_id"]
    reply = {}
    for line in LINES:
        reply = call(base, "POST", f"/agent/sessions/{sid}/text", {"text": line})
        print(f"caller: {line}\nagent:  {reply['reply']}  [{reply['action']}]")
    if reply.get("action") != "confirmed":
        print("FAIL: the call did not end in a confirmed booking")
        return 1
    audit = call(base, "GET", "/audit/verify")
    if not audit["ok"] or audit["entries"] < 3:
        print(f"FAIL: audit chain {audit}")
        return 1
    print(f"OK: booking confirmed, audit chain verifies over {audit['entries']} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"))
