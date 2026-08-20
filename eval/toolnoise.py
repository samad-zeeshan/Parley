"""Tool noise and outage, after PredAct-Bench (arXiv 2608.02372) and safe recovery (arXiv 2606.31307).

    uv run python -m eval.toolnoise        # the forty calls on reference text under each backend fault
"""

from __future__ import annotations

import random
import time

from api.service import ServiceUnavailable

from .harness import CALL_KINDS, reference_source, run_call, update_results
from .scripts import CALLS

MODES = ["clean", "slow", "empty", "contradictory", "contradictory-confirm", "down", "down-at-confirm"]
SLOW_TIMEOUT_SHARE = 0.5
SAFE_ACTIONS = {"unavailable", "no_results"}


class NoisyService:
    """Wraps the booking service and corrupts its answers. The database underneath stays the truth."""

    def __init__(self, svc, mode: str, rng: random.Random):
        self.svc, self.mode, self.rng = svc, mode, rng

    def __getattr__(self, name):
        return getattr(self.svc, name)

    def _gate(self, tool: str) -> None:
        if self.mode == "down" or (self.mode == "down-at-confirm" and tool == "confirm"):
            raise ServiceUnavailable(f"{tool}: backend down")

    def _late(self, out):
        # A slow backend does the work and then the answer arrives after the agent's budget.
        if self.mode == "slow" and self.rng.random() < SLOW_TIMEOUT_SHARE:
            raise TimeoutError("no answer within the tool budget")
        return out

    def list_slots(self, **kw):
        self._gate("list_slots")
        if self.mode == "empty":
            return []
        out = self.svc.list_slots(**kw)
        if self.mode == "contradictory":
            # Same slot ids, but a different day and area from the ones asked for.
            out = [{**s, "starts_at": _shift_day(s["starts_at"]), "area": "Saadiyat Island"} for s in out]
        return self._late(out)

    def hold(self, slot_id, caller_id):
        self._gate("hold")
        return self._late(self.svc.hold(slot_id, caller_id))

    def confirm(self, hold_id, phone, idempotency_key):
        self._gate("confirm")
        out = self.svc.confirm(hold_id, phone, idempotency_key=idempotency_key)
        if self.mode == "contradictory-confirm":
            out = {**out, "slot_id": out["slot_id"][:-2] + "99"}
        return self._late(out)

    def release(self, hold_id):
        self._gate("release")
        return self.svc.release(hold_id)

    def cancel(self, booking_id):
        self._gate("cancel")
        return self.svc.cancel(booking_id)


def _shift_day(iso: str) -> str:
    from datetime import datetime, timedelta

    return (datetime.fromisoformat(iso) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M")


_TRUTH = ("starts_at", "area", "address", "address_ar", "bedrooms", "annual_rent_aed")


def _db_slot(conn, slot_id: str) -> dict | None:
    row = conn.execute("""select s.starts_at, p.area, p.address, p.address_ar, p.bedrooms, p.annual_rent_aed
                          from viewing_slots s join properties p on p.id = s.property_id where s.id = ?""",
                       (slot_id,)).fetchone()
    return dict(zip(_TRUTH, row)) if row else None


def false_facts(action: str, state, conn) -> list[tuple[str, str]]:
    """Slots the reply states that the database contradicts: an offer, a hold or a booking that is not so."""
    out = []
    if action == "offer":
        for s in state.offered or []:
            db = _db_slot(conn, s["slot_id"])
            if db is None or any(db[k] != s[k] for k in _TRUTH):
                out.append(("offer", s["slot_id"]))
    elif action == "ask_confirm" and state.held_slot:
        sid = state.held_slot["slot_id"]
        active = conn.execute("select 1 from holds where slot_id=? and status='active'", (sid,)).fetchone()
        if not active:
            out.append(("hold", sid))
    elif action == "confirmed" and state.booked_slot:
        sid = state.booked_slot["slot_id"]
        booked = conn.execute("select 1 from bookings where slot_id=? and status='confirmed'", (sid,)).fetchone()
        if not booked:
            out.append(("booking", sid))
    return out


def run_mode(mode: str) -> dict:
    calls = []
    for i, (cid, call) in enumerate(CALLS.items()):
        rng = random.Random(f"{mode}/{cid}")
        wrap = None if mode == "clean" else (lambda svc, rng=rng: NoisyService(svc, mode, rng))
        calls.append(run_call(cid, call, reference_source, service=wrap,
                              observe=lambda agent, t: {"false_facts": false_facts(t.action, agent.state,
                                                                                    agent.tools.conn)}))
    failed = [c for c in calls if any(t["action"] in SAFE_ACTIONS for t in c["turns"])]
    safe = [c for c in failed if not any(t["false_facts"] for t in c["turns"]) and not c["wrong_actions"]]
    return {
        "calls": len(calls), "completed": sum(c["completed"] for c in calls),
        "completed_by_language": {k: sum(c["completed"] for c in calls if c["kind"] == k) for k in CALL_KINDS},
        "false_facts_spoken": sum(len(t["false_facts"]) for c in calls for t in c["turns"]),
        "spoken_replies_with_ungrounded_facts": sum(c["spoken_ungrounded"] for c in calls),
        "wrong_actions": sum(len(c["wrong_actions"]) for c in calls),
        "calls_hitting_a_fault": len(failed), "safe_recoveries": len(safe),
        "callbacks_queued": sum(c["callbacks"] for c in calls),
        "bookings_in_database": sum(len(c["bookings"]) for c in calls),
    }


def main() -> dict:
    out = {}
    for mode in MODES:
        t0 = time.time()
        out[mode] = run_mode(mode)
        r = out[mode]
        print(f"{mode}: completed {r['completed']}/40, false facts {r['false_facts_spoken']}, fault calls "
              f"{r['calls_hitting_a_fault']}, safe {r['safe_recoveries']}, callbacks {r['callbacks_queued']} "
              f"({time.time() - t0:.1f} s)", flush=True)
    update_results("tool_noise", {"modes": out, "slow_timeout_share": SLOW_TIMEOUT_SHARE,
                                  "transcripts": "reference text, so the backend is the only source of noise"})
    return out


if __name__ == "__main__":
    main()
