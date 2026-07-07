"""Every booking rule, tested against the synthetic seed."""

from datetime import timedelta

import pytest

from api.audit import verify_chain
from api.service import (
    HOLD_TTL,
    Conflict,
    HoldExpired,
    NotFound,
    SlotUnavailable,
)


def first_open(svc, **kw):
    slots = svc.list_slots(**kw)
    assert slots, f"seed has no open slot for {kw}"
    return slots[0]


# ---- seed ----------------------------------------------------------------

def test_seed_covers_both_emirates_with_aed_prices(conn):
    emirates = {r[0] for r in conn.execute("select distinct emirate from properties")}
    assert emirates == {"Dubai", "Abu Dhabi"}
    prices = [r[0] for r in conn.execute("select annual_rent_aed from properties")]
    assert all(isinstance(p, int) and 30_000 <= p <= 1_000_000 for p in prices)


def test_seed_is_deterministic():
    from api.db import connect
    from api.seed import seed
    from tests.conftest import ANCHOR

    rows = []
    for _ in range(2):
        c = connect(":memory:")
        seed(c, anchor=ANCHOR.date(), rng_seed=7)
        rows.append(c.execute("select * from viewing_slots order by id").fetchall())
    assert rows[0] == rows[1]


def test_slot_times_are_24_hour_on_the_hour_within_office_hours(conn):
    for (starts_at,) in conn.execute("select starts_at from viewing_slots"):
        hh, mm = starts_at[11:13], starts_at[14:16]
        assert mm == "00" and 9 <= int(hh) <= 19


# ---- list ----------------------------------------------------------------

def test_list_filters_by_area_bedrooms_and_budget(svc):
    slots = svc.list_slots(area="Dubai Marina", bedrooms=2, max_rent=200_000)
    assert slots
    for s in slots:
        assert s["area"] == "Dubai Marina"
        assert s["bedrooms"] == 2
        assert s["annual_rent_aed"] <= 200_000


def test_list_filters_by_date_and_time_window(svc):
    slots = svc.list_slots(date="2026-10-01", window=("16:00", "18:00"))
    assert slots
    for s in slots:
        assert s["starts_at"].startswith("2026-10-01T")
        assert "16:00" <= s["starts_at"][11:16] < "18:00"


def test_list_hides_slots_in_the_past(svc, clock):
    for s in svc.list_slots(date="2026-10-01"):
        assert s["starts_at"] >= clock.now().strftime("%Y-%m-%dT%H:%M")


def test_list_unknown_area_is_empty_not_error(svc):
    assert svc.list_slots(area="Atlantis Under The Sea") == []


def test_list_returns_address_and_agent_from_the_database(svc):
    s = first_open(svc)
    assert s["address"] and s["agent_name"] and s["slot_id"]


# ---- hold ----------------------------------------------------------------

def test_hold_removes_slot_from_listing(svc):
    s = first_open(svc)
    svc.hold(s["slot_id"], caller_id="c1")
    assert s["slot_id"] not in {x["slot_id"] for x in svc.list_slots(limit=1000)}


def test_second_hold_on_same_slot_is_rejected(svc):
    s = first_open(svc)
    svc.hold(s["slot_id"], caller_id="c1")
    with pytest.raises(SlotUnavailable):
        svc.hold(s["slot_id"], caller_id="c2")


def test_hold_unknown_slot(svc):
    with pytest.raises(NotFound):
        svc.hold("slot-does-not-exist", caller_id="c1")


def test_hold_expires_and_slot_reopens(svc, clock):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    assert h["expires_at"] == (clock.now() + HOLD_TTL).isoformat(timespec="seconds")
    clock.advance(HOLD_TTL + timedelta(seconds=1))
    assert s["slot_id"] in {x["slot_id"] for x in svc.list_slots(limit=1000)}
    svc.hold(s["slot_id"], caller_id="c2")  # someone else can take it now


def test_release_reopens_the_slot_and_blocks_confirm(svc):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    svc.release(h["hold_id"])
    assert s["slot_id"] in {x["slot_id"] for x in svc.list_slots(limit=1000)}
    with pytest.raises(HoldExpired):
        svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")


def test_release_of_a_confirmed_hold_conflicts(svc):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    with pytest.raises(Conflict):
        svc.release(h["hold_id"])


# ---- confirm -------------------------------------------------------------

def test_confirm_books_the_slot(svc, conn):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    b = svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    assert b["status"] == "confirmed" and b["slot_id"] == s["slot_id"]
    status = conn.execute("select status from viewing_slots where id=?", (s["slot_id"],)).fetchone()[0]
    assert status == "booked"


def test_confirm_is_idempotent_on_key(svc, conn):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    b1 = svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    b2 = svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    assert b1["booking_id"] == b2["booking_id"]
    assert conn.execute("select count(*) from bookings").fetchone()[0] == 1


def test_idempotent_replay_survives_hold_expiry(svc, clock):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    b1 = svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    clock.advance(HOLD_TTL * 3)
    again = svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    assert again["booking_id"] == b1["booking_id"]


def test_key_reuse_for_a_different_hold_conflicts(svc):
    a, b = svc.list_slots(limit=2)
    ha = svc.hold(a["slot_id"], caller_id="c1")
    hb = svc.hold(b["slot_id"], caller_id="c1")
    svc.confirm(ha["hold_id"], phone="0501234567", idempotency_key="k1")
    with pytest.raises(Conflict):
        svc.confirm(hb["hold_id"], phone="0501234567", idempotency_key="k1")


def test_confirm_after_expiry_fails(svc, clock):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    clock.advance(HOLD_TTL + timedelta(seconds=1))
    with pytest.raises(HoldExpired):
        svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")


def test_confirm_rejects_bad_phone(svc):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    with pytest.raises(ValueError):
        svc.confirm(h["hold_id"], phone="12", idempotency_key="k1")


def test_confirm_unknown_hold(svc):
    with pytest.raises(NotFound):
        svc.confirm("hold-nope", phone="0501234567", idempotency_key="k1")


# ---- cancel --------------------------------------------------------------

def test_cancel_frees_the_slot(svc):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    b = svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    out = svc.cancel(b["booking_id"])
    assert out["status"] == "cancelled"
    assert s["slot_id"] in {x["slot_id"] for x in svc.list_slots(limit=1000)}


def test_cancel_twice_is_harmless(svc):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    b = svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    svc.cancel(b["booking_id"])
    assert svc.cancel(b["booking_id"])["status"] == "cancelled"


def test_cancel_unknown_booking(svc):
    with pytest.raises(NotFound):
        svc.cancel("booking-nope")


# ---- audit ---------------------------------------------------------------

def test_every_booking_action_is_audited_and_chain_verifies(svc, conn):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    b = svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    svc.cancel(b["booking_id"])
    actions = [r[0] for r in conn.execute("select action from audit order by seq")]
    assert actions == ["hold", "confirm", "confirm_replay", "cancel"]
    assert verify_chain(conn)


def test_rejected_actions_are_audited_too(svc, conn):
    s = first_open(svc)
    svc.hold(s["slot_id"], caller_id="c1")
    with pytest.raises(SlotUnavailable):
        svc.hold(s["slot_id"], caller_id="c2")
    actions = [r[0] for r in conn.execute("select action from audit order by seq")]
    assert actions == ["hold", "hold_rejected"]


def test_tampering_breaks_the_chain(svc, conn):
    s = first_open(svc)
    h = svc.hold(s["slot_id"], caller_id="c1")
    svc.confirm(h["hold_id"], phone="0501234567", idempotency_key="k1")
    conn.execute(
        "update audit set payload = replace(payload, '0501234567', '0509999999') where action='confirm'"
    )
    assert not verify_chain(conn)
