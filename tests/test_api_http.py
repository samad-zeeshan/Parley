"""The HTTP surface of the booking API."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.service import HOLD_TTL


@pytest.fixture
def client(conn, clock):
    return TestClient(create_app(conn=conn, clock=clock))


def _hold(client):
    slot = client.get("/slots", params={"limit": 1}).json()["slots"][0]
    r = client.post("/holds", json={"slot_id": slot["slot_id"], "caller_id": "c1"})
    assert r.status_code == 201
    return slot, r.json()


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_list_slots_with_filters(client):
    r = client.get("/slots", params={"area": "Dubai Marina", "bedrooms": 2, "max_rent": 200000})
    assert r.status_code == 200
    assert all(s["area"] == "Dubai Marina" for s in r.json()["slots"])


def test_list_slots_time_window(client):
    r = client.get("/slots", params={"date": "2026-10-01", "start": "16:00", "end": "18:00"})
    assert all("16:00" <= s["starts_at"][11:16] < "18:00" for s in r.json()["slots"])


def test_hold_conflict_is_409(client):
    slot, _ = _hold(client)
    r = client.post("/holds", json={"slot_id": slot["slot_id"], "caller_id": "c2"})
    assert r.status_code == 409


def test_hold_unknown_is_404(client):
    assert client.post("/holds", json={"slot_id": "x", "caller_id": "c"}).status_code == 404


def test_confirm_requires_idempotency_key(client):
    _, hold = _hold(client)
    r = client.post("/bookings", json={"hold_id": hold["hold_id"], "phone": "0501234567"})
    assert r.status_code == 400


def test_confirm_then_replay_returns_same_booking(client):
    _, hold = _hold(client)
    body = {"hold_id": hold["hold_id"], "phone": "0501234567"}
    r1 = client.post("/bookings", json=body, headers={"Idempotency-Key": "abc"})
    r2 = client.post("/bookings", json=body, headers={"Idempotency-Key": "abc"})
    assert r1.status_code == 201 and r2.status_code == 200
    assert r1.json()["booking_id"] == r2.json()["booking_id"]


def test_confirm_expired_hold_is_410(client, clock):
    _, hold = _hold(client)
    clock.advance(HOLD_TTL + timedelta(seconds=1))
    r = client.post("/bookings", json={"hold_id": hold["hold_id"], "phone": "0501234567"},
                    headers={"Idempotency-Key": "k"})
    assert r.status_code == 410


def test_confirm_bad_phone_is_422(client):
    _, hold = _hold(client)
    r = client.post("/bookings", json={"hold_id": hold["hold_id"], "phone": "1"},
                    headers={"Idempotency-Key": "k"})
    assert r.status_code == 422


def test_cancel_and_audit_verify(client):
    _, hold = _hold(client)
    b = client.post("/bookings", json={"hold_id": hold["hold_id"], "phone": "0501234567"},
                    headers={"Idempotency-Key": "k"}).json()
    assert client.post(f"/bookings/{b['booking_id']}/cancel").json()["status"] == "cancelled"
    v = client.get("/audit/verify").json()
    assert v == {"ok": True, "entries": 3}


def test_metrics_exposes_booking_counters(client):
    _hold(client)
    text = client.get("/metrics").text
    assert "parley_booking_actions_total" in text
