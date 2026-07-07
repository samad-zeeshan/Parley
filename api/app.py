"""HTTP surface of the booking API, plus /healthz and /metrics.

Run: uv run uvicorn api.app:app --port 8000
"""

from __future__ import annotations

import os
import sqlite3

from fastapi import FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from . import audit
from .clock import SystemClock
from .db import connect
from .metrics import Metrics
from .seed import seed
from .service import BookingError, BookingService, Conflict, HoldExpired, NotFound, SlotUnavailable

_STATUS = {NotFound: 404, SlotUnavailable: 409, Conflict: 409, HoldExpired: 410}


class HoldIn(BaseModel):
    slot_id: str
    caller_id: str


class ConfirmIn(BaseModel):
    hold_id: str
    phone: str


def open_default_db() -> sqlite3.Connection:
    path = os.environ.get("MAJLIS_DB", "majlis.db")
    conn = connect(path)
    if conn.execute("select count(*) from properties").fetchone()[0] == 0:
        from datetime import datetime

        from .clock import DUBAI

        seed(conn, anchor=datetime.now(DUBAI).date())
    return conn


def create_app(conn: sqlite3.Connection | None = None, clock=None) -> FastAPI:
    conn = conn if conn is not None else open_default_db()
    clock = clock or SystemClock()
    svc = BookingService(conn, clock)
    metrics = Metrics()

    app = FastAPI(title="Majlis booking API", version="0.1.0")
    app.state.service = svc
    app.state.metrics = metrics
    app.state.clock = clock

    def fail(action: str, err: Exception):
        metrics.booking_actions.labels(action, getattr(err, "code", "invalid")).inc()
        if isinstance(err, ValueError):
            raise HTTPException(422, str(err))
        raise HTTPException(_STATUS.get(type(err), 400), {"code": err.code, "detail": str(err)})

    @app.get("/healthz")
    def healthz():
        conn.execute("select 1")
        return {"ok": True}

    @app.get("/metrics")
    def prom():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/slots")
    def list_slots(area: str | None = None, bedrooms: int | None = None, max_rent: int | None = None,
                   date: str | None = None, start: str | None = None, end: str | None = None,
                   limit: int = 5):
        window = (start or "00:00", end or "24:00") if (start or end) else None
        slots = svc.list_slots(area=area, bedrooms=bedrooms, max_rent=max_rent, date=date,
                               window=window, limit=min(limit, 100))
        metrics.booking_actions.labels("list", "ok").inc()
        return {"slots": slots}

    @app.post("/holds", status_code=201)
    def hold(body: HoldIn):
        try:
            out = svc.hold(body.slot_id, body.caller_id)
        except BookingError as e:
            fail("hold", e)
        metrics.booking_actions.labels("hold", "ok").inc()
        return out

    @app.post("/bookings", status_code=201)
    def confirm(body: ConfirmIn, response: Response, idempotency_key: str | None = Header(default=None)):
        if not idempotency_key:
            raise HTTPException(400, "Idempotency-Key header is required")
        try:
            out = svc.confirm(body.hold_id, body.phone, idempotency_key)
        except (BookingError, ValueError) as e:
            fail("confirm", e)
        if out.get("replayed"):
            response.status_code = 200
        metrics.booking_actions.labels("confirm", "replay" if out.get("replayed") else "ok").inc()
        return out

    @app.get("/bookings/{booking_id}")
    def get_booking(booking_id: str):
        b = svc.get_booking(booking_id)
        if b is None:
            raise HTTPException(404, "booking not found")
        return b

    @app.post("/bookings/{booking_id}/cancel")
    def cancel(booking_id: str):
        try:
            out = svc.cancel(booking_id)
        except BookingError as e:
            fail("cancel", e)
        metrics.booking_actions.labels("cancel", "ok").inc()
        return out

    @app.get("/audit/verify")
    def verify():
        n = conn.execute("select count(*) from audit").fetchone()[0]
        return {"ok": audit.verify_chain(conn), "entries": n}

    return app


def __getattr__(name: str):
    # `uvicorn api.app:app` builds the default app lazily so importing this module in tests
    # does not create a database file.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
