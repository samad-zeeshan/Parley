"""Booking rules. This module is the oracle: every slot, price and address the agent
may say comes out of list_slots, hold, confirm or cancel, and nowhere else."""

from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta

from . import audit

HOLD_TTL = timedelta(minutes=5)

_PHONE = re.compile(r"^0(5\d{8}|[234679]\d{7})$")


class BookingError(Exception):
    code = "booking_error"


class NotFound(BookingError):
    code = "not_found"


class SlotUnavailable(BookingError):
    code = "slot_unavailable"


class HoldExpired(BookingError):
    code = "hold_expired"


class Conflict(BookingError):
    code = "conflict"


class ServiceUnavailable(BookingError):
    """The booking backend did not answer. Raised by the noisy wrapper in the evaluation, and by real outages."""

    code = "unavailable"


def normalize_phone(phone: str) -> str:
    """Accept 05XXXXXXXX, +9715XXXXXXXX or 9715XXXXXXXX; return the 0-prefixed form."""
    digits = re.sub(r"\D", "", phone or "")
    if digits.startswith("00971"):
        digits = "0" + digits[5:]
    elif digits.startswith("971"):
        digits = "0" + digits[3:]
    if not _PHONE.match(digits):
        raise ValueError(f"not a UAE phone number: {phone!r}")
    return digits


def _minute(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def _second(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


_SLOT_SELECT = """
select s.id, s.starts_at, s.ends_at, p.ref, p.emirate, p.area, p.area_ar, p.address, p.address_ar,
       p.bedrooms, p.annual_rent_aed, a.name, a.name_ar
from viewing_slots s
join properties p on p.id = s.property_id
join agents a on a.id = p.agent_id
"""

_SLOT_KEYS = ("slot_id", "starts_at", "ends_at", "property_ref", "emirate", "area", "area_ar", "address",
              "address_ar", "bedrooms", "annual_rent_aed", "agent_name", "agent_name_ar")


class BookingService:
    def __init__(self, conn: sqlite3.Connection, clock, hold_ttl: timedelta = HOLD_TTL):
        self.conn = conn
        self.clock = clock
        self.hold_ttl = hold_ttl
        self._lock = threading.RLock()

    # ---- helpers ---------------------------------------------------------

    @contextmanager
    def _tx(self):
        with self._lock:
            self.conn.execute("begin immediate")
            try:
                yield
                self.conn.execute("commit")
            except BaseException:
                self.conn.execute("rollback")
                raise

    def _release_expired(self) -> None:
        self.conn.execute(
            "update holds set status='released' where status='active' and expires_at <= ?",
            (_second(self.clock.now()),),
        )

    def _audit(self, action: str, payload: dict) -> None:
        audit.append(self.conn, _second(self.clock.now()), action, payload)

    def _reject(self, action: str, payload: dict, err: BookingError | ValueError) -> None:
        # Rejections are written outside the failed transaction so they survive its rollback.
        with self._lock:
            self._audit(action, {**payload, "error": getattr(err, "code", "invalid"), "detail": str(err)})

    # ---- read ------------------------------------------------------------

    def list_slots(self, area: str | None = None, bedrooms: int | None = None, max_rent: int | None = None,
                   date: str | None = None, window: tuple[str, str] | None = None,
                   limit: int = 5) -> list[dict]:
        with self._lock:
            self._release_expired()
            sql = [_SLOT_SELECT, "where s.status='open' and s.starts_at >= ?",
                   "and not exists (select 1 from holds h where h.slot_id = s.id and h.status='active')"]
            args: list = [_minute(self.clock.now())]
            if area:
                sql.append("and lower(p.area) = lower(?)")
                args.append(area)
            if bedrooms is not None:
                sql.append("and p.bedrooms = ?")
                args.append(int(bedrooms))
            if max_rent is not None:
                sql.append("and p.annual_rent_aed <= ?")
                args.append(int(max_rent))
            if date:
                sql.append("and substr(s.starts_at, 1, 10) = ?")
                args.append(date)
            if window:
                sql.append("and substr(s.starts_at, 12, 5) >= ? and substr(s.starts_at, 12, 5) < ?")
                args.extend(window)
            sql.append("order by s.starts_at, p.annual_rent_aed, s.id limit ?")
            args.append(int(limit))
            rows = self.conn.execute(" ".join(sql), args).fetchall()
        return [dict(zip(_SLOT_KEYS, r)) for r in rows]

    def get_slot(self, slot_id: str) -> dict | None:
        row = self.conn.execute(_SLOT_SELECT + " where s.id = ?", (slot_id,)).fetchone()
        return dict(zip(_SLOT_KEYS, row)) if row else None

    def get_booking(self, booking_id: str) -> dict | None:
        row = self.conn.execute(
            "select id, slot_id, hold_id, phone, idempotency_key, status, created_at from bookings where id=?",
            (booking_id,),
        ).fetchone()
        if not row:
            return None
        keys = ("booking_id", "slot_id", "hold_id", "phone", "idempotency_key", "status", "created_at")
        return dict(zip(keys, row))

    # ---- write -----------------------------------------------------------

    def hold(self, slot_id: str, caller_id: str) -> dict:
        payload = {"slot_id": slot_id, "caller_id": caller_id}
        try:
            with self._tx():
                self._release_expired()
                row = self.conn.execute("select status from viewing_slots where id=?", (slot_id,)).fetchone()
                if row is None:
                    raise NotFound(f"slot {slot_id} not found")
                active = self.conn.execute(
                    "select 1 from holds where slot_id=? and status='active'", (slot_id,)
                ).fetchone()
                if row[0] != "open" or active:
                    raise SlotUnavailable(f"slot {slot_id} is not available")
                now = self.clock.now()
                hold = {
                    "hold_id": "h-" + uuid.uuid4().hex[:12],
                    "slot_id": slot_id,
                    "caller_id": caller_id,
                    "expires_at": _second(now + self.hold_ttl),
                }
                self.conn.execute(
                    "insert into holds values (?,?,?,?,?, 'active')",
                    (hold["hold_id"], slot_id, caller_id, _second(now), hold["expires_at"]),
                )
                self._audit("hold", hold)
        except BookingError as e:
            self._reject("hold_rejected", payload, e)
            raise
        return hold

    def confirm(self, hold_id: str, phone: str, idempotency_key: str) -> dict:
        payload = {"hold_id": hold_id, "idempotency_key": idempotency_key}
        try:
            if not idempotency_key:
                raise ValueError("idempotency_key is required")
            with self._tx():
                prior = self.conn.execute(
                    "select id, hold_id from bookings where idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if prior:
                    if prior[1] != hold_id:
                        raise Conflict("idempotency key already used for a different hold")
                    booking = self.get_booking(prior[0])
                    self._audit("confirm_replay", {**payload, "booking_id": prior[0]})
                    return {**booking, "replayed": True}
                self._release_expired()
                row = self.conn.execute("select slot_id, status from holds where id=?", (hold_id,)).fetchone()
                if row is None:
                    raise NotFound(f"hold {hold_id} not found")
                slot_id, status = row
                clean_phone = normalize_phone(phone)
                if status == "confirmed":
                    raise Conflict("hold already confirmed under another key")
                if status == "released":
                    raise HoldExpired(f"hold {hold_id} expired")
                booking = {
                    "booking_id": "b-" + uuid.uuid4().hex[:12],
                    "slot_id": slot_id,
                    "hold_id": hold_id,
                    "phone": clean_phone,
                    "idempotency_key": idempotency_key,
                    "status": "confirmed",
                    "created_at": _second(self.clock.now()),
                }
                self.conn.execute(
                    "insert into bookings values (?,?,?,?,?,?,?)",
                    tuple(booking[k] for k in ("booking_id", "slot_id", "hold_id", "phone",
                                               "idempotency_key", "status", "created_at")),
                )
                self.conn.execute("update holds set status='confirmed' where id=?", (hold_id,))
                self.conn.execute("update viewing_slots set status='booked' where id=?", (slot_id,))
                self._audit("confirm", {**payload, "booking_id": booking["booking_id"], "slot_id": slot_id,
                                        "phone": clean_phone})
        except (BookingError, ValueError) as e:
            self._reject("confirm_rejected", payload, e)
            raise
        return {**booking, "replayed": False}

    def release(self, hold_id: str) -> dict:
        """Give a held slot back before its expiry (the caller said no)."""
        try:
            with self._tx():
                row = self.conn.execute("select slot_id, status from holds where id=?", (hold_id,)).fetchone()
                if row is None:
                    raise NotFound(f"hold {hold_id} not found")
                if row[1] == "confirmed":
                    raise Conflict("hold already confirmed; cancel the booking instead")
                self.conn.execute("update holds set status='released' where id=?", (hold_id,))
                self._audit("release", {"hold_id": hold_id, "slot_id": row[0]})
        except BookingError as e:
            self._reject("release_rejected", {"hold_id": hold_id}, e)
            raise
        return {"hold_id": hold_id, "slot_id": row[0], "status": "released"}

    def cancel(self, booking_id: str) -> dict:
        try:
            with self._tx():
                booking = self.get_booking(booking_id)
                if booking is None:
                    raise NotFound(f"booking {booking_id} not found")
                if booking["status"] == "cancelled":
                    self._audit("cancel_replay", {"booking_id": booking_id})
                    return booking
                self.conn.execute("update bookings set status='cancelled' where id=?", (booking_id,))
                self.conn.execute("update viewing_slots set status='open' where id=?", (booking["slot_id"],))
                self._audit("cancel", {"booking_id": booking_id, "slot_id": booking["slot_id"]})
        except BookingError as e:
            self._reject("cancel_rejected", {"booking_id": booking_id}, e)
            raise
        return {**booking, "status": "cancelled"}
