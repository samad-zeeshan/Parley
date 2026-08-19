"""Validated tool calls, after Change-Gate's tool layer.

Every call is checked twice before it reaches the booking API:
1. its arguments against the tool's JSON schema, and
2. against what this session has seen: a hold may only target a slot the API
   offered in this session, and a confirm only a hold this session made.

Accepted and rejected calls both go into the hash-chained audit log, tagged
with the session id. The booking API writes its own entries for hold, confirm
and cancel, so a booking leaves a tool_call entry and an API entry.
"""

from __future__ import annotations

import sqlite3

from jsonschema import Draft202012Validator

from api import audit
from api.service import BookingError, BookingService, ServiceUnavailable

from .schema import TOOL_SCHEMAS

_VALIDATORS = {name: Draft202012Validator(schema) for name, schema in TOOL_SCHEMAS.items()}
_METRIC_ACTION = {"list_slots": "list", "hold_slot": "hold", "release_hold": "release",
                  "confirm_booking": "confirm", "cancel_booking": "cancel"}


class ToolFailed(Exception):
    """The backend did not give a usable answer after one retry: down, timed out, or inconsistent."""

    def __init__(self, tool: str, reason: str):
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason


class ToolRejected(Exception):
    def __init__(self, tool: str, reason: str):
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason


class ToolExecutor:
    def __init__(self, service: BookingService, conn: sqlite3.Connection, session_id: str, metrics=None):
        self.svc = service
        self.conn = conn
        self.session_id = session_id
        self.metrics = metrics
        self.offered_ids: set[str] = set()
        self.hold_ids: set[str] = set()
        self.booking_ids: set[str] = set()

    def _log(self, action: str, payload: dict) -> None:
        with self.svc._lock:
            audit.append(self.conn, self.svc.clock.now().isoformat(timespec="seconds"), action,
                         {"session_id": self.session_id, **payload})

    def _reject(self, tool: str, args: dict, reason: str) -> ToolRejected:
        self._log("tool_rejected", {"tool": tool, "args": args, "reason": reason})
        if self.metrics:
            self.metrics.tool_rejections.labels(tool).inc()
        return ToolRejected(tool, reason)

    def validate(self, tool: str, args: dict) -> None:
        if tool not in _VALIDATORS:
            raise self._reject(tool, args, "unknown tool")
        errors = sorted(_VALIDATORS[tool].iter_errors(args), key=lambda e: e.path)
        if errors:
            raise self._reject(tool, args, "schema: " + errors[0].message)
        if tool == "hold_slot" and args["slot_id"] not in self.offered_ids:
            raise self._reject(tool, args, "slot was not offered by the API in this session")
        if tool in ("confirm_booking", "release_hold") and args["hold_id"] not in self.hold_ids:
            raise self._reject(tool, args, "no such hold in this session")
        if tool == "cancel_booking" and args["booking_id"] not in self.booking_ids:
            raise self._reject(tool, args, "no such booking in this session")

    def request_callback(self, phone: str, reason: str) -> None:
        """The callback queue is the local audit log, which stays writable when the booking API is down."""
        self._log("callback_requested", {"phone": phone, "reason": reason})

    def call(self, tool: str, args: dict):
        args = dict(args)
        self.validate(tool, args)
        self._log("tool_call", {"tool": tool, "args": args})
        # One retry. confirm_booking is idempotent on its key, so a retry after a timeout that did the
        # work replays the booking instead of making a second one.
        for attempt in (1, 2):
            try:
                out = self._run(tool, dict(args))
                break
            except (ServiceUnavailable, TimeoutError) as e:
                reason = "timeout" if isinstance(e, TimeoutError) else "unavailable"
                self._log("tool_failed", {"tool": tool, "error": reason, "attempt": attempt})
                self._count(tool, reason)
                if attempt == 2:
                    raise ToolFailed(tool, reason) from e
            except BookingError as e:
                self._log("tool_failed", {"tool": tool, "error": e.code})
                self._count(tool, e.code)
                raise
        self._count(tool, "replay" if isinstance(out, dict) and out.get("replayed") else "ok")
        return out

    def _count(self, tool: str, outcome: str) -> None:
        if self.metrics:
            self.metrics.booking_actions.labels(_METRIC_ACTION[tool], outcome).inc()

    def _run(self, tool: str, args: dict):
        if tool == "list_slots":
            window = (args.pop("start", "00:00"), args.pop("end", "24:00"))
            has_window = window != ("00:00", "24:00")
            out = self.svc.list_slots(window=window if has_window else None, limit=args.pop("limit", 5), **args)
            self.offered_ids.update(s["slot_id"] for s in out)
            return out
        if tool == "hold_slot":
            out = self.svc.hold(args["slot_id"], caller_id=self.session_id)
            self.hold_ids.add(out["hold_id"])
            return out
        if tool == "release_hold":
            return self.svc.release(args["hold_id"])
        if tool == "confirm_booking":
            # One key per session and hold: a retried confirm replays instead of double-booking.
            key = f"{self.session_id}:{args['hold_id']}"
            out = self.svc.confirm(args["hold_id"], args["phone"], idempotency_key=key)
            self.booking_ids.add(out["booking_id"])
            return out
        if tool == "cancel_booking":
            return self.svc.cancel(args["booking_id"])
        raise ToolRejected(tool, "unreachable")
