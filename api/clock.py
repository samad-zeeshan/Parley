"""Injectable clock. The UAE has no daylight saving, so Dubai time is UTC+4 all year."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

DUBAI = timezone(timedelta(hours=4), name="Asia/Dubai")


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(DUBAI).replace(microsecond=0)


class FixedClock:
    """A clock tests and the evaluation harness move by hand."""

    def __init__(self, start: datetime):
        if start.tzinfo is None:
            raise ValueError("FixedClock needs an aware datetime")
        self._now = start.astimezone(DUBAI)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now = self._now + delta

    def set(self, when: datetime) -> None:
        self._now = when.astimezone(DUBAI)
