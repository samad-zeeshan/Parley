from datetime import datetime

import pytest

from api.clock import DUBAI, FixedClock
from api.db import connect
from api.seed import seed
from api.service import BookingService

# 2026-10-01 is a Thursday. Every test runs against the same synthetic world.
ANCHOR = datetime(2026, 10, 1, 9, 0, tzinfo=DUBAI)


@pytest.fixture
def clock():
    return FixedClock(ANCHOR)


@pytest.fixture
def conn():
    c = connect(":memory:")
    seed(c, anchor=ANCHOR.date(), rng_seed=7)
    yield c
    c.close()


@pytest.fixture
def svc(conn, clock):
    return BookingService(conn, clock)
