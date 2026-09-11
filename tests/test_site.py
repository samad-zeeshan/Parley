"""The demo page shows only numbers from eval/results.json, and its recordings exist and stay small."""

import json
import re
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parent.parent / "site"
DATA = SITE / "data" / "calls.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="site data not built")


def load():
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_tiles_match_the_results_file():
    import importlib.util

    spec = importlib.util.spec_from_file_location("site_build", SITE / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    assert load()["summary"] == build.summary()


def test_four_calls_and_the_last_is_the_outage():
    ids = [c["id"] for c in load()["calls"]]
    assert ids == ["english", "arabic", "switched", "outage"]
    outage = load()["calls"][-1]
    assert outage["outage"] and not outage["booking"]


def test_audio_exists_and_the_page_stays_light():
    total = 0
    for c in load()["calls"]:
        f = SITE / c["audio"]
        assert f.exists(), f
        total += f.stat().st_size
    assert total < 8_000_000


def test_timeline_is_ordered():
    for c in load()["calls"]:
        starts = [t["t0"] for t in c["turns"]]
        assert starts == sorted(starts) and starts[-1] < c["duration"]


def test_no_number_is_written_into_the_page():
    """Numbers come from data/calls.json only, so the page cannot drift from the results."""
    html = (SITE / "index.html").read_text(encoding="utf-8")
    text = re.sub(r"<[^>]+>", " ", html)
    assert not re.search(r"\d", text)
