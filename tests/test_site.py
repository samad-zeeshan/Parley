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


# Internal names a stranger cannot read. They may appear only in a tile's small mono tag, never as a label.
BANNED = ["local-v1", "local-tiny", "local-2pass", "en-booking", "gulf-booking", "code-switched", "p95",
          "tool outage", "grounding check", "booking api", "config"]
LABEL_WORDS = 12
# Simplified Technical English caps a descriptive sentence at 25 words.
SENTENCE_WORDS = 25
FILLER = ["comprehensive", "robust", "seamless", "cutting-edge", "state-of-the-art", "powerful", "leverage",
          "utilize", "ensure", "delve", "showcase", "crucial", "various", "numerous", "simply", "just", "really",
          "basically", "actually"]


def reader_strings():
    """Every string from the data file that the page shows as prose or as a label."""
    d = load()
    out = [d["summary"]["note"]]
    for t in d["summary"]["tiles"]:
        out += [t["label"], t.get("verdict", "")]
    for c in d["calls"]:
        out += [c.get("note") or "", c.get("no_booking") or ""]
        if c.get("outage"):
            out += [c["outage"]["title"], *c["outage"]["lines"]]
    if d.get("grounding"):
        out += [d["grounding"]["intro"], d["grounding"]["footnote"]]
    return [s for s in out if s]


def page_strings():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S)
    attrs = re.findall(r'(?:aria-label|content|alt|title)="([^"]*)"', html)
    js = (SITE / "app.js").read_text(encoding="utf-8")
    js = re.sub(r"^\s*//.*$", "", js, flags=re.M)
    literals = re.findall(r'"([^"\n]*)"|`([^`]*)`', js)
    # Split on tags so a heading and the paragraph after it do not read as one long sentence.
    return [*re.split(r"<[^>]+>", html), *attrs, *(a or b for a, b in literals)]


@pytest.mark.parametrize("word", BANNED)
def test_no_internal_name_is_used_as_a_label(word):
    for s in reader_strings() + page_strings():
        assert word not in s.lower(), (word, s[:120])


def test_labels_and_sentences_stay_short():
    d = load()
    for t in d["summary"]["tiles"]:
        assert len(t["label"].split()) <= LABEL_WORDS, t["label"]
    for s in reader_strings() + page_strings() + readme_prose():
        for sentence in sentences(s):
            assert len(sentence.split()) <= SENTENCE_WORDS, sentence



def readme_prose():
    """README paragraphs, without code blocks and tables. Lines inside the generated block count too."""
    text = (SITE.parent / "README.md").read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    # Alt text is read on its own, so it ends a sentence. Link targets are not prose.
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1.", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("|"))
    return [p for p in re.split(r"\n\s*\n|\n(?=[-#])", text) if p.strip()]


def sentences(s):
    return [x for x in re.split(r"(?<=[.!?:])\s+", " ".join(s.split())) if x]


def test_no_filler_words_or_heavy_punctuation():
    for s in reader_strings() + page_strings() + readme_prose():
        for word in FILLER:
            assert not re.search(rf"\b{word}\b", s, re.I), (word, s[:120])
        assert "—" not in s and ";" not in s, s[:120]


# A rough passive check: a form of "be", at most one word, then a past participle.
# It misses passives without "be" and flags some adjectives, so the allowlist below holds the misses we accept.
PARTICIPLE = r"\w+ed|made|done|found|given|held|kept|left|run|seen|sent|shown|taken|written|built|caught"
PASSIVE = re.compile(rf"\b(?:is|are|was|were|be|been|being)\s+(?:\w+\s+)?(?:{PARTICIPLE})\b", re.I)
PASSIVE_OK = {
    # None today. Add a sentence here only with a one-line reason why active voice reads worse.
}


def test_no_passive_voice():
    for s in reader_strings() + page_strings() + readme_prose():
        for sentence in sentences(s):
            if sentence not in PASSIVE_OK:
                assert not PASSIVE.search(sentence), sentence


def test_every_recorder_step_has_a_selector():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for name in ["call-en", "call-ar", "call-outage", "rejected"]:
        assert f'data-demo="{name}"' in html, name
