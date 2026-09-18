"""Add waveform peaks, fact sources and a barge-in clip to data/calls.json: uv run python site/annotate.py

Needs ffmpeg but no speech models. The calls are replayed from the cached ASR to read the audit log.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from api import audit  # noqa: E402
from dialogue.grounding import Facts, ungrounded  # noqa: E402
from eval.harness import ANCHOR, ASRCache, reference_source, run_call  # noqa: E402
from eval.scripts import CALLS  # noqa: E402
from eval.toolnoise import NoisyService  # noqa: E402
from speech.audio import SAMPLE_RATE, write_wav  # noqa: E402
from speech.duplex import simulate  # noqa: E402

# 25 peaks a second is finer than a pixel on the longest call at desktop width.
PEAKS_PER_S = 25
BARGE_PEAKS_PER_S = 200
MONTHS = r"(?:October|أكتوبر)"
SPEECH = {"english": "local-v1", "arabic": "local-v1", "switched": "local-2pass", "outage": "local-v1"}
FIELD = {"address": "address", "agent": "agent_name", "area": "area", "price": "annual_rent_aed",
         "time": "starts_at", "date": "starts_at"}


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar",
                          str(SAMPLE_RATE), "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def peaks(pcm: np.ndarray, per_s: int, top: float | None = None) -> list[int]:
    step = SAMPLE_RATE // per_s
    n = len(pcm) // step
    blocks = np.abs(pcm[: n * step].astype(np.float32)).reshape(n, step).max(axis=1)
    top = top or float(blocks.max()) or 1.0
    # Square root lifts quiet consonants so word gaps stay visible without flattening loud vowels.
    return [int(round(99 * min(1.0, b / top) ** 0.5)) for b in blocks]


def replay(call_id: str, speech: str | None, noisy: str | None = None) -> dict:
    seen = {"n": 0}

    def watch(agent, t):
        es = audit.entries(agent.tools.conn)
        new, seen["n"] = es[seen["n"]:], len(es)
        s = agent.state
        return {"_audit": new, "_offered": s.offered, "_held": s.held_slot, "_hold": s.hold,
                "_booked": s.booked_slot, "_booking": s.booking, "_phone": s.slots.get("phone")}

    wrap = None if noisy is None else (lambda svc: NoisyService(svc, noisy, random.Random(1)))
    source = reference_source if speech is None else ASRCache(speech).get
    return run_call(call_id, CALLS[call_id], source, service=wrap, observe=watch)


def find(text: str, needle: str, taken: list, lo: int = 0, hi: int | None = None):
    hi = len(text) if hi is None else hi
    i = text.find(needle, lo, hi)
    while i >= 0:
        span = (i, i + len(needle))
        if not any(a < span[1] and span[0] < b for a, b in taken):
            return span
        i = text.find(needle, i + 1, hi)
    return None


def slot_facts(text: str, slot: dict, source: dict, taken: list, lo: int = 0, hi: int | None = None) -> list[dict]:
    """Every place a field of this API slot is spoken. Longest first, so an address wins over its area."""
    fields = [("address", slot["address"]), ("address", slot["address_ar"]),
              ("agent", slot["agent_name"]), ("agent", slot["agent_name_ar"]),
              ("area", slot["area"]), ("area", slot["area_ar"]),
              ("price", f"{slot['annual_rent_aed']:,}"), ("time", slot["starts_at"][11:16])]
    day = int(slot["starts_at"][8:10])
    m = re.compile(rf"\S+ {day} {MONTHS}").search(text, lo, len(text) if hi is None else hi)
    if m:
        fields.append(("date", m.group(0)))
    out = []
    for kind, needle in sorted(fields, key=lambda f: -len(f[1])):
        span = find(text, needle, taken, lo, hi)
        if span:
            taken.append(span)
            out.append({"at": list(span), "kind": kind, **source, "field": FIELD[kind]})
    return out


def tool_calls(entries: list[dict]) -> list[dict]:
    keep = {"tool_call", "tool_failed", "callback_requested", "hold", "confirm"}
    return [{"seq": e["seq"], "action": e["action"],
             "detail": {k: v for k, v in e["payload"].items() if k != "session_id"}}
            for e in entries if e["action"] in keep]


def annotate_call(entry: dict, out: dict) -> None:
    agent_turns = [t for t in entry["turns"] if t["who"] == "agent"]
    callers = [t for t in entry["turns"] if t["who"] == "caller"]
    assert len(agent_turns) == len(out["turns"]), entry["id"]
    last_list, phone_said, trace = None, None, []
    for k, (turn, site) in enumerate(zip(out["turns"], agent_turns)):
        assert turn["reply"] == site["text"], (entry["id"], turn["reply"], site["text"])
        calls = tool_calls(turn["_audit"])
        trace += [{**c, "turn": k} for c in calls]
        for c in calls:
            if c["action"] == "tool_call" and c["detail"]["tool"] == "list_slots":
                last_list = c["seq"]
        if turn["_phone"] and phone_said is None:
            phone_said = callers[k]["t0"]
        text, taken, facts = turn["reply"], [], []
        if turn["action"] == "offer" and turn["_offered"]:
            marks = [m.start() for m in re.finditer(r"(?:Option|الخيار) \d+:", text)]
            for i, slot in enumerate(turn["_offered"]):
                lo = marks[i] if i < len(marks) else 0
                hi = marks[i + 1] if i + 1 < len(marks) else None
                facts += slot_facts(text, slot, {"call": "list_slots", "seq": last_list, "ref": slot["slot_id"]},
                                    taken, lo, hi)
        elif turn["action"] == "ask_confirm" and turn["_held"]:
            facts += slot_facts(text, turn["_held"], {"call": "list_slots", "seq": last_list,
                                                       "ref": turn["_held"]["slot_id"],
                                                       "via": f"held by hold_slot {turn['_hold']['hold_id']}"},
                                taken)
        elif turn["action"] == "confirmed" and turn["_booked"]:
            confirm = next((c["seq"] for c in calls if c["action"] == "confirm"), None)
            facts += slot_facts(text, turn["_booked"], {"call": "confirm_booking", "seq": confirm,
                                                         "ref": turn["_booking"]["booking_id"]}, taken)
        span = find(text, turn["_phone"], taken) if turn["_phone"] else None
        if span:
            facts.append({"at": list(span), "kind": "phone", "call": "caller", "said_at": phone_said,
                          "field": "phone"})
        site["facts"] = sorted(facts, key=lambda f: f["at"][0])
        site["checked"] = {"action": turn["action"], "ungrounded": len(turn["ungrounded"])}
    entry["trace"] = trace


def barge_in(english: dict, pcm: np.ndarray) -> dict:
    """The next caller line played over the long offer reply, through the same simulate() the trials use."""
    turns = english["turns"]
    k = next(i for i, t in enumerate(turns) if t["who"] == "agent" and t["t1"] - t["t0"] > 10)
    agent_t, caller_t = turns[k], turns[k + 1]

    def cut(t):
        return pcm[int(t["t0"] * SAMPLE_RATE):int(t["t1"] * SAMPLE_RATE)]

    agent, caller = cut(agent_t), cut(caller_t)
    # Lands inside "Option 1" on this recording, so the stop falls in speech rather than in a pause.
    offset = 4.1
    lead = np.zeros(int(offset * SAMPLE_RATE), dtype=np.int16)
    r = simulate(agent, np.concatenate([lead, caller]), echo_gain=0.25)
    heard = r.stopped_at_s if r.barged_in else len(agent) / SAMPLE_RATE
    total = max(heard, offset + len(caller) / SAMPLE_RATE) + 0.3
    mix = np.zeros(int(total * SAMPLE_RATE), dtype=np.float32)
    n = int(heard * SAMPLE_RATE)
    mix[:n] += agent[:n]
    mix[len(lead):len(lead) + len(caller)] += caller
    wav = ROOT / "audio" / "bargein.wav"
    write_wav(str(wav), np.clip(mix, -32768, 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-map_metadata", "-1", "-fflags", "+bitexact",
                    "-flags:a", "+bitexact", "-c:a", "aac", "-b:a", "48k", "-ac", "1", str(wav.with_suffix(".m4a"))],
                   check=True)
    wav.unlink()
    top = float(max(np.abs(agent.astype(np.float32)).max(), np.abs(caller.astype(np.float32)).max()))
    return {"audio": "audio/bargein.m4a", "agent_text": agent_t["text"], "caller_text": caller_t["text"],
            "onset": round(r.onset_s, 2), "stopped": round(heard, 2),
            "latency": round(heard - r.onset_s, 2), "agent_full": round(len(agent) / SAMPLE_RATE, 2),
            "duration": round(total, 2), "peaks_per_s": BARGE_PEAKS_PER_S,
            "agent_peaks": peaks(agent, BARGE_PEAKS_PER_S, top),
            "caller_peaks": peaks(np.concatenate([lead, caller]), BARGE_PEAKS_PER_S, top)}


def rejected_detail(g: dict) -> dict:
    """Re-run the check on the rejected reply, against the slot the same call books on reference text."""
    cid = re.search(r"\(([\w-]+)\)", g["intro"]).group(1)
    out = replay(cid, None)
    t = next(t for t in out["turns"] if t["action"] == "confirmed")
    if t["reply"] != g["spoken"]:
        return g
    slot = t["_booked"]
    bad = ungrounded(g["model_reply"], Facts.from_slots([slot], caller={"phone": t["_phone"]}), ANCHOR.date())
    return {**g, "violations": [{"kind": v.kind, "value": str(v.value)} for v in bad],
            "truth": {"call": "list_slots", "ref": slot["slot_id"], "starts_at": slot["starts_at"]}}


def main() -> None:
    path = ROOT / "data" / "calls.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data["calls"]:
        pcm = decode(ROOT / entry["audio"])
        entry["peaks"], entry["peaks_per_s"] = peaks(pcm, PEAKS_PER_S), PEAKS_PER_S
        annotate_call(entry, replay(entry["call_id"], SPEECH[entry["id"]],
                                    "down-at-confirm" if entry["outage"] else None))
        if entry["id"] == "english":
            data["barge_in"] = barge_in(entry, pcm)
    if data.get("grounding"):
        data["grounding"] = rejected_detail(data["grounding"])
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
