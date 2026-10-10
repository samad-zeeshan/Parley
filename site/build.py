"""Build the demo's data and audio from real evaluation runs: uv run --extra speech python site/build.py

Each call is replayed through the harness with the cached local ASR, so the page shows exactly what the code does.
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

from eval.harness import ASRCache, render_turn, run_call  # noqa: E402
from eval.scripts import CALLS  # noqa: E402
from eval.toolnoise import NoisyService  # noqa: E402
from speech.audio import SAMPLE_RATE, read_wav, silence, write_wav  # noqa: E402
from speech.translit import to_latin  # noqa: E402
from speech.tts import PiperTTS  # noqa: E402

AFTER_CALLER, AFTER_AGENT = 0.35, 0.55
# The local config that books the most calls on clean audio: the one a customer would run.
DEPLOYED = "local-v1"
ARABIC = re.compile(r"[؀-ۿ]")
VOICE = "A synthetic voice speaks the caller's lines. "
HEARD = "Caller lines show what Parley heard. If that is different from the script, the script line appears under it."
NOTES = {
    "english": "This is a scripted English booking call. " + VOICE + "Parley uses its offline speech setup, which "
               "needs no internet. " + HEARD,
    "arabic": "This is a scripted Gulf Arabic booking call. " + VOICE + "Parley uses its offline speech setup, which "
              "needs no internet. " + HEARD,
    "switched": "This scripted call moves between Gulf Arabic and English. " + VOICE + "Parley uses a second "
                "offline setup, which listens to each part between pauses on its own. " + HEARD,
}


def pick(kind: str, local: dict) -> str:
    """The completed call of this kind with the fewest turns; the shortest failed one if none completed."""
    calls = [c for c in local if c["kind"] == kind]
    done = [c for c in calls if c["completed"]]
    return min(done or calls, key=lambda c: (len(c["turns"]), c["call_id"]))["call_id"]


def replay(call_id: str, speech: str, noisy: str | None = None) -> tuple[dict, dict]:
    state = {}

    def watch(agent, t):
        state.update(slot=agent.state.booked_slot, phone=agent.state.slots.get("phone"), lang=agent.lang)
        return {}

    wrap = None if noisy is None else (lambda svc: NoisyService(svc, noisy, random.Random(1)))
    out = run_call(call_id, CALLS[call_id], ASRCache(speech).get, service=wrap, observe=watch)
    return out, state


def assemble(cid: str, out: dict, tts: PiperTTS, name: str) -> tuple[list[dict], float, float | None]:
    call = CALLS[cid]
    pcm, turns, t, card_at = [], [], 0.0, None
    for turn in out["turns"]:
        clip = read_wav(str(render_turn(cid, turn["line"], call["turns"][turn["line"]], call["voice"],
                                        turn["attempt"])))[0]
        turns.append({"who": "caller", "t0": round(t, 2), "t1": round(t + len(clip) / SAMPLE_RATE, 2),
                      "text": turn["hyp"], "latin": to_latin(turn["hyp"]) if turn["hyp"] != to_latin(turn["hyp"]) else "",
                      "said": turn["ref"]})
        pcm += [clip, silence(AFTER_CALLER)]
        t += len(clip) / SAMPLE_RATE + AFTER_CALLER
        speech = tts.synthesize(turn["reply"], turn["reply_lang"]).pcm
        tag = {"unavailable": "safe reply", "listen": "waits"}.get(turn["action"], "")
        turns.append({"who": "agent", "t0": round(t, 2), "t1": round(t + len(speech) / SAMPLE_RATE, 2),
                      "text": turn["reply"], "latin": to_latin(turn["reply"]) if turn["reply_lang"] == "ar" else "",
                      "tag": tag})
        if turn["action"] in ("confirmed", "unavailable") and card_at is None:
            card_at = round(t, 2)
        pcm += [speech, silence(AFTER_AGENT)]
        t += len(speech) / SAMPLE_RATE + AFTER_AGENT
    wav = ROOT / "audio" / f"{name}.wav"
    wav.parent.mkdir(exist_ok=True)
    write_wav(str(wav), np.concatenate(pcm))
    # AAC in an m4a plays in every current browser, and 48 kb/s mono is plenty for TTS voices.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "aac", "-b:a", "48k", "-ac", "1",
                    str(wav.with_suffix(".m4a"))], check=True)
    wav.unlink()
    return turns, round(t, 2), card_at


def booking_rows(state: dict) -> list[list[str]]:
    s, ar = state["slot"], state["lang"] == "ar"
    return [["When", f"{s['starts_at'][:10]} at {s['starts_at'][11:16]}"],
            ["Where", s["address_ar"] if ar else s["address"]],
            ["Rent", f"AED {s['annual_rent_aed']:,} a year"],
            ["Agent", s["agent_name_ar"] if ar else s["agent_name"]],
            ["Caller", state["phone"]]]


def grounding_example() -> dict | None:
    """A model reply the check refused, preferring a wrong time on a booking confirmation.

    A bare "date" flag can be a false alarm: the normalizer reads "اليوم" (the day) as "today".
    """
    path = ROOT.parent / "eval" / "out" / "mtva" / "llm+phrasing_reference.json"
    if not path.exists():
        return None
    found = []
    for c in json.loads(path.read_text(encoding="utf-8")):
        for t in c["turns"]:
            kinds = set(t["rejected"])
            if not t.get("rejected_text") or kinds == {"language"}:
                continue
            rank = (0 if "time" in kinds and t["action"] == "confirmed" else 1 if "time" in kinds
                    else 2 if "tower" in kinds else 3 if any(ch.isdigit() for ch in t["rejected_text"]) else 9)
            found.append((rank, c["call_id"], t))
    if not found:
        return None
    rank, cid, t = min(found, key=lambda x: x[0])
    return {"call_id": cid,
            "intro": f"This reply comes from a recorded {'Arabic' if ARABIC.search(t['reply']) else 'English'} "
                     "call. A language model on the same computer reworded the booking confirmation and got a fact "
                     "wrong. The fact check caught it, so Parley said its fixed wording instead.",
            "model_reply": t["rejected_text"], "flagged": ", ".join(sorted(set(t["rejected"]))),
            "spoken": t["reply"],
            "footnote": "Before the comparison, the check changes dates, times and prices written as words into "
                        "numbers. It does the same with the words of the caller."}


def summary() -> dict:
    r = json.loads((ROOT.parent / "eval" / "results.json").read_text(encoding="utf-8"))
    v1 = json.loads((ROOT.parent / "eval" / "results_v1.json").read_text(encoding="utf-8"))
    loc = r["speech"][DEPLOYED]["dialogue"]
    sw_key, sw = max(((k, s["dialogue"]["task_completion_by_call_language"]["switch"])
                      for k, s in r["speech"].items() if "|" not in k), key=lambda kv: kv[1]["completed"])
    v1sw = v1["configs"]["rules"]["task_completion_by_call_language"]["switch"]
    faults = r["tool_noise"]["modes"]
    fastest = min(((k, v) for k, v in r["latency"]["by_config"].items() if "|" not in k),
                  key=lambda kv: kv[1]["early_decode"]["p95_s"])
    lat = fastest[1]
    target = r["latency"]["target_s"]
    fast = lat["early_decode"]["p95_s"] <= target
    pace = f"under the {target} second target, which is good" if fast else \
        f"over the {target} second target, which is too slow"
    # Labels are for a reader outside computing. Config names only ever appear in the small mono tag.
    return {"tiles": [
        {"value": f"{loc['completed']} of {loc['calls']}", "label": "Calls that ended in a booking, offline setup",
         "verdict": "Fewer than half of the calls ended in a booking. That is not good enough yet. The tests did "
                    "not include a cloud setup, so there is no comparison.",
         "tag": DEPLOYED},
        {"value": f"{sw['completed']} of {sw['calls']}",
         "label": "Calls in both languages that ended in a booking, best offline setup",
         "verdict": f"This setup booked more of these calls than the first version, which booked "
                    f"{v1sw['completed']} of {v1sw['calls']}. That is still very few.",
         "tag": sw_key},
        {"value": str(sum(m["false_facts_spoken"] for m in faults.values())),
         "label": "Wrong facts spoken when the booking system failed",
         "verdict": f"The tests made the booking system fail in {len(faults) - ('clean' in faults)} different "
                    "ways. Parley never said a wrong fact, which is the result it needs.",
         "tag": "false_facts_spoken"},
        {"value": f"{lat['early_decode']['p95_s']} s", "label": "Slowest 1 in 20 replies, fastest offline setup",
         "verdict": f"That is {pace}. But this setup booked only {lat['completed']} of {lat['calls']} calls, "
                    "so the fast replies cost many bookings.",
         "tag": f"p95, {fastest[0]}"},
    ], "note": "Every figure comes from the saved test results. The project page on GitHub has every table. It "
               "also shows the results that got worse."}


def main() -> None:
    runs = json.loads((ROOT.parent / "eval" / "transcripts.json").read_text(encoding="utf-8"))["calls"]
    main, two_pass = runs[DEPLOYED], runs["local-2pass"]
    tts = PiperTTS()
    specs = [("english", pick("en", main), DEPLOYED, None), ("arabic", pick("gulf", main), DEPLOYED, None),
             ("switched", pick("switch", two_pass), "local-2pass", None),
             ("outage", pick("en", main), DEPLOYED, "down-at-confirm")]
    calls = []
    for name, cid, speech, noisy in specs:
        out, state = replay(cid, speech, noisy)
        turns, duration, card_at = assemble(cid, out, tts, name)
        entry = {"id": name, "call_id": cid, "audio": f"audio/{name}.m4a", "duration": duration, "turns": turns,
                 "completed": out["completed"], "card_at": card_at, "booking": None, "outage": None}
        if noisy:
            entry["note"] = ("This is the same English call, but the booking system does not answer when Parley "
                             "tries to confirm. Parley tells the caller, does not claim the viewing is booked, and "
                             "arranges a call back.")
            entry["outage"] = {"title": "Not booked, and Parley said so",
                               "lines": ["The booking system did not answer when Parley tried to confirm, even "
                                         "after a second try.",
                                         f"Parley arranged a call back to {state['phone']} and wrote it in the log."
                                         if state.get("phone") else "Parley did not know a phone number, so it could not arrange a call back."]}
        else:
            entry["note"] = NOTES[name]
            if out["completed"] and state.get("slot"):
                entry["booking"] = {"rows": booking_rows(state)}
            else:
                entry["no_booking"] = "This call did not end in a booking. The page shows it as it happened."
        calls.append(entry)
        print(name, cid, "completed" if out["completed"] else "not completed", f"{duration:.0f} s", flush=True)
    data = {"calls": calls, "grounding": grounding_example(), "summary": summary()}
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "calls.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8",
                                              newline="\n")
    # Peaks, fact sources and the barge-in clip come from a second pass that needs no speech models.
    import annotate

    annotate.main()


if __name__ == "__main__":
    main()
