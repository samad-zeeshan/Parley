"""Forty scripted callers through each local speech config with the rule parser: ASR, code switching, calls, barge-in.

    uv run --extra speech python -m eval.run_eval --speech local
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from api.clock import DUBAI
from speech.audio import SAMPLE_RATE, read_wav, silence
from speech.backends import CONFIGS
from speech.duplex import BARGE_IN_BOUND_S, simulate
from speech.langid import identify, reply_language
from speech.textnorm import normalize_orthography

from .harness import (ANCHOR, CALL_KINDS, DIALECTS, HANGOVER_S, ROOT, ASRCache, render_turn, run_call,
                      update_results)
from .latency import MODES, turn_latency
from .scoring import (DATE_KINDS, DIGIT_KINDS, TIME_KINDS, entity_errors, f1, percentile, slot_counts,
                      span_errors)
from .scripts import CALLS
from .wer import ZERO, count_errors

LOCAL = ["local", "local-v1", "local-2pass", "local-base", "local-tiny"]


def _r(x, n=3):
    return None if x is None else round(x, n)


def script_hint(call: dict, line: int) -> str | None:
    """The language the agent would be replying in before this line: the ASR hint for the per-line table."""
    return None if line == 0 else reply_language(identify(call["turns"][line - 1]["text"]))


def aggregate_asr(cache: ASRCache) -> dict:
    """Every scripted line once (attempt 0), whether or not a call reached it."""
    from speech.normalize import normalize

    today = ANCHOR.date()
    per = {d: {"raw": ZERO, "norm": ZERO, "digit": ZERO, "date": ZERO, "time": ZERO, "n": 0, "lid": 0}
           for d in DIALECTS}
    cs = {"en": ZERO, "ar": ZERO, "latin": 0, "lines": 0}
    for call_id, call in CALLS.items():
        for line, t in enumerate(call["turns"]):
            hyp = cache.get(call_id, line, 0, script_hint(call, line))["text"]
            p = per[t["dialect"]]
            p["n"] += 1
            p["raw"] += count_errors(normalize_orthography(t["text"]), normalize_orthography(hyp))
            p["norm"] += count_errors(normalize(t["text"], today).text, normalize(hyp, today).text)
            p["digit"] += entity_errors(t["text"], hyp, DIGIT_KINDS, today)
            p["date"] += entity_errors(t["text"], hyp, DATE_KINDS, today)
            p["time"] += entity_errors(t["text"], hyp, TIME_KINDS, today)
            p["lid"] += identify(hyp).label == t["dialect"]
            if {lang for lang, _ in t["segments"]} == {"en", "ar"}:
                s = span_errors(t["segments"], hyp)
                cs["en"] += s["en"]
                cs["ar"] += s["ar"]
                cs["latin"] += s["en_latin_kept"]
                cs["lines"] += 1
    asr = {}
    for d, p in per.items():
        asr[d] = {"lines": p["n"], "wer": _r(p["raw"].wer), "wer_normalized": _r(p["norm"].wer),
                  "digit_wer": _r(p["digit"].wer) if p["digit"].ref_words else None,
                  "date_wer": _r(p["date"].wer) if p["date"].ref_words else None,
                  "time_wer": _r(p["time"].wer) if p["time"].ref_words else None,
                  "dialect_id": _r(p["lid"] / p["n"]) if p["n"] else None}
    codeswitch = {"lines": cs["lines"], "en_words": cs["en"].ref_words, "ar_words": cs["ar"].ref_words,
                  "en_span_wer": _r(cs["en"].wer), "ar_span_wer": _r(cs["ar"].wer),
                  "en_kept_latin": _r(cs["latin"] / cs["en"].ref_words) if cs["en"].ref_words else None}
    return {"asr_by_dialect": asr, "codeswitch": codeswitch}


def aggregate_calls(calls: list[dict]) -> dict:
    per = {d: {"n": 0, "ok": 0, "tp": 0, "fp": 0, "fn": 0, "adh": 0, "lat": []} for d in DIALECTS}
    for c in calls:
        for t in c["turns"]:
            p = per[t["dialect"]]
            p["n"] += 1
            p["ok"] += t["intent"] == t["gold_intent"]
            tp, fp, fn = slot_counts(t["gold_slots"], t["slots"])
            p["tp"], p["fp"], p["fn"] = p["tp"] + tp, p["fp"] + fp, p["fn"] + fn
            p["adh"] += t["adherent"]
            p["lat"].append(turn_latency(t, "early_decode"))
    by = {d: {"turns": p["n"], "intent_accuracy": _r(p["ok"] / p["n"]) if p["n"] else None,
              "slot_f1": _r(f1(p["tp"], p["fp"], p["fn"])),
              "reply_language_adherence": _r(p["adh"] / p["n"]) if p["n"] else None,
              "latency_p95_s": _r(percentile(p["lat"], 95), 2) if p["n"] else None} for d, p in per.items()}
    tasks = {}
    for kind in CALL_KINDS:
        cs = [c for c in calls if c["kind"] == kind]
        tasks[kind] = {"completed": sum(c["completed"] for c in cs), "calls": len(cs),
                       "turns_per_call": [len(c["turns"]) for c in cs]}
    turns = [t for c in calls for t in c["turns"]]
    return {"by_dialect": by, "task_completion_by_call_language": tasks, "turns": len(turns),
            "completed": sum(c["completed"] for c in calls), "calls": len(calls),
            "wrong_actions": sum(len(c["wrong_actions"]) for c in calls),
            "spoken_replies_with_ungrounded_facts": sum(c["spoken_ungrounded"] for c in calls),
            "latency": {m: {"p50_s": _r(percentile([turn_latency(t, m) for t in turns], 50), 2),
                            "p95_s": _r(percentile([turn_latency(t, m) for t in turns], 95), 2)} for m in MODES}}


def barge_in_trials(calls: list[dict]) -> list[dict]:
    """The next caller line played over the agent's real Piper reply, at a random offset, with echo."""
    rng = np.random.default_rng(2026)
    trials = []
    for c in calls:
        turns = c["turns"]
        for k, t in enumerate(turns[:-1]):
            agent_pcm = t.get("_speech")
            if agent_pcm is None or len(agent_pcm) < 1.5 * SAMPLE_RATE:
                continue
            nxt = turns[k + 1]
            call = CALLS[c["call_id"]]
            caller_pcm = read_wav(str(render_turn(c["call_id"], nxt["line"], call["turns"][nxt["line"]],
                                                  call["voice"], nxt["attempt"])))[0]
            offset = float(rng.uniform(0.3, len(agent_pcm) / SAMPLE_RATE - 1.0))
            r = simulate(agent_pcm, np.concatenate([silence(offset), caller_pcm]), echo_gain=0.25)
            echo_only = simulate(agent_pcm, silence(len(agent_pcm) / SAMPLE_RATE + 0.5), echo_gain=0.25)
            ok = r.barged_in and not r.false_stop and r.latency_s is not None and r.latency_s <= BARGE_IN_BOUND_S
            trials.append({"call_id": c["call_id"], "dialect": nxt["dialect"], "success": ok,
                           "latency_s": _r(r.latency_s), "false_stop_on_echo": echo_only.barged_in})
    return trials


def aggregate_barge_in(trials: list[dict]) -> dict:
    out = {}
    for d in DIALECTS:
        ts = [t for t in trials if t["dialect"] == d]
        lats = [t["latency_s"] for t in ts if t["latency_s"] is not None]
        out[d] = {"trials": len(ts), "success": sum(t["success"] for t in ts),
                  "rate": _r(sum(t["success"] for t in ts) / len(ts)) if ts else None,
                  "latency_p50_s": _r(percentile(lats, 50)) if lats else None,
                  "latency_max_s": _r(max(lats)) if lats else None,
                  "false_stops_on_echo_only": sum(t["false_stop_on_echo"] for t in ts)}
    return out


def machine() -> dict:
    cpu = platform.processor()
    try:
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip() or cpu
    except Exception:  # noqa: BLE001
        pass
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "gpu": "none used", "os": platform.platform(),
            "python": sys.version.split()[0]}


def _strip(calls: list[dict]) -> list[dict]:
    return [{**c, "turns": [{k: v for k, v in t.items() if not k.startswith("_")} for t in c["turns"]]} for c in calls]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", default=",".join(LOCAL))
    ap.add_argument("--dialogue", default="rules")
    args = ap.parse_args()
    from speech.tts import PiperTTS

    tts = PiperTTS()
    tts.synthesize("warm up", "en")
    tts.synthesize("تجربة", "ar")
    path = ROOT / "transcripts.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    transcripts = old if isinstance(old.get("calls"), dict) and "rules" not in old["calls"] else {"calls": {}}

    for cfg in args.speech.split(","):
        key = cfg if args.dialogue == "rules" else f"{cfg}|{args.dialogue}"
        t0 = time.time()
        print(f"{cfg}: ASR over every scripted line", flush=True)
        cache = ASRCache(cfg)
        part = aggregate_asr(cache)
        cache.save()
        print(f"{cfg}: forty calls", flush=True)
        calls = [run_call(cid, call, cache.get, args.dialogue, tts, keep_audio=key == "local")
                 for cid, call in CALLS.items()]
        cache.save()
        part["dialogue"] = aggregate_calls(calls)
        part["description"] = CONFIGS[cfg].description
        part["runtime_s"] = round(time.time() - t0, 1)
        update_results("speech", {key: part}, merge=True)
        print(f"{key}: completed {part['dialogue']['completed']}/40, en-span WER "
              f"{part['codeswitch']['en_span_wer']}, {part['runtime_s']} s", flush=True)
        if key == "local":
            trials = barge_in_trials(calls)
            update_results("barge_in_by_dialect", aggregate_barge_in(trials))
            transcripts["barge_in_trials"] = trials
        transcripts["calls"][key] = _strip(calls)
        path.write_text(json.dumps(transcripts, ensure_ascii=False, indent=0), encoding="utf-8")

    update_results("machine", machine())
    update_results("setup", {
        "generated_at": datetime.now(DUBAI).isoformat(timespec="seconds"),
        "calls": len(CALLS), "scripted_lines": sum(len(c["turns"]) for c in CALLS.values()),
        "reference_day": ANCHOR.isoformat(), "vad_hangover_s": HANGOVER_S, "barge_in_bound_s": BARGE_IN_BOUND_S,
        "caller_audio": "synthetic TTS only: Piper en_US-lessac and ar_JO-kareem, Windows SAPI Zira and David",
        "real_audio": "none; Bulbul has no public release found and the NADI 2026 sets on Hugging Face state no licence",
        "dialogue": "rule parser, templates, grounding check",
    })


if __name__ == "__main__":
    main()
