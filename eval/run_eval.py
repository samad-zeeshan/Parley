"""End-to-end evaluation: synthetic callers talk to the agent through ASR, NLU,
policy, the booking API and TTS. Writes eval/results.json and eval/results.md.

    uv run --extra speech python -m eval.run_eval                 # rules NLU + local model NLU
    uv run --extra speech python -m eval.run_eval --configs rules  # no LM Studio needed

Every number is split by language and dialect: en, ar-gulf, ar-msa, mixed.
The caller audio is TTS (Piper, Windows SAPI). No human recordings were used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from api.audit import entries, verify_chain
from api.clock import DUBAI, FixedClock
from api.db import connect
from api.seed import seed
from api.service import BookingService
from dialogue.agent import Agent
from dialogue.nlu import LLMNLU, RuleNLU
from dialogue.phrasing import LLMPhraser, Phraser
from speech.asr import StreamingRecognizer, WhisperASR
from speech.audio import SAMPLE_RATE, read_wav, silence
from speech.duplex import BARGE_IN_BOUND_S, simulate
from speech.langid import identify, reply_language
from speech.pipeline import VoiceCall
from speech.textnorm import normalize_orthography
from speech.tts import PiperTTS
from speech.vad import FRAME, EnergyVAD

from .scoring import DATE_KINDS, DIGIT_KINDS, TIME_KINDS, entity_errors, f1, percentile, slot_counts
from .scripts import CALLS
from .synth import render_to
from .wer import ZERO, count_errors

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
AUDIO = ROOT / "audio" / "calls"
ANCHOR = datetime(2026, 10, 1, 9, 0, tzinfo=DUBAI)
DIALECTS = ["en", "ar-gulf", "ar-msa", "mixed"]
CALL_KINDS = {"en": "English", "gulf": "Gulf Arabic", "msa": "MSA", "switch": "code-switched"}
MAX_TURNS = 12
HANGOVER_S = EnergyVAD().hangover_frames * FRAME / SAMPLE_RATE


# ---------------------------------------------------------------------------
# Stage 1: caller audio and streaming ASR (cached; the same for every config)
# ---------------------------------------------------------------------------

def render_turn(call_id: str, i: int, turn: dict, voice: str, attempt: int = 0) -> Path:
    """Attempt 0 is the scripted voice. A caller asked again repeats the line more slowly,
    as people do: each repeat is 15 percent slower (Piper length scale, SAPI rate)."""
    base = 1.15 if voice == "piper-slow" else 1.0
    v = "piper" if voice == "piper-slow" else voice
    name = f"{i:02d}.wav" if attempt == 0 else f"{i:02d}-r{attempt}.wav"
    if v.startswith("sapi"):
        return render_to(AUDIO / call_id / name, turn["segments"], v, sapi_rate=-2 * attempt)
    scale = base + 0.15 * attempt
    return render_to(AUDIO / call_id / name, turn["segments"], v, length_scale=None if scale == 1.0 else scale)


def stream_asr(asr, pcm: np.ndarray, lang_hint: str | None) -> dict:
    """Play the clip through the VAD-driven recognizer as 20 ms frames."""
    rec = StreamingRecognizer(asr)
    rec.lang_hint = lang_hint
    stream = np.concatenate([silence(0.3), pcm, silence(HANGOVER_S + 0.3)])
    texts, decode, finals = [], 0.0, 0
    for k in range(len(stream) // FRAME):
        for ev in rec.push(stream[k * FRAME:(k + 1) * FRAME]):
            if ev.kind == "final":
                texts.append(ev.text)
                decode += ev.decode_seconds
                finals += 1
    for ev in rec.flush():
        texts.append(ev.text)
        decode += ev.decode_seconds
        finals += 1
    return {"text": " ".join(t for t in texts if t).strip(), "decode_s": decode, "segments": finals}


class ASRCache:
    """Streaming ASR results keyed by audio hash and language hint, stored in eval/out."""

    def __init__(self, size: str):
        self.size = size
        self.path = OUT / f"asr_{size}.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.model = None

    def get(self, call_id: str, line: int, attempt: int, hint: str | None) -> dict:
        call = CALLS[call_id]
        path = render_turn(call_id, line, call["turns"][line], call["voice"], attempt)
        key = f"{call_id}/{line}/{attempt}/{hint}:" + hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        if key not in self.data:
            self.model = self.model or WhisperASR(self.size)
            pcm, _ = read_wav(str(path))
            self.data[key] = stream_asr(self.model, pcm, hint)
            print(f"  asr {call_id}/{line}/{attempt}: {self.data[key]['text'][:70]} "
                  f"({self.data[key]['decode_s']:.2f}s)", flush=True)
        return self.data[key]

    def save(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


def script_hint(call: dict, line: int) -> str | None:
    """Reply language after the previous scripted line: the ASR hint used for the ASR table."""
    if line == 0:
        return None
    return reply_language(identify(call["turns"][line - 1]["text"]))


# ---------------------------------------------------------------------------
# Stage 2: scripted caller against the agent
# ---------------------------------------------------------------------------

def next_line(call: dict, action, last: int) -> int | None:
    """The scripted caller answers the question the agent asked with the matching line."""
    turns = call["turns"]
    name, args = action.name, action.args
    if name == "confirmed":
        return None
    if name == "ask_slot":
        for j, t in enumerate(turns):
            if args["slot"] in t["slots"]:
                return j
    if name in ("offer",):
        for j, t in enumerate(turns):
            if t["intent"] == "choose_option":
                return j
    if name == "ask_confirm":
        return len(turns) - 1
    return last + 1 if last + 1 < len(turns) else None


def run_call(call_id: str, call: dict, asr: ASRCache, config: str, tts: PiperTTS, llm_model: str,
             keep_audio: bool = False) -> dict:
    conn = connect(":memory:")
    clock = FixedClock(ANCHOR)
    seed(conn, anchor=ANCHOR.date(), rng_seed=7)
    svc = BookingService(conn, clock)
    nlu = RuleNLU() if config == "rules" else LLMNLU(model=llm_model, timeout=60)
    phraser = Phraser(llm=LLMPhraser(model=llm_model, timeout=60) if config == "llm+phrasing" else None, clock=clock)
    agent = Agent(svc, conn, clock, session_id=f"{config}:{call_id}", nlu=nlu, phraser=phraser)
    vc = VoiceCall(agent, None, tts)

    turns_out, line, played = [], 0, []
    while line is not None and len(played) < MAX_TURNS:
        attempt = played.count(line)
        played.append(line)
        gold = call["turns"][line]
        a = asr.get(call_id, line, attempt, agent.lang if len(played) > 1 else None)
        reply = vc.respond(a["text"], asr_seconds=a["decode_s"])
        t = reply.turn
        turns_out.append({
            "line": line, "attempt": attempt, "dialect": gold["dialect"], "ref": gold["text"], "hyp": a["text"],
            "asr_segments": a["segments"], "gold_intent": gold["intent"], "intent": t.nlu.intent,
            "gold_slots": {**gold["slots"], **({"choice": gold["choice"]} if gold["choice"] else {})},
            "slots": {**t.nlu.slots, **({"choice": t.nlu.choice} if t.nlu.choice else {})},
            "dialect_pred": t.dialect, "dialect_pred_ref": identify(gold["text"]).label,
            "action": t.action, "reply": t.text, "reply_lang": t.lang, "reply_source": t.source,
            "rejected": [v.kind for v in t.rejected], "ungrounded": [v.kind for v in t.ungrounded],
            "nlu_source": t.nlu.source, "dropped_slots": list(t.nlu.dropped),
            "timings": {k: round(v, 4) for k, v in reply.timings.items()},
            "reply_seconds": round(reply.speech.seconds, 2) if reply.speech is not None else None,
        })
        if keep_audio and reply.speech is not None:
            turns_out[-1]["_speech"] = reply.speech.pcm
        line = next_line(call, _Action(t.action, _args_for(agent)), line)

    booking = conn.execute("select status, phone from bookings").fetchall()
    completed = booking == [("confirmed", call["phone"])] and verify_chain(conn)
    return {"call_id": call_id, "config": config, "completed": completed, "bookings": booking,
            "audit_ok": verify_chain(conn), "audit_entries": len(entries(conn)), "turns": turns_out}


class _Action:
    def __init__(self, name, args):
        self.name, self.args = name, args


def _args_for(agent: Agent) -> dict:
    s = agent.state
    if s.last_spoken == "ask_slot":
        return {"slot": s.last_asked}
    return {}


# ---------------------------------------------------------------------------
# Stage 3: barge-in trials on the real agent and caller audio
# ---------------------------------------------------------------------------

def barge_in_trials(calls_out: list[dict]) -> list[dict]:
    rng = np.random.default_rng(2026)
    trials = []
    for c in calls_out:
        turns = c["turns"]
        for k, t in enumerate(turns[:-1]):
            agent_pcm = t.get("_speech")
            if agent_pcm is None or len(agent_pcm) < 1.5 * SAMPLE_RATE:
                continue
            nxt = turns[k + 1]
            call = CALLS[c["call_id"]]
            caller_pcm, _ = read_wav(str(render_turn(c["call_id"], nxt["line"], call["turns"][nxt["line"]],
                                                     call["voice"], nxt["attempt"])))
            offset = float(rng.uniform(0.3, len(agent_pcm) / SAMPLE_RATE - 1.0))
            caller = np.concatenate([silence(offset), caller_pcm])
            r = simulate(agent_pcm, caller, echo_gain=0.25)
            echo_only = simulate(agent_pcm, silence(len(agent_pcm) / SAMPLE_RATE + 0.5), echo_gain=0.25)
            ok = r.barged_in and not r.false_stop and r.latency_s is not None and r.latency_s <= BARGE_IN_BOUND_S
            trials.append({"call_id": c["call_id"], "dialect": nxt["dialect"], "agent_lang": t["reply_lang"],
                           "offset_s": round(offset, 2), "success": ok,
                           "latency_s": None if r.latency_s is None else round(r.latency_s, 3),
                           "false_stop_on_echo": echo_only.barged_in})
    return trials


# ---------------------------------------------------------------------------
# Stage 4: aggregate
# ---------------------------------------------------------------------------

def _r(x, n=3):
    return None if x is None else round(x, n)


def aggregate_asr(asr: ASRCache) -> dict:
    """ASR and dialect ID over every scripted line once (attempt 0), whether or not a call reached it."""
    from speech.normalize import normalize

    today = ANCHOR.date()
    per = {d: {"raw": ZERO, "norm": ZERO, "digit": ZERO, "date": ZERO, "time": ZERO, "n": 0,
               "lid_hyp": 0, "lid_ref": 0} for d in DIALECTS}
    for call_id, call in CALLS.items():
        for line, t in enumerate(call["turns"]):
            hyp = asr.get(call_id, line, 0, script_hint(call, line))["text"]
            p = per[t["dialect"]]
            p["n"] += 1
            p["raw"] += count_errors(normalize_orthography(t["text"]), normalize_orthography(hyp))
            p["norm"] += count_errors(normalize(t["text"], today).text, normalize(hyp, today).text)
            p["digit"] += entity_errors(t["text"], hyp, DIGIT_KINDS, today)
            p["date"] += entity_errors(t["text"], hyp, DATE_KINDS, today)
            p["time"] += entity_errors(t["text"], hyp, TIME_KINDS, today)
            p["lid_hyp"] += identify(hyp).label == t["dialect"]
            p["lid_ref"] += identify(t["text"]).label == t["dialect"]
    asr_out, lid = {}, {}
    for d, p in per.items():
        asr_out[d] = {
            "utterances": p["n"],
            "wer": _r(p["raw"].wer), "wer_normalized": _r(p["norm"].wer), "ref_words": p["raw"].ref_words,
            "digit_wer": _r(p["digit"].wer) if p["digit"].ref_words else None, "digit_tokens": p["digit"].ref_words,
            "date_wer": _r(p["date"].wer) if p["date"].ref_words else None, "date_tokens": p["date"].ref_words,
            "time_wer": _r(p["time"].wer) if p["time"].ref_words else None, "time_tokens": p["time"].ref_words,
        }
        lid[d] = {"utterances": p["n"], "accuracy_on_asr_transcript": _r(p["lid_hyp"] / p["n"]) if p["n"] else None,
                  "accuracy_on_reference_text": _r(p["lid_ref"] / p["n"]) if p["n"] else None}
    return {"asr": asr_out, "dialect_id": lid}


def aggregate_config(calls_out: list[dict]) -> dict:
    per = {d: {"n": 0, "intent_ok": 0, "tp": 0, "fp": 0, "fn": 0, "lat": [], "asr": [], "dlg": [], "tts": []}
           for d in DIALECTS}
    rejected = llm_replies = dropped = fallbacks = ungrounded_spoken = 0
    for c in calls_out:
        for t in c["turns"]:
            p = per[t["dialect"]]
            p["n"] += 1
            p["intent_ok"] += t["intent"] == t["gold_intent"]
            tp, fp, fn = slot_counts(t["gold_slots"], t["slots"])
            p["tp"] += tp
            p["fp"] += fp
            p["fn"] += fn
            p["lat"].append(t["timings"]["total"])
            p["asr"].append(t["timings"]["asr"])
            p["dlg"].append(t["timings"]["dialogue"])
            p["tts"].append(t["timings"]["tts"])
            rejected += bool(t["rejected"])
            llm_replies += t["reply_source"] == "llm"
            dropped += len(t["dropped_slots"])
            fallbacks += t["nlu_source"] == "rules-fallback"
            ungrounded_spoken += bool(t["ungrounded"])
    by = {}
    for d, p in per.items():
        n = p["n"]
        by[d] = {
            "turns": n,
            "intent_accuracy": _r(p["intent_ok"] / n) if n else None,
            "slot_f1": _r(f1(p["tp"], p["fp"], p["fn"])),
            "latency_p50_s": _r(percentile(p["lat"], 50), 2) if n else None,
            "latency_p95_s": _r(percentile(p["lat"], 95), 2) if n else None,
            "asr_p50_s": _r(percentile(p["asr"], 50), 2) if n else None,
            "dialogue_p50_s": _r(percentile(p["dlg"], 50), 3) if n else None,
            "tts_p50_s": _r(percentile(p["tts"], 50), 2) if n else None,
        }
    all_lat = [x for p in per.values() for x in p["lat"]]
    tasks = {}
    for kind in CALL_KINDS:
        cs = [c for c in calls_out if c["call_id"].split("-")[0] == kind]
        tasks[kind] = {"completed": sum(c["completed"] for c in cs), "calls": len(cs),
                       "rate": _r(sum(c["completed"] for c in cs) / len(cs)) if cs else None,
                       "turns_per_call": [len(c["turns"]) for c in cs]}
    return {
        "by_dialect": by,
        "task_completion_by_call_language": tasks,
        "latency_all_turns": {"p50_s": _r(percentile(all_lat, 50), 2), "p95_s": _r(percentile(all_lat, 95), 2),
                              "turns": len(all_lat), "p95_over_1_5s": percentile(all_lat, 95) > 1.5},
        "llm_replies_spoken": llm_replies, "llm_replies_rejected_by_grounding": rejected,
        "model_slots_dropped_without_evidence": dropped, "nlu_fallbacks_to_rules": fallbacks,
        "spoken_replies_with_ungrounded_facts": ungrounded_spoken,
    }


def aggregate_barge_in(trials: list[dict]) -> dict:
    out = {}
    for d in DIALECTS:
        ts = [t for t in trials if t["dialect"] == d]
        lats = [t["latency_s"] for t in ts if t["latency_s"] is not None]
        out[d] = {"trials": len(ts), "success": sum(t["success"] for t in ts),
                  "rate": _r(sum(t["success"] for t in ts) / len(ts)) if ts else None,
                  "latency_max_s": _r(max(lats)) if lats else None,
                  "latency_p50_s": _r(percentile(lats, 50)) if lats else None,
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


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default=os.environ.get("MAJLIS_ASR_MODEL", "small"))
    ap.add_argument("--configs", default="rules,llm,llm+phrasing")
    ap.add_argument("--llm", default=os.environ.get("MAJLIS_LLM_MODEL", "qwen/qwen3.5-9b"))
    args = ap.parse_args()
    configs = args.configs.split(",")

    t0 = time.time()
    print(f"stage 1: caller audio and streaming ASR ({args.asr}) for every scripted line", flush=True)
    asr = ASRCache(args.asr)
    asr_lid = aggregate_asr(asr)
    asr.save()
    tts = PiperTTS()
    tts.synthesize("warm up", "en")
    tts.synthesize("تجربة", "ar")

    results_calls, per_config = {}, {}
    for config in configs:
        print(f"stage 2: scripted calls, config={config}", flush=True)
        outs = [run_call(cid, call, asr, config, tts, args.llm, keep_audio=(config == "rules"))
                for cid, call in CALLS.items()]
        asr.save()
        for o in outs:
            print(f"  {o['call_id']}: completed={o['completed']} turns={len(o['turns'])}", flush=True)
        results_calls[config] = outs
        per_config[config] = aggregate_config(outs)

    print("stage 3: barge-in trials", flush=True)
    trials = barge_in_trials(results_calls["rules"])

    results = {
        "generated_at": datetime.now(DUBAI).isoformat(timespec="seconds"),
        "machine": machine(),
        "setup": {
            "asr": f"faster-whisper {args.asr}, int8, CPU, greedy, bilingual domain prompt, no language hint",
            "nlu_configs": {"rules": "deterministic rule parser",
                            "llm": f"{args.llm} via LM Studio (GGUF), reasoning off, evidence check on slots",
                            "llm+phrasing": f"llm NLU plus {args.llm} rewording replies, grounding check"},
            "tts": "Piper en_US-lessac-medium and ar_JO-kareem-medium; callers also Windows SAPI Zira",
            "caller_audio": "synthetic TTS only; Gulf lines are Gulf wording in a Jordanian Piper voice",
            "caller_repeats": "asked again, the scripted caller repeats the same line 15 percent slower",
            "calls": len(CALLS), "scripted_lines": sum(len(c["turns"]) for c in CALLS.values()),
            "reference_day": ANCHOR.isoformat(), "vad_hangover_s": HANGOVER_S,
            "barge_in_bound_s": BARGE_IN_BOUND_S, "barge_in_echo_gain": 0.25,
        },
        "asr_by_dialect": asr_lid["asr"],
        "dialect_id_by_dialect": asr_lid["dialect_id"],
        "configs": per_config,
        "barge_in_by_dialect": aggregate_barge_in(trials),
        "runtime_s": round(time.time() - t0, 1),
    }
    (ROOT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    detail = {k: [{**c, "turns": [{kk: vv for kk, vv in t.items() if not kk.startswith("_")} for t in c["turns"]]}
                  for c in v] for k, v in results_calls.items()}
    (ROOT / "transcripts.json").write_text(json.dumps({"calls": detail, "barge_in_trials": trials}, ensure_ascii=False,
                                               indent=1), encoding="utf-8")
    from .report import write_markdown

    write_markdown(results, ROOT / "results.md")
    print(f"done in {results['runtime_s']} s", flush=True)
    return results


if __name__ == "__main__":
    main()
