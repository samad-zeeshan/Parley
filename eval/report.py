"""Render eval/results.json, with v1 numbers from eval/results_v1.json, as the README's results tables.

    uv run python -m eval.report --check    # exit 1 if README.md does not contain the current tables
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"
KINDS = {"en": "English", "gulf": "Gulf Arabic", "msa": "MSA", "switch": "Code-switched"}
NOT_RUN = "not run"
SPEECH = [("local-v1", "local"), ("local-v1|rules-fuzzy", "local, fuzzy parser"), ("local", "mixed prompt"),
          ("local-2pass", "two-pass decode"), ("hosted", "hosted")]
ASR_ONLY = [c for c in SPEECH if "|" not in c[0]]


def _f(x, n=3):
    if x is None:
        return "n/a"
    return f"{x:.{n}f}" if isinstance(x, float) else str(x)


def _frac(a, b):
    return f"{a}/{b}"


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(head, rows) -> list[str]:
    return [_row(head), "|" + "---|" * len(head), *(_row(r) for r in rows)]


def calls_table(r: dict, v1: dict) -> list[str]:
    sp = r.get("speech", {})
    v1t = v1["configs"]["rules"]["task_completion_by_call_language"]
    head = ["calls booked", "v1 (8 calls)", *(name for _, name in SPEECH)]
    rows = []
    for k, name in KINDS.items():
        cells = [name, _frac(v1t[k]["completed"], v1t[k]["calls"])]
        for cfg, _ in SPEECH:
            d = sp.get(cfg, {}).get("dialogue")
            t = d["task_completion_by_call_language"][k] if d else None
            cells.append(_frac(t["completed"], t["calls"]) if t else NOT_RUN)
        rows.append(cells)
    total = ["All", _frac(sum(v["completed"] for v in v1t.values()), sum(v["calls"] for v in v1t.values()))]
    wrong = ["Wrong actions", "n/a"]
    false = ["Replies with an ungrounded fact", str(v1["configs"]["rules"]["spoken_replies_with_ungrounded_facts"])]
    for cfg, _ in SPEECH:
        d = sp.get(cfg, {}).get("dialogue")
        total.append(_frac(d["completed"], d["calls"]) if d else NOT_RUN)
        wrong.append(d["wrong_actions"] if d else NOT_RUN)
        false.append(d["spoken_replies_with_ungrounded_facts"] if d else NOT_RUN)
    return _table(head, rows + [total, wrong, false])


def codeswitch_line(r: dict) -> str:
    sp = r.get("speech", {})

    def row(key):
        return ", ".join(f"{name} {_f(sp[c]['codeswitch'][key]) if c in sp else NOT_RUN}" for c, name in ASR_ONLY)
    return (f"Code-switched lines scored per language as in arXiv 2605.19069. WER on the English words: {row('en_span_wer')}. "
            f"WER on the Arabic words: {row('ar_span_wer')}. English words kept in Latin script: {row('en_kept_latin')}.")


def mtva_table(r: dict) -> list[str]:
    runs = r.get("mtva", {}).get("runs", {})
    conds = [("reference", "reference text"), ("asr:local", "local ASR text"), ("asr:hosted", "hosted ASR text"),
             ("noise@0.05", "5% injected errors"), ("noise@0.1", "10% injected errors"),
             ("noise@0.2", "20% injected errors"), ("noise@0.3", "30% injected errors"), ("split", "split turns")]
    head = ["transcript", "transcript WER", "rules intent", "rules slot F1", "rules booked", "rules, no carry-over booked",
            "Jev intent", "Jev booked", "9B intent", "9B booked", "reply language kept"]
    rows = []
    for c, label in conds:
        def g(model, key, fmt=_f):
            x = runs.get(f"{model}|{c}")
            if not x or x.get("not_run"):
                return NOT_RUN
            return fmt(x[key]) if key != "completed" else _frac(x["completed"], x["calls"])
        base = runs.get(f"rules|{c}")
        wer = NOT_RUN if not base or base.get("not_run") else _f(base["transcript_wer"])
        rows.append([label, wer, g("rules", "intent_accuracy"), g("rules", "slot_f1"), g("rules", "completed"),
                     g("rules-nocarry", "completed"), g("jev", "intent_accuracy"), g("jev", "completed"),
                     g("llm", "intent_accuracy"), g("llm", "completed"), g("rules", "reply_language_adherence")])
    return _table(head, rows)


def phrasing_line(r: dict) -> str:
    runs = r.get("mtva", {}).get("runs", {})
    parts = []
    for c, label in [("reference", "reference text"), ("asr:local", "local ASR text")]:
        x = runs.get(f"llm+phrasing|{c}")
        if x and not x.get("not_run"):
            parts.append(f"on {label} {x['model_replies_spoken']} model replies spoken, "
                         f"{x['model_replies_rejected']} rejected, reply language kept {_f(x['reply_language_adherence'])}, "
                         f"{x['spoken_replies_with_ungrounded_facts']} replies with an ungrounded fact")
    return ("With the 9B model rewording replies, " + ". Then ".join(parts) + ".") if parts else ""


def trace_table(r: dict) -> list[str]:
    arms = r.get("trace", {}).get("by_arm", {})
    head = ["arm (local)", *KINDS.values(), "wrong actions", "recovered after trouble", "extra turns", "hosted"]
    rows = []
    for arm, row in arms.items():
        wrong = sum(v["wrong_actions"] for v in row.values())
        rec = _frac(sum(v["recovered"] for v in row.values()), sum(v["troubled"] for v in row.values()))
        extra = [v["extra_turns_vs_clean"] for v in row.values() if v["extra_turns_vs_clean"] is not None]
        rows.append([arm, *(_frac(row[k]["completed"], row[k]["calls"]) for k in KINDS), wrong, rec,
                     _f(sum(extra) / len(extra), 1) if extra else "n/a", NOT_RUN])
    return _table(head, rows)


def latency_table(r: dict, v1: dict) -> list[str]:
    lat = r.get("latency", {}).get("by_config", {})
    head = ["speech config", "booked", "p50 (s)", "p95 (s)", "p95 without early decode (s)"]
    v = v1["configs"]["rules"]["latency_all_turns"]
    v1_booked = sum(x["completed"] for x in v1["configs"]["rules"]["task_completion_by_call_language"].values())
    rows = [["v1 (whisper small)", _frac(v1_booked, 8), _f(v["p50_s"], 2), "n/a", _f(v["p95_s"], 2)]]
    names = {"local-v1": "local (whisper small)", "local": "mixed prompt", "local-2pass": "two-pass decode",
             "local-base": "whisper base", "local-tiny": "whisper tiny", "hosted": "hosted"}
    for cfg, name in names.items():
        x = lat.get(cfg)
        if not x:
            rows.append([name, *[NOT_RUN] * 4])
            continue
        rows.append([name, _frac(x["completed"], x["calls"]), _f(x["early_decode"]["p50_s"], 2),
                     _f(x["early_decode"]["p95_s"], 2), _f(x["after_close"]["p95_s"], 2)])
    return _table(head, rows)


def toolnoise_line(r: dict) -> str:
    modes = r.get("tool_noise", {}).get("modes", {})
    if not modes:
        return f"Booking API faults: {NOT_RUN}."
    parts = [f"{m} {_frac(x['completed'], x['calls'])}" for m, x in modes.items()]
    false = sum(x["false_facts_spoken"] for x in modes.values())
    safe = sum(x["safe_recoveries"] for x in modes.values())
    hit = sum(x["calls_hitting_a_fault"] for x in modes.values())
    cb = sum(x["callbacks_queued"] for x in modes.values())
    return (f"Booking API faults on reference text (arXiv 2608.02372, 2606.31307), calls booked: {', '.join(parts)}. "
            f"False facts spoken in all modes: {false}. Calls that hit a fault and were handled safely: "
            f"{_frac(safe, hit)}. Callbacks queued: {cb}. Hosted: {NOT_RUN}.")


def rubric_sentence(r: dict) -> str:
    rb = r.get("rubric")
    if not rb:
        return f"Gulf dialect rubric: {NOT_RUN}."
    t, m, c = rb["templates"], rb["model_rewording"], rb["calibration"]
    pens = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(m["penalties"].items(), key=lambda kv: -kv[1]))
    return (f"Gulf dialect rubric (arXiv 2608.29990), calibrated score out of 1: agent templates {_f(t['mean_score_calibrated'])} "
            f"over {t['replies']} replies, 9B rewordings {_f(m['mean_score_calibrated'])} over {m['replies']} "
            f"(penalties: {pens}). Judge {rb['judge']} agrees with {c['anchors']} labelled anchors on "
            f"{_f(c['criterion_agreement'])} of criteria. Native-speaker spot check "
            f"{'done' if rb['spot_check']['done'] else 'not done yet'}.")


def bargein_line(r: dict, v1: dict) -> str:
    b, b1 = r.get("barge_in_by_dialect", {}), v1["barge_in_by_dialect"]
    names = {"en": "English", "ar-gulf": "Gulf", "ar-msa": "MSA", "mixed": "code-switched"}
    parts = [f"{n} {_frac(b[d]['success'], b[d]['trials'])} (v1 {_frac(b1[d]['success'], b1[d]['trials'])})"
             for d, n in names.items() if d in b]
    worst = max((b[d]["latency_max_s"] for d in names if d in b), default=None)
    echo = sum(b[d]["false_stops_on_echo_only"] for d in names if d in b)
    return (f"Barge-in, agent stopped within 0.2 s of the caller: {', '.join(parts)}. Slowest stop {_f(worst, 2)} s, "
            f"and false stops on the agent's own echo: {echo}.")


def tables(r: dict, v1: dict) -> str:
    out = ["Calls booked by the scripted callers, rule parser:", "", *calls_table(r, v1), "",
           codeswitch_line(r), "",
           "The dialogue layer alone on the same calls, MTVA style (arXiv 2609.20152):", "", *mtva_table(r), ""]
    line = phrasing_line(r)
    if line:
        out += [line, ""]
    out += ["Acoustic stress on the local config, TRACE style (arXiv 2609.29452), plot in `eval/trace.svg`:", "",
            *trace_table(r), "",
            "Turn latency on the development CPU, all turns, target 1.5 s. Timed from when the caller's 0.5 s pause "
            "closes the turn. Early decode starts recognition on the first quiet frame:", "",
            *latency_table(r, v1), "", toolnoise_line(r), "", rubric_sentence(r), "", bargein_line(r, v1)]
    return "\n".join(out) + "\n"


def load() -> tuple[dict, dict]:
    r = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    v1 = json.loads((ROOT / "results_v1.json").read_text(encoding="utf-8"))
    return r, v1


def check_readme() -> bool:
    readme = (ROOT.parent / "README.md").read_text(encoding="utf-8")
    if BEGIN not in readme or END not in readme:
        return False
    block = readme.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    return block == tables(*load()).strip()


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        ok = check_readme()
        print("README matches eval/results.json" if ok else "README tables differ from eval/results.json")
        sys.exit(0 if ok else 1)
    print(tables(*load()))
