"""Render eval/results.json as markdown. The README table is this output, pasted.

    uv run python -m eval.report            # rewrite eval/results.md
    uv run python -m eval.report --check    # exit 1 if README.md does not contain the current tables
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {"en": "English", "ar-gulf": "Gulf Arabic", "ar-msa": "MSA", "mixed": "Code-switched"}
CALL_NAMES = {"en": "English", "gulf": "Gulf Arabic", "msa": "MSA", "switch": "Code-switched"}
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"


def _f(x, n=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{n}f}" if isinstance(x, float) else str(x)


def tables(r: dict) -> str:
    out = []
    out.append("ASR (faster-whisper " + r["setup"]["asr"].split(",")[0].split()[-1] + ", CPU) and dialect ID, "
               "per scripted line:\n")
    out.append("| language / dialect | lines | WER | WER after number normalization | digit WER | date WER "
               "| time WER | dialect ID on ASR text | dialect ID on reference text |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for d, name in NAMES.items():
        a, lid = r["asr_by_dialect"][d], r["dialect_id_by_dialect"][d]
        out.append(f"| {name} | {a['utterances']} | {_f(a['wer'])} | {_f(a['wer_normalized'])} | "
                   f"{_f(a['digit_wer'])} ({a['digit_tokens']}) | {_f(a['date_wer'])} ({a['date_tokens']}) | "
                   f"{_f(a['time_wer'])} ({a['time_tokens']}) | {_f(lid['accuracy_on_asr_transcript'])} | "
                   f"{_f(lid['accuracy_on_reference_text'])} |")
    out.append("\nThe count in brackets is the number of reference tokens the digit, date or time WER is over.\n")

    for cfg, c in r["configs"].items():
        out.append(f"Dialogue, config `{cfg}` ({r['setup']['nlu_configs'][cfg]}):\n")
        out.append("| language / dialect | turns | intent accuracy | slot F1 | intent accuracy, reference text "
                   "| slot F1, reference text | turn latency p50 (s) | turn latency p95 (s) | ASR p50 (s) "
                   "| dialogue p50 (s) | TTS p50 (s) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for d, name in NAMES.items():
            b = c["by_dialect"][d]
            out.append(f"| {name} | {b['turns']} | {_f(b['intent_accuracy'])} | {_f(b['slot_f1'])} | "
                       f"{_f(b['intent_accuracy_on_reference_text'])} | {_f(b['slot_f1_on_reference_text'])} | "
                       f"{_f(b['latency_p50_s'], 2)} | {_f(b['latency_p95_s'], 2)} | {_f(b['asr_p50_s'], 2)} | "
                       f"{_f(b['dialogue_p50_s'], 3)} | {_f(b['tts_p50_s'], 2)} |")
        tc = c["task_completion_by_call_language"]
        comp = ", ".join(f"{CALL_NAMES[k]} {v['completed']}/{v['calls']}" for k, v in tc.items())
        lat = c["latency_all_turns"]
        out.append(f"\nTask completion (booking in the database with the caller's phone and a valid audit chain): "
                   f"{comp}. All turns: p50 {_f(lat['p50_s'], 2)} s, p95 {_f(lat['p95_s'], 2)} s"
                   f"{' (above the 1.5 s target)' if lat['p95_over_1_5s'] else ''}. "
                   f"Replies spoken with an ungrounded fact: {c['spoken_replies_with_ungrounded_facts']}. "
                   f"Model slots dropped for lack of evidence: {c['model_slots_dropped_without_evidence']}. "
                   f"NLU fallbacks to rules: {c['nlu_fallbacks_to_rules']}. "
                   f"Model replies spoken: {c['llm_replies_spoken']}; rejected by the grounding check: "
                   f"{c['llm_replies_rejected_by_grounding']}.\n")

    out.append(f"Barge-in (caller line played over the agent's real Piper reply, agent echo in the mic at "
               f"{r['setup']['barge_in_echo_gain']} gain, success means stopped within "
               f"{r['setup']['barge_in_bound_s']} s of the caller's first voiced frame):\n")
    out.append("| interrupting caller | trials | success | success rate | latency p50 (s) | latency max (s) "
               "| false stops on echo alone |")
    out.append("|---|---|---|---|---|---|---|")
    for d, name in NAMES.items():
        b = r["barge_in_by_dialect"][d]
        out.append(f"| {name} | {b['trials']} | {b['success']} | {_f(b['rate'])} | {_f(b['latency_p50_s'])} | "
                   f"{_f(b['latency_max_s'])} | {b['false_stops_on_echo_only']} |")
    return "\n".join(out) + "\n"


def write_markdown(results: dict, path: Path) -> None:
    head = (f"# Evaluation results\n\nGenerated {results['generated_at']} on {results['machine']['cpu']} "
            f"({results['machine']['logical_cores']} logical cores, no GPU). "
            f"Source: `eval/results.json`. Caller audio: {results['setup']['caller_audio']}.\n\n")
    path.write_text(head + tables(results), encoding="utf-8")


def check_readme() -> bool:
    r = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    readme = (ROOT.parent / "README.md").read_text(encoding="utf-8")
    if BEGIN not in readme or END not in readme:
        return False
    block = readme.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    return block == tables(r).strip()


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        ok = check_readme()
        print("README matches eval/results.json" if ok else "README tables differ from eval/results.json")
        sys.exit(0 if ok else 1)
    write_markdown(json.loads((ROOT / "results.json").read_text(encoding="utf-8")), ROOT / "results.md")
