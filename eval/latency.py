"""Turn latency three ways, and the latency table per speech config.

    uv run python -m eval.latency      # recompute the table from eval/transcripts.json
"""

from __future__ import annotations

import json

from .harness import HANGOVER_S, ROOT, update_results
from .scoring import percentile

MODES = ("after_close", "early_decode", "from_speech_end")
TARGET_S = 1.5


def turn_latency(turn: dict, mode: str) -> float:
    """after_close is v1's number: every decode starts once the hangover has closed the turn.

    early_decode starts the final decode on the first quiet frame, so the hangover hides part of it.
    from_speech_end adds the hangover back: what the caller waits after they stop talking.
    """
    total = turn["timings"]["total"]
    if mode == "after_close":
        return total
    last, hidden = (turn["decodes"][-1] if turn["decodes"] else (0.0, 0.0))
    early = total - min(last, hidden)
    return early if mode == "early_decode" else early + HANGOVER_S


def table(calls_by_config: dict) -> dict:
    out = {}
    for cfg, calls in calls_by_config.items():
        turns = [t for c in calls for t in c["turns"]]
        row = {"turns": len(turns), "completed": sum(c["completed"] for c in calls), "calls": len(calls)}
        for m in MODES:
            xs = [turn_latency(t, m) for t in turns]
            row[m] = {"p50_s": round(percentile(xs, 50), 2), "p95_s": round(percentile(xs, 95), 2)}
        row["asr_p50_s"] = round(percentile([t["timings"]["asr"] for t in turns], 50), 2)
        row["tts_p50_s"] = round(percentile([t["timings"]["tts"] for t in turns], 50), 2)
        row["meets_target"] = row["early_decode"]["p95_s"] < TARGET_S
        out[cfg] = row
    return out


def main() -> dict:
    tr = json.loads((ROOT / "transcripts.json").read_text(encoding="utf-8"))
    t = table(tr["calls"])
    update_results("latency", {"target_s": TARGET_S, "hangover_s": HANGOVER_S, "by_config": t})
    for cfg, row in t.items():
        print(cfg, row["early_decode"], row["after_close"])
    return t


if __name__ == "__main__":
    main()
