"""Fit Jev head temperatures on a held-out slice, pick the cascade threshold, and time the model.

    uv run --extra jev python -m eval.jev_calibrate [Qwen/Qwen3-0.6B ...] [--write]

The labelled items (eval/jev_calib_set.py and the ASR smoke set) are split in two by
alternation: the fit slice sets each head's temperature and the threshold, the test
slice reports accuracy and expected calibration error (ECE) before and after scaling.
With --write, the first model's result is saved to dialogue/jev_calibration.json,
which JevDecider reads. Every model's numbers go to eval/jev_smoke.json.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

from dialogue.jev import (
    CALIBRATION_FILE,
    HEADS,
    LABELS,
    HFLogitModel,
    expected_calibration_error,
    facts_text,
    fit_temperature,
    softmax,
)

from .jev_calib_set import DIALECT, INTENT, YES_NO, judge_items
from .smoke_set import SMOKE_SET, reference_text

ROOT = Path(__file__).resolve().parent
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
TARGET_ACCEPTED_ACCURACY = 0.9
_SMOKE_DIALECT = {"en": "english", "ar-gulf": "gulf", "ar-msa": "msa", "mixed": "switch"}


def items() -> dict[str, list[tuple[dict, str]]]:
    today = date(2026, 10, 1)  # noqa: F841 -- judge items use the same reference day as the evaluation
    return {
        "intent": [({"text": t, "context": c}, y) for t, c, y in INTENT],
        "yes_no": [({"text": t, "question": "Shall I confirm the booking?"}, y) for t, y in YES_NO],
        "dialect": [({"text": t}, y) for t, y in DIALECT]
                   + [({"text": reference_text(it)}, _SMOKE_DIALECT[it["dialect"]]) for it in SMOKE_SET],
        "judge": [({"reply": r, "facts": facts_text(s)}, y) for r, s, y in judge_items()],
    }


def run_model(name: str) -> dict:
    t0 = time.perf_counter()
    model = HFLogitModel(name)
    load_s = time.perf_counter() - t0
    out = {"model": name, "load_s": round(load_s, 1), "heads": {}}
    fit_all, pass_times = [], []
    for head, rows in items().items():
        opts = HEADS[head].options
        scored = []
        for kw, gold in rows:
            t = time.perf_counter()
            lg = model.label_logits(head, HEADS[head].prompt(**kw), list(LABELS[:len(opts)]))
            pass_times.append(time.perf_counter() - t)
            scored.append((lg, opts.index(gold)))
        fit, test = scored[0::2], scored[1::2]
        temp = fit_temperature(fit)

        def pairs(rows_, tt):
            res = []
            for lg, y in rows_:
                p = softmax(lg, tt)
                k = max(range(len(p)), key=p.__getitem__)
                res.append((p[k], k == y))
            return res

        before, after = pairs(test, 1.0), pairs(test, temp)
        out["heads"][head] = {
            "temperature": temp, "fit_n": len(fit), "test_n": len(test),
            "test_accuracy": round(sum(ok for _, ok in after) / len(after), 3),
            "test_ece_before": round(expected_calibration_error(before), 3),
            "test_ece_after": round(expected_calibration_error(after), 3),
        }
        fit_all += pairs(fit, temp)
    # One threshold for every head: the lowest that keeps accepted decisions on the fit slice at or
    # above the target accuracy. If none does, the highest is used and most decisions escalate.
    chosen, table = THRESHOLDS[-1], []
    for th in THRESHOLDS:
        acc = [ok for p, ok in fit_all if p >= th]
        rate = len(acc) / len(fit_all)
        accuracy = sum(acc) / len(acc) if acc else None
        table.append({"threshold": th, "accepted_share": round(rate, 3),
                      "accepted_accuracy": None if accuracy is None else round(accuracy, 3)})
    for row in table:
        if row["accepted_accuracy"] is not None and row["accepted_accuracy"] >= TARGET_ACCEPTED_ACCURACY:
            chosen = row["threshold"]
            break
    pass_times.sort()
    out.update({"threshold": chosen, "threshold_table_fit_slice": table,
                "forward_pass_p50_s": round(pass_times[len(pass_times) // 2], 3),
                "forward_pass_p95_s": round(pass_times[int(0.95 * (len(pass_times) - 1))], 3),
                "passes": len(pass_times)})
    return out


def main(argv: list[str]) -> None:
    write = "--write" in argv
    names = [a for a in argv if not a.startswith("--")] or ["Qwen/Qwen3-0.6B"]
    path = ROOT / "jev_smoke.json"
    smoke = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    first = None
    for name in names:
        r = run_model(name)
        smoke[name] = r
        first = first or r
        print(json.dumps(r, indent=1), flush=True)
    path.write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    if write:
        CALIBRATION_FILE.write_text(json.dumps(first, indent=2), encoding="utf-8")
        print(f"wrote {CALIBRATION_FILE}")


if __name__ == "__main__":
    main(sys.argv[1:])
