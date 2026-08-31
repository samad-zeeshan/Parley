"""The dialogue layer scored the MTVA way (arXiv 2609.20152): the same forty calls on reference, ASR and corrupted transcripts.

    uv run --extra jev python -m eval.mtva
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

from speech.textnorm import normalize_orthography

from .harness import ANCHOR, DIALECTS, OUT, ASRCache, jev_decider, load_results, reference_source, run_call, \
    update_results
from .scoring import f1, slot_counts
from .scripts import CALLS
from .wer import ZERO, count_errors

CONDITIONS = ["reference", "asr:local", "asr:hosted", "noise@0.05", "noise@0.1", "noise@0.2", "noise@0.3", "split"]
# The 9B model with rewording costs two model calls a turn, so it runs on the two conditions that matter most.
PLAN = {
    "rules": CONDITIONS,
    "rules-nocarry": CONDITIONS,   # v1 behaviour: every message parsed on its own
    "jev": CONDITIONS,
    "llm": CONDITIONS,
    "llm+phrasing": ["reference", "asr:local", "asr:hosted"],
}

# Letters ASR confuses in Arabic, chosen so orthographic normalization keeps both sides distinct.
_AR_CONFUSE = {"ح": "خه", "خ": "ح", "ع": "غا", "غ": "ع", "س": "ص", "ص": "س", "ت": "طن", "ط": "ت", "د": "ض",
               "ض": "دظ", "ذ": "ز", "ز": "ذر", "ق": "كج", "ك": "ق", "ب": "تي", "ن": "ت", "ر": "ز", "ي": "ب"}
_EN_CONFUSE = {"a": "e", "e": "i", "i": "e", "o": "u", "u": "o", "b": "p", "p": "b", "d": "t", "t": "d", "s": "z",
               "z": "s", "v": "f", "f": "v", "m": "n", "n": "m", "r": "l", "l": "r", "g": "k", "k": "g"}
_TRANSLIT = dict(zip("abcdefghijklmnopqrstuvwxyz", "ابكديفجهيجكلمنوبكرستوفوكيز"))
_FILLERS_EN, _FILLERS_AR = ["uh", "the", "and"], ["يعني", "و", "آه"]


def _is_ar(w: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in w)


def _substitute(w: str, arabic_context: bool, rng: random.Random) -> str:
    if not _is_ar(w) and arabic_context and rng.random() < 0.5:
        # English inside an Arabic sentence written in Arabic letters, as Whisper did with "Sunday".
        return "".join(_TRANSLIT.get(c, "") for c in w.lower()) or w
    table = _AR_CONFUSE if _is_ar(w) else _EN_CONFUSE
    spots = [i for i, c in enumerate(w) if c.lower() in table]
    if spots:
        i = rng.choice(spots)
        return w[:i] + rng.choice(table[w[i].lower()]) + w[i + 1:]
    return w[:-1] if len(w) > 2 else w + w[-1]


def _rng(key: str, rate: float) -> random.Random:
    return random.Random(int(hashlib.sha1(f"{key}|{rate}".encode()).hexdigest()[:12], 16))


def inject(text: str, rate: float, key: str) -> str:
    """Corrupt each word with probability `rate`: 60 percent substitution, 30 deletion, 10 insertion."""
    if rate <= 0:
        return text
    rng = _rng(key, rate)
    arabic = _is_ar(text)
    out = []
    for w in text.split():
        if rng.random() >= rate:
            out.append(w)
            continue
        r = rng.random()
        if r < 0.6:
            out.append(_substitute(w, arabic, rng))
        elif r < 0.9:
            continue
        else:
            out += [w, rng.choice(_FILLERS_AR if arabic else _FILLERS_EN)]
    return " ".join(out)


def realized_wer(lines: list[str], rate: float) -> float:
    total = ZERO
    for i, t in enumerate(lines):
        total += count_errors(normalize_orthography(t), normalize_orthography(inject(t, rate, key=str(i))))
    return total.wer


def source_for(condition: str):
    if condition in ("reference", "split"):
        return reference_source
    if condition == "asr:local":
        return ASRCache("local").get
    rate = float(condition.split("@")[1])

    def noisy(call_id, line, attempt, hint):
        text = CALLS[call_id]["turns"][line]["text"]
        return {"text": inject(text, rate, key=f"{call_id}/{line}/{attempt}"), "decode_s": 0.0, "decodes": [],
                "segments": 1}
    return noisy


def group_lines(turns: list[dict]) -> list[list[dict]]:
    """Fragments of one split caller turn belong together; every other turn stands alone."""
    groups: list[list[dict]] = []
    for t in turns:
        if t.get("fragment") and groups and (groups[-1][-1]["line"], groups[-1][-1]["attempt"]) == (t["line"],
                                                                                                   t["attempt"]):
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def aggregate(calls: list[dict]) -> dict:
    per = {d: {"n": 0, "ok": 0, "tp": 0, "fp": 0, "fn": 0} for d in DIALECTS}
    adherent = replies = 0
    werc = ZERO
    for c in calls:
        for g in group_lines(c["turns"]):
            p = per[g[0]["dialect"]]
            p["n"] += 1
            # A split line is right if one of its messages carries the caller's intent, and its slots
            # are what the messages found together.
            p["ok"] += any(t["intent"] == t["gold_intent"] for t in g)
            slots = {}
            for t in g:
                slots.update(t["slots"])
            tp, fp, fn = slot_counts(g[0]["gold_slots"], slots)
            p["tp"], p["fp"], p["fn"] = p["tp"] + tp, p["fp"] + fp, p["fn"] + fn
            werc += count_errors(normalize_orthography(g[0]["ref"]),
                                 normalize_orthography(" ".join(t["hyp"] for t in g)))
        for t in c["turns"]:
            replies += 1
            adherent += t["adherent"]
    n = sum(p["n"] for p in per.values())
    ok = sum(p["ok"] for p in per.values())
    tp, fp, fn = (sum(p[k] for p in per.values()) for k in ("tp", "fp", "fn"))
    return {
        "caller_turns": n, "replies": replies,
        "transcript_wer": round(werc.wer, 3),
        "intent_accuracy": round(ok / n, 3), "slot_f1": round(f1(tp, fp, fn), 3),
        "reply_language_adherence": round(adherent / replies, 3),
        "by_dialect": {d: {"turns": p["n"], "intent_accuracy": round(p["ok"] / p["n"], 3) if p["n"] else None,
                           "slot_f1": None if f1(p["tp"], p["fp"], p["fn"]) is None
                           else round(f1(p["tp"], p["fp"], p["fn"]), 3)} for d, p in per.items()},
        "completed": sum(c["completed"] for c in calls), "calls": len(calls),
        "wrong_actions": sum(len(c["wrong_actions"]) for c in calls),
        "spoken_replies_with_ungrounded_facts": sum(c["spoken_ungrounded"] for c in calls),
        "model_replies_spoken": sum(t["reply_source"] == "llm" for c in calls for t in c["turns"]),
        "model_replies_rejected": sum(bool(t["rejected"]) for c in calls for t in c["turns"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogue", default=",".join(PLAN))
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--llm", default="qwen/qwen3.5-9b")
    args = ap.parse_args()
    done = load_results().get("mtva", {}).get("runs", {})
    for dialogue in args.dialogue.split(","):
        if dialogue.startswith("jev"):
            jev_decider().decide("yes_no", text="yes", question="ok?")
        conds = args.conditions.split(",") if args.conditions else PLAN[dialogue]
        for cond in conds:
            key = f"{dialogue}|{cond}"
            if key in done and not args.force:
                continue
            if cond == "asr:hosted":
                update_results("mtva", {"runs": {**done, key: {"not_run": True}}}, merge=True)
                done[key] = {"not_run": True}
                continue
            t0 = time.time()
            src = source_for(cond)
            calls = [run_call(cid, call, src, dialogue, split=cond == "split", llm_model=args.llm)
                     for cid, call in CALLS.items()]
            if cond == "asr:local":
                ASRCache("local").save()
            agg = aggregate(calls)
            agg["runtime_s"] = round(time.time() - t0, 1)
            done[key] = agg
            update_results("mtva", {"runs": done, "conditions": CONDITIONS, "reference_day": ANCHOR.isoformat()},
                           merge=True)
            (OUT / "mtva").mkdir(parents=True, exist_ok=True)
            (OUT / "mtva" / f"{dialogue}_{cond.replace(':', '-').replace('@', '-')}.json").write_text(
                json.dumps(calls, ensure_ascii=False, indent=0), encoding="utf-8")
            print(f"{key}: intent {agg['intent_accuracy']} slot F1 {agg['slot_f1']} adherence "
                  f"{agg['reply_language_adherence']} completed {agg['completed']}/40 ({agg['runtime_s']} s)",
                  flush=True)


if __name__ == "__main__":
    main()
