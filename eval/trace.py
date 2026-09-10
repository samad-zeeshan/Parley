"""TRACE acoustic stress (arXiv 2609.29452): the forty calls again on stressed copies of the caller audio.

    uv run --extra speech python -m eval.trace      # writes the trace section of eval/results.json and eval/trace.svg
"""

from __future__ import annotations

import argparse
import time

from .acoustics import ARMS
from .harness import CALL_KINDS, ROOT, ASRCache, load_results, run_call, update_results
from .scripts import CALLS

NAMES = {"en": "English", "gulf": "Gulf Arabic", "msa": "MSA", "switch": "Code-switched"}


def had_trouble(call: dict) -> bool:
    """Something went wrong in the call: a misheard intent, a wrong slot value, or the agent asking again."""
    for t in call["turns"]:
        if t["intent"] != t["gold_intent"] or t["action_args"].get("reason") == "not_understood":
            return True
        if any(k in t["gold_slots"] and t["gold_slots"][k] != v for k, v in t["slots"].items()):
            return True
    return False


def aggregate_arm(calls: list[dict], clean: list[dict]) -> dict:
    base = {c["call_id"]: c for c in clean}
    out = {}
    for kind in CALL_KINDS:
        cs = [c for c in calls if c["kind"] == kind]
        done = [c for c in cs if c["completed"]]
        troubled = [c for c in cs if had_trouble(c)]
        both = [c for c in done if base.get(c["call_id"], {}).get("completed")]
        out[kind] = {
            "calls": len(cs), "completed": len(done),
            "wrong_actions": sum(len(c["wrong_actions"]) for c in cs),
            "troubled": len(troubled), "recovered": sum(c["completed"] for c in troubled),
            "turns_per_completed_call": round(sum(len(c["turns"]) for c in done) / len(done), 2) if done else None,
            "extra_turns_vs_clean": round(sum(len(c["turns"]) - len(base[c["call_id"]]["turns"]) for c in both)
                                          / len(both), 2) if both else None,
        }
    return out


def plot_svg(rows: dict, speech: str = "local") -> str:
    """One panel per language, one bar per arm: the share of calls completed."""
    arms = list(rows)
    bar, gap, label_w, plot_w, pad = 14, 6, 92, 170, 16
    panel_h = 34 + len(arms) * (bar + gap)
    panel_w = label_w + plot_w + 44
    width, height = 2 * panel_w + 3 * pad, 2 * panel_h + 3 * pad + 24
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
           f'height="{height}" font-family="system-ui, Segoe UI, sans-serif" font-size="11">',
           "<style>.bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.muted{fill:#52514e}.bar{fill:#2a78d6}.grid{stroke:#e4e2dc}"
           "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.ink{fill:#ffffff}.muted{fill:#c3c2b7}"
           ".bar{fill:#3987e5}.grid{stroke:#3a3936}}</style>",
           f'<rect class="bg" width="{width}" height="{height}"/>',
           f'<text class="ink" x="{pad}" y="{pad + 8}" font-size="13" font-weight="600">Calls completed under '
           f'acoustic stress, {speech} speech config</text>']
    for i, kind in enumerate(NAMES):
        x0 = pad + (i % 2) * (panel_w + pad)
        y0 = 24 + pad + (i // 2) * (panel_h + pad)
        out.append(f'<g class="panel" transform="translate({x0},{y0})">')
        out.append(f'<text class="ink" x="0" y="14" font-weight="600">{NAMES[kind]}</text>')
        for frac in (0, 0.5, 1):
            gx = label_w + frac * plot_w
            out.append(f'<line class="grid" x1="{gx}" x2="{gx}" y1="24" y2="{panel_h - 6}"/>')
            out.append(f'<text class="muted" x="{gx}" y="{panel_h + 6}" text-anchor="middle">{int(frac * 100)}%</text>')
        for j, arm in enumerate(arms):
            r = rows[arm][kind]
            share = r["completed"] / r["calls"] if r["calls"] else 0
            y = 28 + j * (bar + gap)
            w = max(share * plot_w, 0)
            out.append(f'<text class="muted" x="{label_w - 6}" y="{y + bar - 3}" text-anchor="end">{arm}</text>')
            if w > 0:
                out.append(f'<rect class="bar" x="{label_w}" y="{y}" width="{w:.1f}" height="{bar}" rx="3">'
                           f'<title>{NAMES[kind]}, {arm}: {r["completed"]} of {r["calls"]}</title></rect>')
            out.append(f'<text class="ink" x="{label_w + w + 4:.1f}" y="{y + bar - 3}">{r["completed"]}/{r["calls"]}'
                       f'</text>')
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--speech", default="local")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    clean = [run_call(cid, call, ASRCache(args.speech).get) for cid, call in CALLS.items()]
    done = load_results().get("trace", {}).get("by_arm", {})
    for arm in args.arms.split(","):
        if arm in done and not args.force:
            continue
        t0 = time.time()
        cache = ASRCache(args.speech, arm)
        calls = clean if arm == "clean" else [run_call(cid, call, cache.get) for cid, call in CALLS.items()]
        cache.save()
        done[arm] = aggregate_arm(calls, clean)
        update_results("trace", {"speech": args.speech, "by_arm": done}, merge=True)
        total = sum(r["completed"] for r in done[arm].values())
        print(f"{arm}: completed {total}/40, wrong actions "
              f"{sum(r['wrong_actions'] for r in done[arm].values())} ({time.time() - t0:.0f} s)", flush=True)
    (ROOT / "trace.svg").write_text(plot_svg({a: done[a] for a in ARMS if a in done}, args.speech), encoding="utf-8",
                                    newline="\n")


if __name__ == "__main__":
    main()
