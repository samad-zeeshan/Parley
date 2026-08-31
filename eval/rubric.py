"""Gulf Arabic replies scored with a rubric adapted from arXiv 2608.29990, by a local judge calibrated on anchors.

Anchor labels were written for this project; no native Gulf speaker has checked them or the judge yet.
"""

from __future__ import annotations

import argparse
import json
import random

from .harness import ANCHOR, OUT, ROOT, new_world, update_results

# Positive criteria, atomic and applied to every reply. The paper derives them per prompt from an expert
# ground truth. Here the template draft is the ground truth for facts and the criteria are fixed.
CRITERIA = {
    "gulf_register": "Where Gulf and Modern Standard Arabic differ, the reply uses the Gulf form.",
    "courtesy": "The reply is polite in the way a Gulf service call is: warm, not curt, not stiff.",
    "clarity": "A caller knows exactly what is being said or asked, with one reading only.",
    "gulf_lexicon": "No word is Levantine, Egyptian or Maghrebi only; words carry their Gulf meaning.",
    "fidelity": "Every fact in the draft is kept and no fact is added.",
}
# Penalties for errors the reply introduces, from the paper's taxonomy. Hallucination costs double
# because in a booking call it sends someone to the wrong place.
PENALTIES = {"hallucination": 2, "ambiguous_framing": 1, "register_flattening": 1, "wrong_dialect": 1,
             "script_error": 1}


def score(verdict: dict) -> float:
    met = sum(bool(verdict["criteria"].get(c)) for c in CRITERIA)
    lost = sum(PENALTIES[p] for p in set(verdict.get("penalties", [])) if p in PENALTIES)
    return max(-1.0, (met - lost) / len(CRITERIA))


def _v(miss=(), pen=()):
    return {"criteria": {c: c not in miss for c in CRITERIA}, "penalties": list(pen)}


_CONFIRM = "حجزت لك مبدئيا موعد الخميس 1 أكتوبر الساعة 16:00 في شقة 1203، برج ب4، دبي مارينا. أأكد الحجز؟"
_OFFER = ("لقيت هذه المواعيد. الخيار 1: الخميس 1 أكتوبر الساعة 16:00، غرفتين في دبي مارينا، 160,000 درهم في السنة. "
          "الخيار 2: الخميس 1 أكتوبر الساعة 17:00، غرفتين في دبي مارينا، 165,000 درهم في السنة. أي خيار يناسبك؟")
ANCHORS = [
    {"draft": "في أي منطقة تبحث؟", "reply": "هلا فيك، في أي منطقة تدوّر شقة؟", "reference": _v()},
    {"draft": "في أي منطقة تبحث؟", "reply": "في أي منطقة تبحث عن الشقة؟ أرجو إفادتي.",
     "reference": _v(["gulf_register"], ["register_flattening"])},
    {"draft": "في أي منطقة تبحث؟", "reply": "إنت عايز شقة في أنهي منطقة؟",
     "reference": _v(["gulf_register", "gulf_lexicon"], ["wrong_dialect"])},
    {"draft": "في أي منطقة تبحث؟", "reply": "شو المنطقة اللي بدك ياها؟",
     "reference": _v(["gulf_register", "gulf_lexicon"], ["wrong_dialect"])},
    {"draft": "في أي منطقة تبحث؟", "reply": "قول المنطقة.", "reference": _v(["courtesy"])},
    {"draft": _CONFIRM, "reply": _CONFIRM + " والإيجار شامل الفواتير.",
     "reference": _v(["fidelity"], ["hallucination"])},
    {"draft": _OFFER, "reply": "عندي موعدين، تبي هذا ولا ذاك؟",
     "reference": _v(["clarity", "fidelity"], ["ambiguous_framing"])},
    {"draft": _OFFER, "reply": ("لقيت لك موعدين يوم الخميس 1 أكتوبر في دبي مارينا، غرفتين: الأول الساعة 16:00 "
                                "والثاني الساعة 17:00. أي واحد يناسبك؟"), "reference": _v(["fidelity"])},
    {"draft": "تم الحجز. موعد المعاينة الخميس 1 أكتوبر الساعة 16:00 في شقة 1203، برج ب4، دبي مارينا. الوكيل عائشة. شكرا لك.",
     "reply": "تمام، حجزنا لك المعاينة يوم الخميس 1 أكتوبر الساعة 16:00 في شقة 1203، برج ب4، دبي مارينا. "
              "الوكيلة عائشة بتكون موجودة. مشكور.", "reference": _v()},
    {"draft": "تم الحجز. موعد المعاينة الخميس 1 أكتوبر الساعة 16:00 في شقة 1203، برج ب4، دبي مارينا. الوكيل عائشة. شكرا لك.",
     "reply": "7ajazt lik mow3ed yom al5amees", "reference": _v(["courtesy", "clarity", "fidelity"], ["script_error"])},
    {"draft": "عذرا، نظام الحجز ما يرد عدل الحين، فما أقدر أأكد لك شي. بنتصل فيك على 0501234567.",
     "reply": "المعذرة، النظام ما يرد الحين، فما أقدر أأكد لك شي. بنتصل فيك على 0501234567.", "reference": _v()},
    {"draft": "ما لقيت موعد متاح لهذا الطلب. تبي يوم ثاني أو ميزانية أعلى؟",
     "reply": "نعتذر، لا توجد مواعيد متاحة هذا الأسبوع بأكمله.",
     "reference": _v(["gulf_register", "fidelity"], ["register_flattening", "hallucination"])},
    {"draft": "كم غرفة نوم تحتاج؟", "reply": "زين، كم غرفة تبي؟", "reference": _v()},
]

_SYSTEM = """You grade one reply from a property-viewing phone agent speaking to a caller from the UAE.
The caller speaks Gulf Arabic. You get the DRAFT (the facts the reply must carry) and the REPLY.
For each criterion answer true or false:
{criteria}
Then list every penalty that applies, from: {penalties}.
- hallucination: the reply states a fact that is not in the draft.
- ambiguous_framing: the reply can be read two ways, or hides which option is meant.
- register_flattening: the reply drifts into formal Modern Standard Arabic where a Gulf speaker would not.
- wrong_dialect: the reply uses a word from another Arabic dialect.
- script_error: the reply is not in Arabic script, or has broken spelling.
Answer with JSON only."""


def judge_schema() -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["criteria", "penalties"],
            "properties": {
                "criteria": {"type": "object", "additionalProperties": False, "required": list(CRITERIA),
                             "properties": {c: {"type": "boolean"} for c in CRITERIA}},
                "penalties": {"type": "array", "items": {"enum": list(PENALTIES)}}}}


class LMStudioJudge:
    def __init__(self, model: str = "qwen/qwen3.6-35b-a3b", url: str = "http://127.0.0.1:1234/v1"):
        self.model, self.url = model, url

    def __call__(self, draft: str, reply: str) -> dict:
        import httpx

        system = _SYSTEM.format(criteria="\n".join(f"- {k}: {v}" for k, v in CRITERIA.items()),
                                penalties=", ".join(PENALTIES))
        body = {"model": self.model, "temperature": 0, "max_tokens": 300, "reasoning_effort": "none",
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": f"DRAFT: {draft}\nREPLY: {reply}"}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "rubric", "strict": True, "schema": judge_schema()}}}
        r = httpx.post(f"{self.url}/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])


def fit_line(x: list[float], y: list[float]) -> tuple[float, float]:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    var = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / var if var else 1.0
    return slope, my - slope * mx


def calibrate(judged: list[dict]) -> dict:
    """How far the judge agrees with the anchor labels, and the line that maps its scores onto theirs.

    The anchor-and-map idea is from arXiv 2609.29431: a judge's raw scale is not comparable until it is
    fitted to a small labelled set.
    """
    agree = total = pen_agree = 0
    for a, v in zip(ANCHORS, judged):
        for c in CRITERIA:
            agree += bool(v["criteria"].get(c)) == a["reference"]["criteria"][c]
            total += 1
        pen_agree += set(v.get("penalties", [])) == set(a["reference"]["penalties"])
    raw = [score(v) for v in judged]
    ref = [score(a["reference"]) for a in ANCHORS]
    slope, intercept = fit_line(raw, ref)
    fitted = [slope * r + intercept for r in raw]
    return {"anchors": len(ANCHORS), "criterion_agreement": round(agree / total, 3),
            "penalty_sets_matching": pen_agree,
            "mean_abs_error_raw": round(sum(abs(a - b) for a, b in zip(raw, ref)) / len(ref), 3),
            "mean_abs_error_calibrated": round(sum(abs(a - b) for a, b in zip(fitted, ref)) / len(ref), 3),
            "slope": round(slope, 4), "intercept": round(intercept, 4)}


def template_replies() -> list[dict]:
    """Every Arabic template the agent can speak, filled from the seeded listings."""
    from dialogue.phrasing import template
    from dialogue.policy import Action

    conn, clock, svc = new_world()
    slots = svc.list_slots(area="Dubai Marina", bedrooms=2, date=ANCHOR.date().isoformat(), limit=3)
    s = slots[0]
    actions = [Action("ask_slot", {"slot": k}) for k in ("area", "bedrooms", "date", "phone", "choice")]
    actions += [Action("ask_slot", {"slot": "date", "reason": "none_suitable"}),
                Action("ask_slot", {"slot": "phone", "reason": "not_understood"}),
                Action("offer", {"slots": slots}), Action("offer", {"slots": slots, "note": "taken"}),
                Action("no_results", {}), Action("ask_confirm", {"slot": s, "phone": "0501234567"}),
                Action("confirmed", {"slot": s, "booking_id": "b-000000000000"}), Action("cancelled", {}),
                Action("redirect", {"next": "area"}), Action("error", {}), Action("goodbye", {}),
                Action("listen", {}), Action("unavailable", {"phone": "0501234567"}), Action("unavailable", {})]
    out = []
    for a in actions:
        text = template(a, "ar")
        slot = a.args.get("slot")
        variant = a.args.get("reason") or a.args.get("note") or (slot if isinstance(slot, str) else None)
        out.append({"action": a.name, "variant": variant, "draft": text, "text": text, "source": "template"})
    conn.close()
    return out


def model_replies(limit: int = 40) -> list[dict]:
    """The 9B model's rewordings for Gulf callers, spoken or rejected, from the MTVA llm+phrasing runs."""
    out = []
    for path in sorted((OUT / "mtva").glob("llm+phrasing_*.json")):
        for c in json.loads(path.read_text(encoding="utf-8")):
            for t in c["turns"]:
                if t["dialect"] != "ar-gulf" or t["reply_lang"] != "ar" or not t.get("draft"):
                    continue
                text = t["reply"] if t["reply_source"] == "llm" else t.get("rejected_text")
                if text:
                    out.append({"action": t["action"], "draft": t["draft"], "text": text, "source": "llm",
                                "spoken": t["reply_source"] == "llm"})
    random.Random(7).shuffle(out)
    return out[:limit]


def summarize(items: list[dict], cal: dict) -> dict:
    if not items:
        return {"replies": 0}
    raw = [score(i["verdict"]) for i in items]
    fitted = [cal["slope"] * r + cal["intercept"] for r in raw]
    pens: dict = {}
    for i in items:
        for p in set(i["verdict"]["penalties"]):
            pens[p] = pens.get(p, 0) + 1
    return {"replies": len(items), "mean_score_raw": round(sum(raw) / len(raw), 3),
            "mean_score_calibrated": round(sum(fitted) / len(fitted), 3),
            "criterion_rates": {c: round(sum(i["verdict"]["criteria"][c] for i in items) / len(items), 3)
                                for c in CRITERIA},
            "penalties": pens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen/qwen3.6-35b-a3b")
    args = ap.parse_args()
    judge = LMStudioJudge(args.judge)
    judged_anchors = [judge(a["draft"], a["reply"]) for a in ANCHORS]
    cal = calibrate(judged_anchors)
    print("calibration", cal, flush=True)
    templates = [{**r, "verdict": judge(r["draft"], r["text"])} for r in template_replies()]
    model = [{**r, "verdict": judge(r["draft"], r["text"])} for r in model_replies()]
    spot = random.Random(11).sample(templates + model, min(10, len(templates + model)))
    (ROOT / "rubric_spotcheck.json").write_text(json.dumps(
        [{"draft": s["draft"], "reply": s["text"], "source": s["source"], "judge": s["verdict"],
          "native_speaker_verdict": None} for s in spot], ensure_ascii=False, indent=1), encoding="utf-8",
        newline="\n")
    (OUT / "rubric_items.json").write_text(json.dumps(templates + model, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    update_results("rubric", {
        "judge": args.judge, "calibration": cal, "templates": summarize(templates, cal),
        "model_rewording": summarize(model, cal),
        "spot_check": {"sample": len(spot), "file": "eval/rubric_spotcheck.json", "done": False},
    })


if __name__ == "__main__":
    main()
