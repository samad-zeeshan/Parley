"""Decision heads read from one forward pass (the Jev readout).

After Open-Jev / JevLite (arXiv 2609.23959) and JEV-as-a-Judge (arXiv 2609.26550),
implemented here from the papers' description; no code was published with them.

A head declares its options, tags each with a one-token label (A, B, C, ...),
and asks one question. One forward pass of a small causal LM gives the logits
at the last position; the softmax over just the label tokens, divided by a
fitted temperature, is one probability per option. No text is generated.

A cascade accepts a decision whose top probability reaches the threshold and
escalates the rest: intent and yes/no go to the rule parser, the judge goes to
the deterministic grounding check. Slots are never read from Jev; they stay
with the rule parser. See docs/adr/0004-jev-decision-heads.md.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from speech.normalize import Normalized, normalize

from .grounding import Facts, Violation, ungrounded
from .nlu import RuleNLU
from .schema import INTENTS, NLUResult

LABELS = "ABCDEFGHIJKLMNOP"
CALIBRATION_FILE = Path(__file__).resolve().parent / "jev_calibration.json"
DEFAULT_THRESHOLD = 0.8

INTENT_HELP = {
    "book_viewing": "wants to book or look for a property viewing",
    "provide_details": "answers a question: budget, day, time, phone number, bedrooms",
    "choose_option": "picks one of the viewings the agent offered (first, second...)",
    "confirm": "says yes, agrees, asks to go ahead",
    "deny": "says no, refuses what was offered",
    "cancel_booking": "wants to cancel an existing booking",
    "repeat": "asks the agent to repeat",
    "greet": "only says hello",
    "goodbye": "ends the call",
    "out_of_scope": "asks about something unrelated to property viewings",
    "unclear": "nothing usable: noise, a fragment, or unintelligible",
}


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Head:
    name: str
    options: list[str]
    describe: dict[str, str]
    question: Callable[..., str]
    example: dict = field(default_factory=dict)

    def prompt(self, **kw) -> str:
        lines = [self.question(**kw), "", "Options:"]
        for label, opt in zip(LABELS, self.options):
            lines.append(f"{label}) {self.describe.get(opt, opt)}")
        lines += ["", "Reply with the letter only."]
        return "\n".join(lines)


def _q_intent(text: str, context: str = "") -> str:
    ctx = f"The agent had just asked about: {context}.\n" if context else ""
    return (f"A caller on a UAE property-viewing booking line said (English, Gulf Arabic, MSA or a mix):\n"
            f"\"{text}\"\n{ctx}What does the caller want?")


def _q_yes_no(text: str, question: str) -> str:
    return (f"The agent asked: \"{question}\"\nThe caller answered (English or Arabic): \"{text}\"\n"
            f"Did the caller say yes or no?")


def _q_dialect(text: str) -> str:
    return f"Transcript of a caller: \"{text}\"\nWhich language variety is this?"


def _q_judge(reply: str, facts: str) -> str:
    return ("Facts returned by the booking database:\n" + facts + "\n\n"
            f"Candidate reply to the caller:\n\"{reply}\"\n\n"
            "Is every date, time, price, address, area and number in the reply taken from the facts?")


HEADS: dict[str, Head] = {
    "intent": Head("intent", list(INTENTS), INTENT_HELP, _q_intent, {"text": "hello"}),
    "yes_no": Head("yes_no", ["yes", "no"], {"yes": "yes", "no": "no"}, _q_yes_no,
                   {"text": "yes", "question": "Shall I confirm the booking?"}),
    "dialect": Head("dialect", ["english", "gulf", "msa", "switch"],
                    {"english": "English", "gulf": "Gulf Arabic (Emirati, Khaleeji)",
                     "msa": "Modern Standard Arabic (fusha)",
                     "switch": "Arabic and English mixed in one sentence"},
                    _q_dialect, {"text": "hello"}),
    "judge": Head("judge", ["grounded", "not_grounded"],
                  {"grounded": "yes, everything stated is in the facts",
                   "not_grounded": "no, something stated is not in the facts"},
                  _q_judge, {"reply": "ok", "facts": "(none)"}),
}

DIALECT_TO_LABEL = {"english": "en", "gulf": "ar-gulf", "msa": "ar-msa", "switch": "mixed"}
LABEL_TO_DIALECT = {v: k for k, v in DIALECT_TO_LABEL.items()}


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    t = max(temperature, 1e-6)
    m = max(logits)
    ex = [math.exp((x - m) / t) for x in logits]
    s = sum(ex)
    return [e / s for e in ex]


def fit_temperature(rows: list[tuple[list[float], int]]) -> float:
    """Temperature minimizing negative log likelihood of the gold option, by grid search."""
    if not rows:
        return 1.0
    grid = [round(0.05 * 1.12 ** k, 4) for k in range(60)]  # 0.05 .. about 40
    best_t, best = 1.0, float("inf")
    for t in grid:
        nll = -sum(math.log(max(softmax(lg, t)[y], 1e-12)) for lg, y in rows) / len(rows)
        if nll < best - 1e-9:
            best_t, best = t, nll
    return best_t


def expected_calibration_error(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    """ECE over (confidence, correct) pairs with equal-width bins."""
    if not pairs:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pairs if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if sel:
            conf = sum(c for c, _ in sel) / len(sel)
            acc = sum(ok for _, ok in sel) / len(sel)
            total += abs(acc - conf) * len(sel) / len(pairs)
    return total


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FixedLogits:
    """Test double: logits come from a function of (head, prompt, n_options)."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0
        self.name = "fixed"

    def label_logits(self, head: str, prompt: str, labels: list[str]) -> list[float]:
        self.calls += 1
        return list(self.fn(head, prompt, len(labels)))


class HFLogitModel:
    """A causal LM on CPU through transformers. One forward pass per decision."""

    def __init__(self, name: str | None = None, threads: int | None = None, adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name or os.environ.get("PARLEY_JEV_MODEL", "Qwen/Qwen3-1.7B")
        torch.set_num_threads(threads or max(1, (os.cpu_count() or 4) - 2))
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=torch.float32)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self._lock = threading.Lock()
        self.label_ids = {}
        for label in LABELS:
            ids = self.tok.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"label {label!r} is not one token for {self.name}")
            self.label_ids[label] = ids[0]

    def render(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def label_logits(self, head: str, prompt: str, labels: list[str]) -> list[float]:
        enc = self.tok(self.render(prompt), return_tensors="pt")
        with self._lock, self.torch.no_grad():
            last = self.model(**enc).logits[0, -1]
        return [float(last[self.label_ids[label]]) for label in labels]


# ---------------------------------------------------------------------------
# Decider and cascade
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    grounded: bool
    source: str                  # jev | check
    probability: float           # Jev's probability of its own choice
    overruled: bool = False      # Jev said grounded, the check found a violation
    violations: list[Violation] = field(default_factory=list)


def load_calibration(path: Path = CALIBRATION_FILE) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


class JevDecider:
    def __init__(self, model, threshold: float | None = None, temperatures: dict | None = None):
        cal = load_calibration() if temperatures is None or threshold is None else {}
        self.model = model
        env = os.environ.get("PARLEY_JEV_THRESHOLD")
        self.threshold = threshold if threshold is not None else float(env or cal.get("threshold",
                                                                                     DEFAULT_THRESHOLD))
        self.temperatures = temperatures if temperatures is not None else {
            k: v["temperature"] for k, v in cal.get("heads", {}).items()}

    def logits(self, head: str, **kw) -> list[float]:
        h = HEADS[head]
        return self.model.label_logits(head, h.prompt(**kw), list(LABELS[:len(h.options)]))

    def probabilities(self, head: str, **kw) -> dict[str, float]:
        lg = self.logits(head, **kw)
        p = softmax(lg, self.temperatures.get(head, 1.0))
        return dict(zip(HEADS[head].options, p))

    def decide(self, head: str, **kw) -> tuple[str, float, dict]:
        probs = self.probabilities(head, **kw)
        best = max(probs, key=probs.get)
        return best, probs[best], probs

    def judge(self, reply: str, slots: list[dict], today: date, caller: dict | None = None) -> Verdict:
        facts = Facts.from_slots(slots, caller=caller)
        choice, p, _ = self.decide("judge", reply=reply, facts=facts_text(slots, caller))
        if p >= self.threshold and choice == "not_grounded":
            return Verdict(False, "jev", p)
        # A "grounded" verdict, confident or not, is never taken on trust: the oracle rule is checked.
        violations = ungrounded(reply, facts, today)
        if p >= self.threshold:
            return Verdict(not violations, "jev", p, overruled=bool(violations), violations=violations)
        return Verdict(not violations, "check", p, violations=violations)


def facts_text(slots: list[dict], caller: dict | None = None) -> str:
    lines = []
    for i, s in enumerate(slots, 1):
        lines.append(f"{i}. {s['starts_at'][:10]} {s['starts_at'][11:16]}-{s['ends_at'][11:16]}, "
                     f"{s['bedrooms']} bedroom, {s['area']} ({s.get('area_ar', '')}), "
                     f"{s['annual_rent_aed']} AED a year, {s['address']} / {s.get('address_ar', '')}, "
                     f"agent {s['agent_name']}")
    if caller:
        lines.append("The caller said: " + ", ".join(f"{k}={v}" for k, v in caller.items()))
    return "\n".join(lines) or "(none)"


YES_NO_QUESTION = {"ask_confirm": "Shall I confirm the booking?", "offer": "Would you like this viewing?"}


class JevNLU:
    """Intent from a Jev head, slots and option choice from the rule parser, rules as the escalation."""

    name = "jev"

    def __init__(self, decider: JevDecider, rules: RuleNLU | None = None):
        self.decider = decider
        self.rules = rules or RuleNLU()
        self.accepted = 0
        self.escalated = 0

    def parse(self, text: str, today: date, norm: Normalized | None = None, context: str = "") -> NLUResult:
        norm = norm or normalize(text, today)
        rules = self.rules.parse(text, today, norm)
        if context == "ask_confirm":
            head = "yes_no"
            choice, p, probs = self.decider.decide("yes_no", text=text, question=YES_NO_QUESTION[context])
            intent = "confirm" if choice == "yes" else "deny"
        else:
            head = "intent"
            intent, p, probs = self.decider.decide("intent", text=text, context=context)
        if p >= self.decider.threshold:
            self.accepted += 1
            source = "jev"
        else:
            self.escalated += 1
            intent, source = rules.intent, "jev-escalated"
        choice = rules.choice if intent == "choose_option" else None
        return NLUResult(intent=intent, slots=dict(rules.slots), choice=choice, source=source,
                         confidence=p, head=head, probabilities=probs)


class JevPhraser:
    """Wraps a Phraser: after the model rewords a reply, the Jev judge decides whether it may be spoken."""

    def __init__(self, phraser, decider: JevDecider):
        self.phraser = phraser
        self.decider = decider
        self.verdicts: list[Verdict] = []

    def render(self, action, lang: str, caller: dict | None = None):
        from .phrasing import Rendered, action_slots, template

        out = self.phraser.render(action, lang, caller=caller)
        if out.source != "llm":
            return out
        v = self.decider.judge(out.text, action_slots(action), self.phraser.today(), caller=caller)
        self.verdicts.append(v)
        if v.grounded:
            return out
        return Rendered(template(action, lang), lang, "template",
                        v.violations or [Violation("judge", round(v.probability, 3))], rejected_text=out.text)
