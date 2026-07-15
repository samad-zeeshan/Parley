"""Turn a policy action into words, in English or Arabic.

Templates are the source of truth: each one only inserts values from the API
result or from what the caller said. An optional local model may reword the
reply; its text is spoken only if dialogue/grounding.py finds no fact in it
that the API did not return, and it is in the right script. Otherwise the
template is spoken and the rejection is counted.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date

from .grounding import Facts, Violation, ungrounded
from .policy import Action

EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AR_DAYS = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"]
AR_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر",
             "نوفمبر", "ديسمبر"]


def fmt_date(iso: str, lang: str) -> str:
    d = date.fromisoformat(iso[:10])
    if lang == "ar":
        return f"{AR_DAYS[d.weekday()]} {d.day} {AR_MONTHS[d.month - 1]}"
    return f"{EN_DAYS[d.weekday()]} {d.day} {EN_MONTHS[d.month - 1]}"


def fmt_rent(aed: int, lang: str) -> str:
    return f"{aed:,} درهم في السنة" if lang == "ar" else f"AED {aed:,} a year"


def fmt_beds(n: int, lang: str) -> str:
    if lang == "ar":
        return {0: "استوديو", 1: "غرفة نوم", 2: "غرفتين"}.get(n, f"{n} غرف")
    return "studio" if n == 0 else f"{n} bedroom"


def _slot_line(s: dict, lang: str) -> str:
    if lang == "ar":
        return (f"{fmt_date(s['starts_at'], 'ar')} الساعة {s['starts_at'][11:16]}، {fmt_beds(s['bedrooms'], 'ar')} "
                f"في {s['area_ar']}، {fmt_rent(s['annual_rent_aed'], 'ar')}")
    return (f"{fmt_date(s['starts_at'], 'en')} at {s['starts_at'][11:16]}, a {fmt_beds(s['bedrooms'], 'en')} "
            f"in {s['area']}, {fmt_rent(s['annual_rent_aed'], 'en')}")


ASK = {
    "en": {
        "area": "Which area are you looking in?",
        "bedrooms": "How many bedrooms do you need?",
        "date": "Which day would you like to view?",
        "phone": "What is your mobile number?",
        "choice": "Which option would you like?",
    },
    "ar": {
        "area": "في أي منطقة تبحث؟",
        "bedrooms": "كم غرفة نوم تحتاج؟",
        "date": "أي يوم يناسبك للمعاينة؟",
        "phone": "ما رقم جوالك؟",
        "choice": "أي خيار يناسبك؟",
    },
}


def template(action: Action, lang: str) -> str:
    a, en = action.args, lang != "ar"
    if action.name == "ask_slot":
        q = ASK[lang][a["slot"]]
        if a.get("reason") == "none_suitable":
            return ("No problem. " if en else "ولا يهمك. ") + q
        if a.get("reason") == "not_understood":
            return ("Sorry, I did not catch that. " if en else "عذرا، ما سمعتك زين. ") + q
        return q
    if action.name == "offer":
        lines = [f"{'Option' if en else 'الخيار'} {i}: {_slot_line(s, lang)}." for i, s in enumerate(a["slots"], 1)]
        head = "I found these viewings." if en else "لقيت هذه المواعيد."
        if a.get("note") == "choice_out_of_range":
            head = "Please pick one of these." if en else "اختر واحد من هذه المواعيد."
        if a.get("note") == "taken":
            head = "That viewing was just taken. These are still free." if en else "هذا الموعد انحجز الحين. هذه المواعيد متاحة."
        return " ".join([head, *lines, ASK[lang]["choice"]])
    if action.name == "no_results":
        return ("I could not find a free viewing for that. Would you like another day or a higher budget?" if en
                else "ما لقيت موعد متاح لهذا الطلب. تبي يوم ثاني أو ميزانية أعلى؟")
    if action.name == "ask_confirm":
        s = a["slot"]
        where = s["address"] if en else s["address_ar"]
        when = f"{fmt_date(s['starts_at'], lang)} {'at' if en else 'الساعة'} {s['starts_at'][11:16]}"
        if en:
            return f"I am holding {when} at {where}. Shall I confirm the booking?"
        return f"حجزت لك مبدئيا موعد {when} في {where}. أأكد الحجز؟"
    if action.name == "confirmed":
        s = a["slot"]
        when = f"{fmt_date(s['starts_at'], lang)} {'at' if en else 'الساعة'} {s['starts_at'][11:16]}"
        if en:
            return (f"Your viewing is booked for {when} at {s['address']}. "
                    f"Your agent is {s['agent_name']}. Thank you.")
        return f"تم الحجز. موعد المعاينة {when} في {s['address_ar']}. الوكيل {s['agent_name_ar']}. شكرا لك."
    if action.name == "cancelled":
        return "Your booking is cancelled." if en else "تم إلغاء الحجز."
    if action.name == "goodbye":
        return "Thank you, goodbye." if en else "شكرا لك، مع السلامة."
    if action.name == "redirect":
        head = "I can only help with booking property viewings. " if en else "أقدر أساعدك في حجز معاينة العقارات فقط. "
        return head + ASK[lang][a["next"]]
    if action.name == "error":
        return "Sorry, something went wrong. Let us try again." if en else "عذرا، صار خطأ. خلنا نحاول مرة ثانية."
    raise ValueError(f"no template for {action.name}")


def action_slots(action: Action) -> list[dict]:
    if "slots" in action.args:
        return list(action.args["slots"])
    if action.args.get("slot"):
        return [action.args["slot"]] if isinstance(action.args["slot"], dict) else []
    return []


@dataclass
class Rendered:
    text: str
    lang: str
    source: str                   # template | llm
    rejected: list[Violation] = field(default_factory=list)


_ARABIC = re.compile(r"[؀-ۿ]")


class Phraser:
    def __init__(self, llm=None, today: date | None = None, clock=None):
        self.llm = llm
        self._today = today
        self.clock = clock
        self.rejections = 0

    def today(self) -> date:
        return self._today or self.clock.now().date()

    def render(self, action: Action, lang: str, caller: dict | None = None) -> Rendered:
        base = template(action, lang)
        if self.llm is None:
            return Rendered(base, lang, "template")
        slots = action_slots(action)
        facts = Facts.from_slots(slots, caller=caller)
        payload = json.dumps({"action": action.name, "facts": slots, "draft": base}, ensure_ascii=False)
        try:
            reply = (self.llm.phrase(action, payload, lang) or "").strip()
        except Exception:  # noqa: BLE001 -- a phrasing failure falls back to the template
            return Rendered(base, lang, "template")
        bad = ungrounded(reply, facts, self.today())
        if not reply or (lang == "ar") != bool(_ARABIC.search(reply)):
            bad.append(Violation("language", lang))
        if bad:
            self.rejections += 1
            return Rendered(base, lang, "template", bad)
        return Rendered(reply, lang, "llm")


class LLMPhraser:
    """Rewords the template with a local model. Never trusted: Phraser checks the result."""

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 30.0):
        self.model = model or os.environ.get("MAJLIS_LLM_MODEL", "qwen/qwen3.5-9b")
        self.base_url = (base_url or os.environ.get("MAJLIS_LLM_URL", "http://127.0.0.1:1234/v1")).rstrip("/")
        self.timeout = timeout

    def phrase(self, action: Action, payload: str, lang: str) -> str:
        import httpx

        language = "Arabic (Gulf register, simple)" if lang == "ar" else "English"
        system = (f"You are the voice of a property-viewing booking line. Reword the draft reply in {language} "
                  "so it sounds natural when spoken. Use only the facts given. Do not add any date, time, price, "
                  "address, area or number that is not in the facts or the draft. One or two short sentences "
                  "per option. Reply with the spoken text only.")
        body = {"model": self.model, "temperature": 0.3, "max_tokens": 250, "reasoning_effort": "none",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": payload}]}
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
