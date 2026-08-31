"""Intent and slots into the strict schema in dialogue/schema.py, by rules or by a local model through LM Studio.

Both read the normalizer's output. Model output is validated, and the rule parser is the fallback.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta

from jsonschema import ValidationError, validate

from speech.normalize import Normalized, _norm_word, normalize

from .schema import AREA_NAMES, INTENTS, NLU_SCHEMA, NLUResult


def _n(words: list[str]) -> list[str]:
    return [" ".join(_norm_word(w) for w in phrase.split()) for phrase in words]


AREA_ALIASES: dict[str, str] = {}
for _area, _aliases in {
    "Dubai Marina": ["dubai marina", "marina", "دبي مارينا", "مارينا", "المارينا", "دبي مارينه", "مارينه"],
    "Jumeirah Lake Towers": ["jumeirah lake towers", "jlt", "j l t", "أبراج بحيرات جميرا", "ابراج بحيرات جميرا",
                             "جي ال تي"],
    "Downtown Dubai": ["downtown dubai", "downtown", "داون تاون", "داونتاون", "وسط مدينة دبي", "وسط دبي"],
    "Business Bay": ["business bay", "الخليج التجاري", "بزنس باي", "بيزنس باي"],
    "Al Barsha": ["al barsha", "barsha", "البرشاء", "البرشا", "برشاء"],
    "Jumeirah Village Circle": ["jumeirah village circle", "jvc", "j v c", "قرية جميرا الدائرية", "جي في سي"],
    "Al Reem Island": ["al reem island", "reem island", "al reem", "reem", "جزيرة الريم", "الريم"],
    "Khalifa City": ["khalifa city", "مدينة خليفة", "خليفة سيتي"],
    "Al Raha Beach": ["al raha beach", "raha beach", "al raha", "شاطئ الراحة", "الراحة"],
    "Saadiyat Island": ["saadiyat island", "saadiyat", "جزيرة السعديات", "السعديات"],
}.items():
    for _alias in _n(_aliases):
        AREA_ALIASES[_alias] = _area
_AREA_KEYS = sorted(AREA_ALIASES, key=lambda a: -len(a.split()))

_ROOM_EN = {"bedroom", "bedrooms", "bed", "beds", "br", "bhk", "room", "rooms", "bedder"}
_ROOM_AR = set(_n(["غرف", "غرفة", "غرفه", "غرفات"]))
_ROOM_DUAL = set(_n(["غرفتين", "غرفتان"]))
_ONE_AR = set(_n(["وحدة", "واحدة", "وحده", "واحده"]))
_STUDIO = set(_n(["studio", "ستوديو", "استوديو", "ستديو"]))

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1, "1st": 1, "2nd": 2, "3rd": 3}
_ORDINALS.update({w: i for i, ws in enumerate([
    ["الاول", "الاولى", "اول"], ["الثاني", "الثانيه"], ["الثالث", "الثالثه"], ["الرابع"], ["الخامس"]], start=1)
    for w in _n(ws)})
_OPTION_WORDS = set(_n(["option", "number", "choice", "الخيار", "خيار", "رقم"]))

CANCEL = set(_n(["cancel", "الغ", "الغي", "الغاء", "كنسل", "الغيه"]))
GOODBYE = set(_n(["bye", "goodbye", "باي"])) | {"مع السلامه", "في امان الله"}
DENY = set(_n(["no", "nope", "not", "don't", "dont", "none", "neither", "لا", "مب", "مو", "كلا"])) | {
    "ما ابي", "ما ابغي", "ما اريد", "لا اريد", "ما يناسبني", "doesn't work", "does not work"}
CONFIRM = set(_n(["yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "correct", "fine", "perfect",
                  "نعم", "ايوه", "ايه", "اي", "تمام", "اكيد", "اكد", "أكد", "أؤكد", "زين", "خلاص", "موافق",
                  "ماشي", "طيب", "اوكي"])) | {"go ahead", "book it", "please do", "that works"}
BOOK = set(_n(["book", "booking", "viewing", "view", "visit", "see", "looking", "want", "apartment", "flat",
               "احجز", "حجز", "معاينه", "اشوف", "ابحث", "ابي", "ابغي", "اريد", "شقه", "نبي", "ابا"]))
GREET = set(_n(["hi", "hello", "hey", "salam", "مرحبا", "هلا", "اهلا"])) | {"السلام عليكم"}
REPEAT = set(_n(["repeat", "again", "عيد", "كرر"])) | {"ما سمعت", "say that again", "مره ثانيه"}
OUT_OF_SCOPE = set(_n(["weather", "temperature", "news", "football", "joke", "recipe", "الطقس", "الجو",
                       "اخبار", "نكته", "مباراه"]))


def _is_arabic(w: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in w)


def _has(words: list[str], text: str, lexicon: set[str]) -> bool:
    joined = f" {text} "
    return any(w in lexicon for w in words) or any(" " in p and f" {p} " in joined for p in lexicon)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _near(token: str, target: str) -> bool:
    """Close enough to be the same domain word misspelt by ASR: one edit, two for long words."""
    if len(target) < 5 or abs(len(token) - len(target)) > 2:
        return False
    return edit_distance(token, target) <= (1 if max(len(token), len(target)) < 7 else 2)


class RuleNLU:
    name = "rules"

    def __init__(self, fuzzy: bool = False):
        # Whisper small misspells short Arabic answers by a letter or two ("أورفتين" for "غرفتين"). With
        # fuzzy on, area names, ordinals and the bedroom dual also match within a small edit distance.
        self.fuzzy = fuzzy

    def parse(self, text: str, today: date, norm: Normalized | None = None) -> NLUResult:
        norm = norm or normalize(text, today)
        words = norm.text.split()
        joined = norm.text
        slots: dict = {}

        for key in _AREA_KEYS:
            if f" {key} " in f" {joined} ":
                slots["area"] = AREA_ALIASES[key]
                break
        if self.fuzzy and "area" not in slots:
            area = self._fuzzy_area(words)
            if area:
                slots["area"] = area

        beds = self._bedrooms(words, norm)
        if beds is None and self.fuzzy and any(_near(w, d) for w in words for d in _ROOM_DUAL):
            beds = 2
        if beds is not None:
            slots["bedrooms"] = beds

        for e in norm.entities:
            if e.kind == "phone":
                slots["phone"] = e.value
            elif e.kind == "date":
                slots["date"] = e.value
            elif e.kind == "time_window":
                slots["time_window"] = list(e.value)
            elif e.kind == "time":
                start = datetime.strptime(e.value, "%H:%M")
                end = start + timedelta(hours=1)
                slots["time_window"] = [e.value, "24:00" if end.day != start.day else end.strftime("%H:%M")]
            elif e.kind == "number" and isinstance(e.value, int) and 10_000 <= e.value <= 5_000_000:
                slots["budget"] = e.value

        choice = self._choice(words)
        if choice is None and self.fuzzy:
            choice = self._fuzzy_choice(words)
        if _has(words, joined, CANCEL):
            intent = "cancel_booking"
        elif _has(words, joined, GOODBYE):
            intent = "goodbye"
        elif _has(words, joined, OUT_OF_SCOPE):
            intent = "out_of_scope"
        elif choice is not None:
            intent = "choose_option"
        elif _has(words, joined, DENY):
            intent = "deny"
        elif _has(words, joined, CONFIRM):
            intent = "confirm"
        elif _has(words, joined, BOOK) or "area" in slots:
            intent = "book_viewing"
        elif slots:
            intent = "provide_details"
        elif _has(words, joined, REPEAT):
            intent = "repeat"
        elif _has(words, joined, GREET):
            intent = "greet"
        else:
            intent = "unclear"
        return NLUResult(intent=intent, slots=slots, choice=choice, source=self.name)

    @staticmethod
    def _fuzzy_area(words: list[str]) -> str | None:
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                for alias in _AREA_KEYS:
                    if len(alias.split()) == n and _near(gram, alias):
                        return AREA_ALIASES[alias]
        return None

    @staticmethod
    def _fuzzy_choice(words: list[str]) -> int | None:
        """Ordinals with or without the article: ثاني is as common as الثاني in Gulf speech."""
        for w in words:
            bare = w[2:] if w.startswith("ال") else w
            for word, n in _ORDINALS.items():
                if n <= 0 or not _is_arabic(word):
                    continue
                target = word[2:] if word.startswith("ال") else word
                if len(bare) >= 3 and (bare == target or (len(target) >= 5 and _near(bare, target))):
                    return n
        return None

    @staticmethod
    def _bedrooms(words: list[str], norm: Normalized) -> int | None:
        for i, w in enumerate(words):
            if w in _STUDIO:
                return 0
            if w in _ROOM_DUAL:
                return 2
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if w.isdigit() and int(w) <= 5 and (nxt in _ROOM_EN or nxt in _ROOM_AR):
                return int(w)
            # Arabic puts the count after the noun, and the normalizer has already written وحدة as 1.
            if w in _ROOM_AR and (nxt in _ONE_AR or nxt == "1"):
                return 1
        return None

    @staticmethod
    def _choice(words: list[str]) -> int | None:
        for i, w in enumerate(words):
            if w in _ORDINALS and _ORDINALS[w] > 0:
                return _ORDINALS[w]
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if w in _OPTION_WORDS and nxt.isdigit() and 1 <= int(nxt) <= 5:
                return int(nxt)
        return None


# ---------------------------------------------------------------------------
# Local model
# ---------------------------------------------------------------------------

_SYSTEM = """You fill a JSON form for a property-viewing booking line in the UAE.
The caller may speak English, Gulf Arabic, Modern Standard Arabic, or mix them.
Return only JSON matching the schema. Rules:
- intent is one of: {intents}.
- slots holds only what THIS utterance states. Leave a slot out if it is not said.
- area must be one of: {areas}. Map Arabic names to these English names.
- bedrooms: 0 for a studio. "غرفتين" means 2.
- budget: annual rent in AED as an integer.
- date: YYYY-MM-DD. Today is {today} ({weekday}).
- time_window: ["HH:MM","HH:MM"] 24-hour. A single time T means [T, T+1h].
- phone: UAE number as digits starting with 0.
- choice: the option number when the caller picks one of the offered viewings, else null.
- The text has already been normalized: numbers, dates and times appear as digits,
  ISO dates and HH:MM. Trust those values.
Context: the agent last asked about: {context}."""


class LLMNLU:
    """Intent and slots from a local model through LM Studio. Validated, with rule fallback."""

    name = "llm"

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 30.0,
                 fallback: RuleNLU | None = None):
        self.model = model or os.environ.get("PARLEY_LLM_MODEL", "qwen/qwen3.5-9b")
        self.base_url = (base_url or os.environ.get("PARLEY_LLM_URL", "http://127.0.0.1:1234/v1")).rstrip("/")
        self.timeout = timeout
        self.fallback = fallback or RuleNLU()
        self.rejections = 0
        self.dropped_slots = 0

    def _messages(self, norm: Normalized, today: date, context: str) -> list[dict]:
        system = _SYSTEM.format(intents=", ".join(INTENTS), areas=", ".join(AREA_NAMES), today=today.isoformat(),
                                weekday=today.strftime("%A"), context=context or "nothing yet")
        ents = [{"kind": e.kind, "value": e.value} for e in norm.entities]
        user = json.dumps({"utterance": norm.text, "entities": ents}, ensure_ascii=False)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def raw(self, norm: Normalized, today: date, context: str = "") -> str:
        import httpx

        body = {
            "model": self.model,
            "messages": self._messages(norm, today, context),
            "temperature": 0,
            "max_tokens": 200,
            # The served models think by default, and thinking costs 10 to 50 s per turn on CPU.
            "reasoning_effort": "none",
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "nlu", "strict": True, "schema": NLU_SCHEMA}},
        }
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def parse(self, text: str, today: date, norm: Normalized | None = None, context: str = "") -> NLUResult:
        norm = norm or normalize(text, today)
        try:
            content = self.raw(norm, today, context)
            data = json.loads(_strip_fences(content))
            validate(data, NLU_SCHEMA)
        except (ValidationError, ValueError, KeyError, OSError, Exception):  # noqa: BLE001
            self.rejections += 1
            out = self.fallback.parse(text, today, norm)
            out.source = "rules-fallback"
            return out
        rules = self.fallback.parse(text, today, norm)
        kept, dropped = evidenced_slots(data.get("slots", {}), norm, rules)
        self.dropped_slots += len(dropped)
        # The model often leaves out a slot the utterance plainly states (in the evaluation it dropped
        # "JVC" and "Dubai Marina"). Slots are therefore the rule parser's reading plus the model's
        # evidenced values on top. The intent and the option choice are the model's own.
        slots = {**rules.slots, **kept}
        choice = data.get("choice")
        if choice is None and data["intent"] == "choose_option":
            choice = rules.choice
        return NLUResult(intent=data["intent"], slots=slots, choice=choice, source=self.name, dropped=dropped)


def evidenced_slots(slots: dict, norm: Normalized, rules: NLUResult) -> tuple[dict, dict]:
    """Keep a model-proposed slot only if the utterance itself carries it.

    In the model smoke test every served model filled slots nobody said, including a
    phone number. So the model's slot values are treated like its replies: a value
    must be backed by an entity from the normalizer or by the rule parser's reading
    of the same words, or it is dropped.
    """
    kinds: dict[str, set] = {}
    for e in norm.entities:
        kinds.setdefault(e.kind, set()).add(e.value if not isinstance(e.value, tuple) else e.value)
    keep, dropped = {}, {}
    for k, v in slots.items():
        ok = False
        if k == "phone":
            ok = v in kinds.get("phone", set())
        elif k == "date":
            ok = v in kinds.get("date", set())
        elif k == "budget":
            ok = v in kinds.get("number", set())
        elif k == "time_window":
            ok = tuple(v) in kinds.get("time_window", set()) or v[0] in kinds.get("time", set())
        elif k in ("area", "bedrooms"):
            ok = rules.slots.get(k) == v
        (keep if ok else dropped)[k] = v
    return keep, dropped


def _strip_fences(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    return m.group(0) if m else s
