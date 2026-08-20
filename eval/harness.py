"""Shared evaluation machinery: caller audio, cached streaming ASR per speech config and stress arm, the scripted call loop.

Every harness (run_eval, mtva, trace, toolnoise) plays calls through run_call with a different transcript source.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from api.audit import entries, verify_chain
from api.clock import DUBAI, FixedClock
from api.db import connect
from api.seed import seed
from api.service import BookingService
from dialogue.agent import Agent
from dialogue.nlu import RuleNLU
from dialogue.phrasing import Phraser
from speech.audio import SAMPLE_RATE, read_wav, silence
from speech.langid import identify, reply_language
from speech.pipeline import VoiceCall
from speech.vad import FRAME, EnergyVAD

from .acoustics import stress
from .synth import render_to

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
AUDIO = ROOT / "audio" / "calls"
ANCHOR = datetime(2026, 10, 1, 9, 0, tzinfo=DUBAI)
DIALECTS = ["en", "ar-gulf", "ar-msa", "mixed"]
CALL_KINDS = {"en": "English", "gulf": "Gulf Arabic", "msa": "MSA", "switch": "code-switched"}
# v1 allowed 12. The v2 calls have eight lines, so 16 leaves the same room for repeats.
MAX_TURNS = 16
HANGOVER_S = EnergyVAD().hangover_frames * FRAME / SAMPLE_RATE

_LATIN = re.compile(r"[A-Za-z]")
_ARABIC = re.compile(r"[؀-ۿ]")


# ---------------------------------------------------------------------------
# Caller audio and streaming ASR
# ---------------------------------------------------------------------------

def render_turn(call_id: str, i: int, turn: dict, voice: str, attempt: int = 0) -> Path:
    """Attempt 0 is the scripted voice. A caller asked again repeats the line 15 percent slower each time."""
    base = 1.15 if voice == "piper-slow" else 1.0
    v = "piper" if voice == "piper-slow" else voice
    name = f"{i:02d}.wav" if attempt == 0 else f"{i:02d}-r{attempt}.wav"
    if v.startswith("sapi"):
        return render_to(AUDIO / call_id / name, turn["segments"], v, sapi_rate=-2 * attempt)
    scale = base + 0.15 * attempt
    return render_to(AUDIO / call_id / name, turn["segments"], v, length_scale=None if scale == 1.0 else scale)


def stream_asr(asr, pcm: np.ndarray, lang_hint: str | None) -> dict:
    """Play the clip through the VAD-driven recognizer as 20 ms frames."""
    from speech.asr import StreamingRecognizer

    rec = StreamingRecognizer(asr, early_final=True)
    rec.lang_hint = lang_hint
    stream = np.concatenate([silence(0.3), pcm, silence(HANGOVER_S + 0.3)])
    texts, decodes = [], []
    for k in range(len(stream) // FRAME):
        for ev in rec.push(stream[k * FRAME:(k + 1) * FRAME]):
            if ev.kind == "final":
                texts.append(ev.text)
                decodes.append([round(ev.decode_seconds, 4), round(ev.hidden_seconds, 3)])
    for ev in rec.flush():
        texts.append(ev.text)
        decodes.append([round(ev.decode_seconds, 4), 0.0])
    return {"text": " ".join(t for t in texts if t).strip(), "decode_s": round(sum(d for d, _ in decodes), 4),
            "decodes": decodes, "segments": len(decodes)}


_TALKERS: list[np.ndarray] = []


def talkers() -> list[np.ndarray]:
    """Babble and competing speech come from the ASR smoke-set clips, which are not in any scripted call."""
    if not _TALKERS:
        from .smoke_set import SMOKE_SET

        for i, item in enumerate(SMOKE_SET):
            path = render_to(ROOT / "audio" / "talkers" / f"{i:02d}.wav", item["segments"], "piper")
            _TALKERS.append(read_wav(str(path))[0])
    return _TALKERS


class ASRCache:
    """Streaming ASR results for one speech config and one stress arm, keyed by clip and language hint."""

    def __init__(self, config: str, arm: str = "clean"):
        self.config, self.arm = config, arm
        self.path = OUT / "asr" / f"{config}@{arm}.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.model = None
        self.new = 0

    def _asr(self):
        if self.model is None:
            from speech.backends import CONFIGS
            from speech.backends.local import build_asr

            self.model = build_asr(CONFIGS[self.config])
        return self.model

    def get(self, call_id: str, line: int, attempt: int, hint: str | None) -> dict:
        from .scripts import CALLS

        call = CALLS[call_id]
        path = render_turn(call_id, line, call["turns"][line], call["voice"], attempt)
        key = f"{call_id}/{line}/{attempt}/{hint}:" + hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        if key not in self.data:
            pcm = read_wav(str(path))[0]
            if self.arm != "clean":
                pcm = stress(pcm, self.arm, key=f"{call_id}/{line}/{attempt}", talkers=talkers())
            self.data[key] = stream_asr(self._asr(), pcm, hint)
            self.new += 1
            if self.new % 25 == 0:
                self.save()
        return self.data[key]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")


RESULTS = ROOT / "results.json"


def load_results() -> dict:
    r = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    return {} if "configs" in r else r  # a v1 file is replaced, not merged; v1 numbers live in results_v1.json


def update_results(section: str, data, merge: bool = False) -> None:
    """Each harness owns one section of eval/results.json, so they can be rerun one at a time."""
    r = load_results()
    r[section] = {**r.get(section, {}), **data} if merge else data
    RESULTS.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")


def reference_source(call_id: str, line: int, attempt: int, hint: str | None) -> dict:
    from .scripts import CALLS

    return {"text": CALLS[call_id]["turns"][line]["text"], "decode_s": 0.0, "decodes": [], "segments": 1}


def cache_source(cache: ASRCache):
    return cache.get


# ---------------------------------------------------------------------------
# The scripted caller
# ---------------------------------------------------------------------------

def next_line(call: dict, action: str, args: dict, last: int, played: list[int]) -> int | None:
    """The caller answers the question the agent asked with the line that carries the answer."""
    turns = call["turns"]
    if action in ("confirmed", "goodbye", "cancelled", "unavailable"):
        return None  # after "unavailable" the caller has been promised a callback and hangs up
    if action == "listen":
        return last  # told to go on after saying everything, a caller says it again
    nxt = last + 1
    if nxt < len(turns) and turns[nxt].get("volunteer") and nxt not in played:
        return nxt
    if action == "redirect":
        action, args = ("offer", {}) if args.get("next") == "choice" else ("ask_slot", {"slot": args.get("next")})
    if action == "ask_slot":
        lines = [j for j, t in enumerate(turns) if args.get("slot") in t["slots"]]
        fresh = [j for j in lines if j not in played]
        if fresh or lines:
            return (fresh or lines)[0]
    if action == "offer":
        lines = [j for j, t in enumerate(turns) if t["intent"] == "choose_option"]
        if lines:
            return lines[0]
    if action == "ask_confirm":
        lines = [j for j, t in enumerate(turns) if t["intent"] == "confirm"]
        if lines:
            return lines[-1]
    return nxt if nxt < len(turns) else None


def gold_constraints(call: dict) -> dict:
    g: dict = {}
    for t in call["turns"]:
        g.update(t["slots"])
        if t.get("choice"):
            g["choice"] = t["choice"]
    return g


def _search_matches(searched: dict | None, g: dict) -> bool:
    if not searched:
        return False
    want = {k: g[k] for k in ("area", "bedrooms", "date") if k in g}
    if "budget" in g:
        want["max_rent"] = g["budget"]
    if "time_window" in g:
        want["start"], want["end"] = g["time_window"]
    return searched == want


def _slot_violation(slot: dict, g: dict) -> str | None:
    if "area" in g and slot["area"] != g["area"]:
        return "area"
    if "bedrooms" in g and slot["bedrooms"] != g["bedrooms"]:
        return "bedrooms"
    if "date" in g and slot["starts_at"][:10] != g["date"]:
        return "date"
    if "time_window" in g and not (g["time_window"][0] <= slot["starts_at"][11:16] < g["time_window"][1]):
        return "time"
    if "budget" in g and slot["annual_rent_aed"] > g["budget"]:
        return "budget"
    return None


def wrong_actions(call: dict, events: list[dict]) -> list[dict]:
    """A committed action the caller did not ask for: a hold or booking outside what they said, or a cancel.

    Wrong actions are TRACE's key harm (arXiv 2609.29452): worse than failing, because the caller is misled.
    """
    g = gold_constraints(call)
    out = []
    for e in events:
        why = None
        if e["kind"] == "cancel":
            why = "cancel"
        elif e["kind"] == "hold":
            why = _slot_violation(e["slot"], g)
            if why is None and e.get("consistent") and g.get("choice") and len(e["offered"]) >= g["choice"]:
                if e["offered"][g["choice"] - 1] != e["slot"]["slot_id"]:
                    why = "not the option chosen"
        elif e["kind"] == "confirm":
            why = "phone" if e["phone"] != g.get("phone") else _slot_violation(e["slot"], g)
        if why:
            out.append({"kind": e["kind"], "why": why})
    return out


def split_turn(text: str) -> list[str]:
    """MTVA's split caller: the same words delivered as two messages, cut near the middle."""
    words = text.split()
    if len(words) < 4:
        return [text]
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def expected_reply_language(gold_text: str) -> str:
    return reply_language(identify(gold_text))


def script_ok(reply: str, lang: str) -> bool:
    """An Arabic reply carries no Latin letters and an English reply no Arabic letters."""
    return not _LATIN.search(reply) if lang == "ar" else not _ARABIC.search(reply)


# ---------------------------------------------------------------------------
# One call
# ---------------------------------------------------------------------------

def make_dialogue(config: str, clock, llm_model: str = "qwen/qwen3.5-9b"):
    from dialogue.nlu import LLMNLU
    from dialogue.phrasing import LLMPhraser

    if config in ("rules", "rules-nocarry"):
        nlu = RuleNLU()
    elif config.startswith("jev"):
        from dialogue.jev import JevNLU

        nlu = JevNLU(jev_decider())
    else:
        nlu = LLMNLU(model=llm_model, timeout=60)
    phrasing = config in ("llm+phrasing", "jev+llm")
    phraser = Phraser(llm=LLMPhraser(model=llm_model, timeout=60) if phrasing else None, clock=clock)
    if config == "jev+llm":
        from dialogue.jev import JevPhraser

        phraser = JevPhraser(phraser, jev_decider())
    return nlu, phraser


_JEV: dict = {}


def jev_decider():
    if "d" not in _JEV:
        from dialogue.jev import HFLogitModel, JevDecider

        _JEV["d"] = JevDecider(HFLogitModel())
    return _JEV["d"]


def new_world():
    conn = connect(":memory:")
    clock = FixedClock(ANCHOR)
    seed(conn, anchor=ANCHOR.date(), rng_seed=7)
    return conn, clock, BookingService(conn, clock)


def run_call(call_id: str, call: dict, source, dialogue: str = "rules", tts=None, split: bool = False,
             service=None, keep_audio: bool = False, llm_model: str = "qwen/qwen3.5-9b", observe=None) -> dict:
    conn, clock, svc = new_world()
    if service is not None:
        svc = service(svc)
    nlu, phraser = make_dialogue(dialogue, clock, llm_model)
    agent = Agent(svc, conn, clock, session_id=f"{dialogue}:{call_id}", nlu=nlu, phraser=phraser,
                  carry=not dialogue.endswith("-nocarry"))
    vc = VoiceCall(agent, None, tts)
    g = gold_constraints(call)

    turns_out, events, played, line = [], [], [], 0
    while line is not None and len(played) < MAX_TURNS:
        attempt = played.count(line)
        played.append(line)
        gold = call["turns"][line]
        a = source(call_id, line, attempt, agent.lang if len(played) > 1 else None)
        pieces = split_turn(a["text"]) if split else [a["text"]]
        for k, text in enumerate(pieces):
            s = agent.state
            before = (s.hold and s.hold["hold_id"], s.booking and s.booking["booking_id"], s.cancelled)
            reply = vc.respond(text, asr_seconds=a["decode_s"] if k == len(pieces) - 1 else 0.0)
            t = reply.turn
            if s.hold and s.hold["hold_id"] != before[0]:
                events.append({"kind": "hold", "slot": s.held_slot, "offered": [x["slot_id"] for x in s.offered or []],
                               "consistent": _search_matches(s.searched_with, g), "turn": len(turns_out)})
            if s.booking and s.booking["booking_id"] != before[1]:
                events.append({"kind": "confirm", "slot": s.booked_slot, "phone": s.slots.get("phone"),
                               "turn": len(turns_out)})
            if s.cancelled and not before[2]:
                events.append({"kind": "cancel", "turn": len(turns_out)})
            expected = expected_reply_language(gold["text"])
            turns_out.append({
                "line": line, "attempt": attempt, "fragment": k if split else None, "dialect": gold["dialect"],
                "ref": gold["text"], "hyp": text, "gold_intent": gold["intent"], "intent": t.nlu.intent,
                "gold_slots": {**gold["slots"], **({"choice": gold["choice"]} if gold["choice"] else {})},
                "slots": {**t.nlu.slots, **({"choice": t.nlu.choice} if t.nlu.choice else {})},
                "dialect_pred": t.dialect, "action": t.action, "action_args": t.action_args, "reply": t.text,
                "reply_lang": t.lang, "expected_lang": expected,
                "adherent": t.lang == expected and script_ok(t.text, t.lang),
                "reply_source": t.source, "rejected": [v.kind for v in t.rejected], "rejected_text": t.rejected_text, "draft": t.draft,
                "ungrounded": [v.kind for v in t.ungrounded], "nlu_source": t.nlu.source,
                "dropped_slots": list(t.nlu.dropped), "decodes": a.get("decodes", []),
                "timings": {k2: round(v, 4) for k2, v in reply.timings.items()},
                "reply_seconds": round(reply.speech.seconds, 2) if reply.speech is not None else None,
            })
            if observe is not None:
                turns_out[-1].update(observe(agent, t))
            if keep_audio and reply.speech is not None:
                turns_out[-1]["_speech"] = reply.speech.pcm
        line = next_line(call, t.action, t.action_args, line, played)

    booking = conn.execute("select status, phone from bookings").fetchall()
    completed = booking == [("confirmed", call["phone"])] and verify_chain(conn)
    return {"call_id": call_id, "kind": call_id.split("-")[0], "dialogue": dialogue, "completed": completed,
            "bookings": booking, "audit_ok": verify_chain(conn), "audit_entries": len(entries(conn)),
            "turns": turns_out, "events": events, "wrong_actions": wrong_actions(call, events),
            "spoken_ungrounded": sum(bool(t["ungrounded"]) for t in turns_out),
            "callbacks": sum(e["action"] == "callback_requested" for e in entries(conn)),
            "judge_verdicts": [{"grounded": v.grounded, "source": v.source, "p": round(v.probability, 4),
                                "overruled": v.overruled} for v in getattr(phraser, "verdicts", [])]}
