"""The deployable service: booking API, agent sessions, /config, /metrics and the demo page.

Run: uv run uvicorn api.server:app --port 8000

Configuration comes from the environment (a ConfigMap in Kubernetes):
  PARLEY_DB          sqlite path (default parley.db)
  PARLEY_NLU         rules | llm (default rules)
  PARLEY_LLM_MODEL   model id served by LM Studio (default qwen/qwen3.5-9b)
  PARLEY_LLM_URL     OpenAI-compatible base URL (default http://127.0.0.1:1234/v1)
  PARLEY_ASR_MODEL   faster-whisper size (default small); "none" disables audio turns
  PARLEY_TTS         piper | none (default piper)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from dialogue.agent import Agent
from dialogue.nlu import LLMNLU, RuleNLU
from dialogue.phrasing import Phraser
from speech.audio import SAMPLE_RATE, decode_any, read_wav, resample, wav_bytes
from speech.pipeline import VoiceCall
from speech.translit import to_latin

from .app import create_app
from .service import HOLD_TTL

WEB = Path(__file__).resolve().parent.parent / "web"


class TextIn(BaseModel):
    text: str


def effective_config() -> dict:
    cfg = {
        "nlu": os.environ.get("PARLEY_NLU", "rules"),
        "llm_model": os.environ.get("PARLEY_LLM_MODEL", "qwen/qwen3.5-9b"),
        "llm_url": os.environ.get("PARLEY_LLM_URL", "http://127.0.0.1:1234/v1"),
        "asr_model": os.environ.get("PARLEY_ASR_MODEL", "small"),
        "tts": os.environ.get("PARLEY_TTS", "piper"),
        "hold_ttl_s": int(HOLD_TTL.total_seconds()),
    }
    cfg["config_hash"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    return cfg


class _Lazy:
    """Load a heavy model on first use; remember a failure instead of retrying every request."""

    def __init__(self, factory):
        self.factory = factory
        self.value = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self.value is None and self.error is None:
                try:
                    self.value = self.factory()
                except Exception as e:  # noqa: BLE001 -- the image may be built without speech extras
                    self.error = f"{type(e).__name__}: {e}"
            return self.value


def _default_asr():
    if os.environ.get("PARLEY_ASR_MODEL", "small") == "none":
        raise RuntimeError("audio turns disabled (PARLEY_ASR_MODEL=none)")
    from speech.asr import WhisperASR

    return WhisperASR()


def _default_tts():
    if os.environ.get("PARLEY_TTS", "piper") == "none":
        raise RuntimeError("tts disabled")
    from speech.tts import PiperTTS

    tts = PiperTTS()
    tts.synthesize("ok", "en")
    return tts


def create_server(conn=None, clock=None, asr=None, tts=None):
    app = create_app(conn=conn, clock=clock)
    svc, metrics, clock = app.state.service, app.state.metrics, app.state.clock
    cfg = effective_config()
    asr_ref = _Lazy(lambda: asr) if asr is not None else _Lazy(_default_asr)
    tts_ref = _Lazy(lambda: tts) if tts is not None else _Lazy(_default_tts)
    sessions: dict[str, Agent] = {}

    def new_nlu():
        return LLMNLU(model=cfg["llm_model"], base_url=cfg["llm_url"]) if cfg["nlu"] == "llm" else RuleNLU()

    def agent_for(session_id: str) -> Agent:
        if session_id not in sessions:
            raise HTTPException(404, "unknown session")
        return sessions[session_id]

    def turn_json(transcript: str, asr_lang, reply, speak: bool) -> dict:
        t = reply.turn
        out = {
            "transcript": transcript, "transcript_latin": to_latin(transcript), "asr_lang": asr_lang,
            "dialect": t.dialect, "lang": t.lang, "action": t.action, "reply": t.text,
            "reply_latin": to_latin(t.text), "reply_source": t.source,
            "entities": [{"kind": e.kind, "value": e.value} for e in t.normalized.entities],
            "slots": t.nlu.slots, "intent": t.nlu.intent,
            "timings": {k: round(v, 3) for k, v in reply.timings.items()},
        }
        if speak and reply.speech is not None:
            out["reply_audio_wav_b64"] = base64.b64encode(wav_bytes(reply.speech.pcm)).decode()
        return out

    @app.get("/config")
    def config():
        return cfg

    @app.post("/agent/sessions", status_code=201)
    def new_session():
        sid = "web-" + uuid.uuid4().hex[:10]
        sessions[sid] = Agent(svc, svc.conn, clock, session_id=sid, nlu=new_nlu(), phraser=Phraser(clock=clock),
                              metrics=metrics)
        return {"session_id": sid}

    @app.post("/agent/sessions/{session_id}/text")
    def text_turn(session_id: str, body: TextIn, speak: bool = False):
        agent = agent_for(session_id)
        vc = VoiceCall(agent, None, tts_ref.get() if speak else None)
        reply = vc.respond(body.text)
        return turn_json(body.text, None, reply, speak)

    @app.post("/agent/sessions/{session_id}/audio")
    async def audio_turn(session_id: str, request: Request):
        agent = agent_for(session_id)
        data = await request.body()
        if not data:
            raise HTTPException(400, "empty audio")
        model = asr_ref.get()
        if model is None:
            raise HTTPException(503, f"ASR unavailable: {asr_ref.error}")
        if data[:4] == b"RIFF":
            pcm, rate = read_wav(data)
            pcm = resample(pcm, rate, SAMPLE_RATE)
        else:
            pcm = decode_any(data)
        t0 = time.perf_counter()
        vc = VoiceCall(agent, model, tts_ref.get())
        reply = vc.utterance(pcm)
        metrics.stage_seconds.labels("asr").observe(reply.timings["asr"])
        metrics.stage_seconds.labels("turn_total").observe(time.perf_counter() - t0)
        return turn_json(reply.transcript, reply.asr_lang, reply, True)

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = WEB / "index.html"
        if not page.exists():
            return HTMLResponse("<p>Parley API. The demo page is not in this build.</p>")
        return FileResponse(page)

    return app


def __getattr__(name: str):
    if name == "app":
        globals()["app"] = create_server()
        return globals()["app"]
    raise AttributeError(name)
