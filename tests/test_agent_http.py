"""Agent session endpoints used by the demo page and the CI booking gate."""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import create_server
from eval.scripts import CALLS
from speech.asr import FakeASR
from speech.audio import wav_bytes
from speech.translit import to_latin


class FakeTTS:
    def synthesize(self, text, lang):
        from speech.tts import Speech

        return Speech(np.zeros(1600, dtype=np.int16), f"fake-{lang}")


@pytest.fixture
def client(conn, clock):
    return TestClient(create_server(conn=conn, clock=clock, asr=FakeASR(["أبي أحجز معاينة في دبي مارينا"]),
                                    tts=FakeTTS()))


def test_text_call_books_over_http(client, conn):
    sid = client.post("/agent/sessions").json()["session_id"]
    call = CALLS["en-booking"]
    for t in call["turns"]:
        r = client.post(f"/agent/sessions/{sid}/text", json={"text": t["text"]})
        assert r.status_code == 200
    body = r.json()
    assert body["action"] == "confirmed" and body["lang"] == "en"
    assert conn.execute("select phone from bookings").fetchone()[0] == call["phone"]
    assert client.get("/audit/verify").json()["ok"]


def test_audio_turn_returns_transcript_in_both_scripts_and_reply_audio(client):
    sid = client.post("/agent/sessions").json()["session_id"]
    wav = wav_bytes(np.zeros(16000, dtype=np.int16))
    r = client.post(f"/agent/sessions/{sid}/audio", content=wav, headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "أبي أحجز معاينة في دبي مارينا"
    assert body["transcript_latin"] == to_latin(body["transcript"])
    assert body["dialect"] == "ar-gulf" and body["lang"] == "ar"
    assert body["reply_latin"] and body["reply_audio_wav_b64"]
    assert base64.b64decode(body["reply_audio_wav_b64"])[:4] == b"RIFF"


def test_unknown_session_is_404(client):
    assert client.post("/agent/sessions/nope/text", json={"text": "hi"}).status_code == 404


def test_config_endpoint_reports_the_effective_config(client):
    cfg = client.get("/config").json()
    assert set(cfg) >= {"asr_model", "llm_model", "nlu", "hold_ttl_s", "config_hash"}


def test_transliteration():
    assert to_latin("دبي مارينا") == "dby maryna"
    assert to_latin("أبي أحجز") == "aby a7jz"
    assert to_latin("hello 16:00") == "hello 16:00"


def test_demo_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Hold to talk" in r.text and "/agent/sessions" in r.text
