"""The wrapper against a fake LITranscriber that speaks its v1 WebSocket and /health."""

import asyncio
import base64
import io
import json
import threading
import wave
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient
from websockets.asyncio.server import serve

import kurukshetra_lit as k

MODE = {"lit": "ok"}  # ok | error | close | unhealthy
SEEN: dict = {}


async def fake_lit(conn) -> None:
    SEEN["path"], frames = conn.request.path, []
    async for raw in conn:
        msg = json.loads(raw)
        if msg["type"] == "audio_frame":
            frames.append(msg)
            if MODE["lit"] == "close":
                await conn.close(1013, "try again later")
                return
            if MODE["lit"] == "error" and len(frames) == 1:
                await conn.send(json.dumps({"type": "error", "error_code": "STREAM_RESET", "error_message": "gap"}))
            if MODE["lit"] == "ok" and len(frames) == 2:  # an utterance the VAD closed mid-clip
                await conn.send(json.dumps({"type": "transcript", "is_final": False, "text": "पह"}))
                await conn.send(json.dumps({"type": "transcript", "is_final": True, "text": "पहला"}))
        elif msg["type"] == "flush":
            SEEN["frames"] = frames
            await asyncio.sleep(0.1)  # the flushed utterance's final lands after the flush
            await conn.send(json.dumps({"type": "transcript", "is_final": True, "text": "दूसरा"}))


def health(conn, request):
    if request.path == "/health":
        ok = MODE["lit"] != "unhealthy"
        return conn.respond(HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE, "{}\n")
    return None


@pytest.fixture(scope="module")
def lit_url():
    ready, port = threading.Event(), []

    async def run():
        async with serve(fake_lit, "127.0.0.1", 0, process_request=health) as server:
            port.append(server.sockets[0].getsockname()[1])
            ready.set()
            await asyncio.Future()

    threading.Thread(target=asyncio.run, args=(run(),), daemon=True).start()
    assert ready.wait(5)
    return f"http://127.0.0.1:{port[0]}"


@pytest.fixture
def client(lit_url, monkeypatch):
    monkeypatch.setattr(k, "LIT_URL", lit_url)
    monkeypatch.setattr(k, "FRAME_S", 0)  # no real-time pacing in tests
    monkeypatch.setattr(k, "QUIET_S", 0.5)
    monkeypatch.setattr(k, "AUTH_TOKEN", "")
    MODE["lit"] = "ok"
    SEEN.clear()
    return TestClient(k.app)


def wav(seconds=0.1, rate=16000, channels=1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate) * channels)
    return buf.getvalue()


def post(client, audio=None, language="hi", model=k.MODEL, headers=None):
    return client.post(
        "/transcribe",
        files={"audio": ("clip.wav", wav() if audio is None else audio, "audio/wav")},
        data={"language": language, "model": model},
        headers=headers or {},
    )


def test_hinglish_goes_to_lit_as_hindi_and_finals_are_joined(client):
    r = post(client, language="hi-en")
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "पहला दूसरा"}  # the partial is ignored, the post-flush final kept
    assert SEEN["path"].endswith("?language=hi")
    frames = SEEN["frames"]  # 0.1 s = 3200 bytes -> 3 full frames + 1 zero-padded
    assert [f["sequence"] for f in frames] == [0, 1, 2, 3]
    assert all(len(base64.b64decode(f["audio"])) == 1024 for f in frames)
    assert base64.b64decode(frames[-1]["audio"])[128:] == b"\0" * 896


def test_lit_error_is_a_retryable_502(client):
    MODE["lit"] = "error"
    r = post(client)
    assert r.status_code == 502 and "STREAM_RESET" in r.text


def test_lit_closing_1013_or_down_is_a_retryable_503(client, monkeypatch):
    MODE["lit"] = "close"
    assert post(client).status_code == 503
    monkeypatch.setattr(k, "LIT_URL", "http://127.0.0.1:1")
    assert post(client).status_code == 503
    assert client.get("/ping").status_code == 503


def test_ping_follows_lit_health(client):
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json()["contract"] == "kurukshetra-asr/1" and "hi-en" in r.json()["languages"]
    MODE["lit"] = "unhealthy"
    assert client.get("/ping").status_code == 503


def test_empty_clip_is_an_empty_answer_without_calling_lit(client, monkeypatch):
    monkeypatch.setattr(k, "LIT_URL", "http://127.0.0.1:1")
    r = post(client, audio=wav(seconds=0))
    assert r.status_code == 200 and r.json() == {"text": ""}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "whisper"},
        {"language": "ta"},
        {"audio": wav(rate=8000)},
        {"audio": wav(channels=2)},
        {"audio": b"not a wav"},
    ],
)
def test_bad_requests_are_final_400s(client, kwargs):
    assert post(client, **kwargs).status_code == 400


def test_bearer_token_when_set(client, monkeypatch):
    monkeypatch.setattr(k, "AUTH_TOKEN", "s3cret")
    assert post(client).status_code == 401
    assert post(client, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert post(client, headers={"Authorization": "Bearer s3cret"}).status_code == 200
