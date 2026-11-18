"""kurukshetra-asr/1 in front of LITranscriber: the arena posts a WAV, LIT's Nemotron streams it back as text.

Each clip is one LIT call: the WAV is replayed over LIT's v1 WebSocket at real-time pace, so LIT's VAD and
Nemotron's streaming run exactly as they do on a live call, and the call's finals are joined into the answer.
"""

import asyncio
import base64
import hmac
import io
import json
import logging
import os
import time
import urllib.request
import uuid
import wave

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

LIT_URL = os.environ.get("LIT_URL", "http://localhost:8080").rstrip("/")  # LITranscriber's HTTP base
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")  # the bearer token registered with the arena; empty = no auth

MODEL = "nemotron-3.5-asr-streaming-0.6b"
# What Nemotron 3.5 on LITranscriber accepts (LITranscriber docs/nemotron-serving.md, "Language").
LIT_LANGUAGES = ["en", "es", "fr", "it", "pt", "nl", "de", "tr", "ru", "ar", "hi", "ja", "ko", "vi", "uk"]
LANGUAGE_MAP = {"hi-en": "hi"}  # Hinglish has no Nemotron locale; it is transcribed as Hindi
LANGUAGES = LIT_LANGUAGES + list(LANGUAGE_MAP)

SAMPLE_RATE = 16000
FRAME_BYTES = 1024  # 32 ms of PCM16, the frame LITranscriber's clients send
FRAME_S = 0.032
# LIT sends no "done" after flush, so the call ends after this much silence from it.
# shortcut: a final later than QUIET_S after flush is lost; upgrade when LIT acks a flush.
QUIET_S = 3.0
MAX_AUDIO_S = 600  # the arena's longest timeout_s: a longer clip can never answer in time
MAX_WAV_BYTES = MAX_AUDIO_S * SAMPLE_RATE * 2 + 4096  # PCM plus room for the header

log = logging.getLogger("kurukshetra_lit")
app = FastAPI(title="kurukshetra-lit")


@app.get("/ping")
async def ping() -> dict:
    if not await asyncio.to_thread(_lit_healthy):
        raise HTTPException(503, "LITranscriber is not healthy")
    return {"status": "pong", "contract": "kurukshetra-asr/1", "model": MODEL, "languages": LANGUAGES}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(...),
    model: str = Form(...),
    authorization: str | None = Header(None),
) -> dict:
    if AUTH_TOKEN and not hmac.compare_digest((authorization or "").encode(), f"Bearer {AUTH_TOKEN}".encode()):
        raise HTTPException(401, "missing or wrong bearer token")
    lit_language = LANGUAGE_MAP.get(language.lower(), language.lower())
    if model != MODEL or lit_language not in LIT_LANGUAGES:
        raise HTTPException(400, f"unsupported model {model!r} or language {language!r}")
    pcm = _read_wav(await audio.read(MAX_WAV_BYTES + 1))
    if not pcm:
        return {"text": ""}
    finals = await transcribe_pcm(pcm, lit_language)
    return {"text": " ".join(t.strip() for t in finals if t.strip())}


async def transcribe_pcm(pcm: bytes, language: str) -> list[str]:
    """One LIT call for the clip; its final transcripts in order. Raises HTTPException 502/503 on failure."""
    call_id = f"krk-{uuid.uuid4().hex}"
    finals: list[str] = []
    heard = [time.monotonic()]  # when LIT last sent anything
    try:
        async with connect(f"ws{LIT_URL[4:]}/ws/{call_id}?language={language}") as ws:
            reader = asyncio.create_task(_read(ws, finals, heard))
            try:
                await _stream(ws, call_id, pcm, reader)
                await _wait_quiet(reader, heard)
            finally:
                reader.cancel()
    except (OSError, TimeoutError, InvalidHandshake, ConnectionClosed) as e:
        # Refused, a 503 handshake, or a 1013 close: all "try again", which the arena retries.
        raise HTTPException(503, f"LITranscriber unavailable: {type(e).__name__}: {e}") from None
    return finals


async def _read(ws, finals: list[str], heard: list[float]) -> None:
    async for raw in ws:
        heard[0] = time.monotonic()
        msg = json.loads(raw)
        if msg.get("type") == "transcript" and msg.get("is_final"):
            finals.append(msg.get("text") or "")
        elif msg.get("type") == "error":
            # BACKPRESSURE, STREAM_RESET, PROCESSING_FAILED…: the arena retries a 502.
            raise HTTPException(502, f"LITranscriber {msg.get('error_code')}: {msg.get('error_message')}")


async def _stream(ws, call_id: str, pcm: bytes, reader: asyncio.Task) -> None:
    """Send the clip as 32 ms frames on a real-time clock, then flush. Stops early if LIT reported an error."""
    start = time.monotonic()
    for seq, i in enumerate(range(0, len(pcm), FRAME_BYTES)):
        if reader.done():
            break
        frame = pcm[i : i + FRAME_BYTES].ljust(FRAME_BYTES, b"\0")
        await ws.send(json.dumps({
            "type": "audio_frame",
            "call_id": call_id,
            "sequence": seq,
            "audio": base64.b64encode(frame).decode(),
            "timestamp": int(time.time() * 1000),
            "sample_rate": SAMPLE_RATE,
        }))
        await asyncio.sleep(start + (seq + 1) * FRAME_S - time.monotonic())
    if reader.done():
        reader.result()  # raises LIT's error
    await ws.send(json.dumps({"type": "flush"}))


async def _wait_quiet(reader: asyncio.Task, heard: list[float]) -> None:
    """Return once LIT has been silent for QUIET_S since the flush; raise if the reader failed first."""
    flushed = time.monotonic()
    while not reader.done():
        idle = time.monotonic() - max(flushed, heard[0])
        if idle >= QUIET_S:
            return
        await asyncio.wait({reader}, timeout=QUIET_S - idle)
    reader.result()


def _read_wav(data: bytes) -> bytes:
    if len(data) > MAX_WAV_BYTES:
        raise HTTPException(413, f"clip longer than {MAX_AUDIO_S} s")
    try:
        with wave.open(io.BytesIO(data)) as w:
            shape = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            pcm = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise HTTPException(400, f"audio is not a readable WAV: {e or type(e).__name__}") from None
    if shape != (1, 2, SAMPLE_RATE):
        raise HTTPException(400, f"audio must be mono 16-bit 16 kHz PCM, got channels/bytes/rate {shape}")
    return pcm


def _lit_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{LIT_URL}/health", timeout=3) as r:
            return r.status == 200
    except OSError:  # refused, timed out, or a 503 from an unhealthy worker
        return False


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    if not AUTH_TOKEN:
        log.warning("AUTH_TOKEN is not set: /transcribe accepts any caller")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
