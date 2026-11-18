# kurukshetra-lit

Puts LIT's Nemotron 3.5 (`nvidia/nemotron-3.5-asr-streaming-0.6b`, served by LITranscriber's `optimized/nemotron-asr`
branch) into the [ASR Kurukshetra](https://huggingface.co/spaces/kapturecx/ASR-Kurukshetra) arena. It speaks the
arena's contract, `kurukshetra-asr/1`, on one side and LITranscriber's v1 WebSocket on the other.

```
arena ──HTTPS──▶ kurukshetra-lit  GET /ping · POST /transcribe
                       │  ws://…/ws/{call_id}?language=hi   (1024-byte frames at real-time pace, then flush)
                       ▼
                 LITranscriber + Nemotron
```

## Run it

On the Nemotron VM, next to LITranscriber (uv is already installed there):

```bash
uv sync --no-dev
LIT_URL=http://localhost:8080 AUTH_TOKEN=<long random string> PORT=8000 uv run kurukshetra-lit
```

| Env var | Default | Meaning |
|---|---|---|
| `LIT_URL` | `http://localhost:8080` | LITranscriber's base URL; the WebSocket URL is derived from it (`http`→`ws`, `https`→`wss`) |
| `AUTH_TOKEN` | empty | Bearer token the arena must send on `/transcribe`. Empty accepts anyone; set it on any reachable box |
| `PORT` | `8000` | Port to listen on |

The arena must be able to reach this port over HTTPS (load balancer or reverse proxy in front of it).

Smoke test from the box:

```bash
curl -s localhost:8000/ping
curl -s -X POST localhost:8000/transcribe -H "Authorization: Bearer $AUTH_TOKEN" \
  -F audio=@clip.wav -F language=hi -F model=nemotron-3.5-asr-streaming-0.6b
```

## Register the gladiator

On the arena's Gladiators page:

| Field | Value |
|---|---|
| `endpoint` | the public base URL of this server |
| `model` | `nemotron-3.5-asr-streaming-0.6b` |
| `languages` | the ones you want to fight in, e.g. `hi, en, hi-en` (all supported: `GET /ping`) |
| `auth_token` | the same `AUTH_TOKEN` |
| `max_concurrency` | `8` (each clip holds one LIT call for its whole length) |
| `timeout_s` | longest clip in the battleground + 15 s, e.g. `90` |

## How it behaves

- **One clip = one LIT call.** The WAV is replayed in real time, so LIT's VAD and Nemotron's 320 ms streaming run as on
  a live call, and every final of the call is joined with spaces. A clip takes its own length plus about 3 s.
- **End of clip.** LIT sends nothing to say a flush is done, so the answer is returned once LIT has been silent for
  3 s after the flush (`QUIET_S`). A final later than that would be lost.
- **Languages.** Nemotron's 15 locales pass through. `hi-en` (Hinglish) is sent to LIT as `hi`, so the answer is
  Devanagari Hindi. `ta` and anything else not in `/ping` is a 400.
- **Errors.** LIT down, unhealthy at the handshake, or closing with 1013 → 503. A LIT `error` message
  (`TRANSCRIPTION_BACKPRESSURE`, `STREAM_RESET`, `PROCESSING_FAILED`) → 502. The arena retries both twice. A wrong
  model, language or audio format → 400, which it doesn't retry. A wrong token → 401.
- `/ping` answers 200 only while LIT's `/health` does, so a down box shows as inactive instead of failing clips.

## Test

```bash
uv run pytest -q
```

The tests run against a fake LITranscriber; nothing touches a real box.
