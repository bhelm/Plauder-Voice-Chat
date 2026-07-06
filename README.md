# 🎙️ Plauder

Speech-to-speech chat in the browser: **microphone → STT → LLM → TTS → speaker** —
with turn-taking (debounce/coalescing), barge-in, low latency through
end-to-end streaming, wake word, text input and image uploads.

The server is a clean Python package (`plauder/`) with **pluggable
backends** for STT, TTS and LLM. Via `.env` you can switch between cloud APIs
(OpenAI, Fireworks), local models (faster-whisper, OmniVoice) and gateway
backends (OpenClaw) — without changing any code.

---

## Features

- 🎙️ **Voice-to-Voice** in the browser, no app needed (just `static/index.html`).
- ⚡ **Streaming for low latency** — LLM tokens are synthesized sentence by
  sentence immediately and played back progressively as audio chunks; the microphone
  streams along live and delivers interim transcripts (see [Streaming](#streaming--latency)).
- 🔔 **Wake word** — selectable input mode (alongside VAD & push-to-talk): the AI
  only reacts to "Antonia …", everything else is discarded; with a conversation window
  for follow-up questions. Off by default at startup, toggleable in the UI.
- 🔒 **Voice lock** — optional speaker verification: enroll your voice from the UI,
  and only *your* voice is transcribed — other people and background voices (TV,
  kids) are dropped before the LLM. Works with any STT backend; needs
  `sherpa-onnx` + a small CAM++/WeSpeaker ONNX model (see [Voice lock](#voice-lock)).
- 🔀 **Pluggable backends** — STT/TTS/LLM each selectable independently via `.env`,
  cloud or local/GPU, any combination allowed.
- 🗣️ **Turn-taking & barge-in** — debounce/coalescing, interrupting stops
  playback immediately.
- 💬 Text input and 🖼️ image uploads (multimodal) in parallel with the voice.

---

## Getting Started

> Full walkthrough — including local backends, voice lock, systemd and reverse
> proxy — in the **[installation guide](INSTALL.md)**. Below is the short version.

### Requirements

- **Python 3.11** and a microphone-capable browser (Chrome/Edge/Firefox).
- An **OpenAI API key** (STT/TTS in the cloud default) and an **OpenAI-compatible
  LLM endpoint** (e.g. Fireworks, or a local server). For fully local
  operation see [Local / GPU](#local--gpu).

### Quick start (cloud default, no GPU)

```bash
cp .env.example .env          # 1) Keys eintragen (mind. OPENAI_API_KEY + LLM_*)
./start.sh            # 2) legt venv an, installiert Deps, startet Server
```

Then open in the browser: **http://localhost:8319**, allow the microphone and speak.

By default you start in **VAD mode** (everything you say is sent).
In the **Input mode** box you can switch to **wake word** — then you begin
with **"Antonia, …"** (e.g. "Antonia, what time is it?") and may keep talking for
~8 s right afterwards without saying "Antonia" again. If you want the UI to start
in wake mode straight away: `WAKE_WORD_ENABLED=1` in the `.env`.

`start.sh` is idempotent (callable any number of times) and listens on
`${HOST}:${PORT}` — the `.env.example` template sets `0.0.0.0:8319`; without a
`HOST` entry the server binds `127.0.0.1` only.

### Minimal `.env` (cloud)

```bash
# STT + TTS via OpenAI
OPENAI_API_KEY=sk-...
# LLM via an OpenAI-compatible endpoint (Fireworks, a local server, …)
LLM_BACKEND=openai_compat
LLM_BASE_URL=https://api.fireworks.ai/inference/v1
LLM_API_KEY=fw_...
LLM_MODEL=accounts/fireworks/models/glm-5p2
```

All options are documented in [`.env.example`](.env.example).

### Language (`APP_LANGUAGE`)

`APP_LANGUAGE` sets **both** the UI language of the browser client (which is now
fully internationalized, English + German) **and** the assistant's default spoken
language as well as the STT default language.

| Variable | Values | Default | Meaning |
|---|---|---|---|
| `APP_LANGUAGE` | `en` · `de` | `en` | UI language + assistant/STT default language |

The browser client auto-renders in the selected language, so no extra setup is
needed. `APP_LANGUAGE=de` restores the previous German experience (German UI and
a German-speaking assistant).

### Sub-path / reverse proxy (`BASE_PATH`)

To run the app under a sub-path instead of the domain root (e.g. behind a shared
reverse proxy), set `BASE_PATH`:

```bash
BASE_PATH=/voice
```

All routes then live under that prefix (`/voice/`, `/voice/ws`, `/voice/upload`,
`/voice/static/…`, `/voice/uploads/…`, `/voice/healthz`), and the server injects
the prefix into the page so the client builds its WebSocket, upload and asset
URLs accordingly. Empty (default) serves everything at the root.

Configure the proxy to forward the sub-path **without stripping** it — the app
expects to see `/voice/...`. Example for nginx (note: WebSockets need the upgrade
headers):

```nginx
location /voice/ {
    proxy_pass         http://127.0.0.1:8319;   # no trailing slash → prefix kept
    proxy_http_version 1.1;
    proxy_set_header   Upgrade $http_upgrade;
    proxy_set_header   Connection "upgrade";
    proxy_set_header   Host $host;
}
```

### Health check

```bash
curl http://localhost:8319/healthz   # 200 + aktive Backends
```

---

## Architecture

```
                         Browser (static/index.html)
                          │   ▲
   16kHz f32 PCM-Frames    │   │  PCM-Chunks (VCT2, gestreamt) / WAV (VCT1)
   (VAD live / Push-to-Talk)│  │  + JSON-Events
                          ▼   │
   ┌───────────────────────────────────────────────────────────────┐
   │                  plauder/server.py  (HTTP/WS-Layer)          │
   │   Routen: / , /healthz , /ws , /upload , /uploads/… , /static/… │
   │   ws_handler ─ Audio-Segmente/-Frames & Text                   │
   │       ├─► turn_state.TurnState   (Debounce + Coalescing)        │
   │       ├─► wake                   (Wake-Word-Gate)               │
   │       ├─► sanitizer              (Ghost-Filter, Merge, NO_REPLY)│
   │       ├─► audio                  (PCM/WAV/numpy, Satz-Splitter) │
   │       └─► session.ConversationManager (Verlauf pro Session-Key) │
   └───────────────────────────────────────────────────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
      ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
      │  STTBackend  │     │  LLMBackend  │     │  TTSBackend  │
      │ .transcribe  │     │ .chat        │     │ .synth       │
      │ .—           │     │ .chat_stream │     │ .synth_stream│
      ├──────────────┤     ├──────────────┤     ├──────────────┤
      │ openai_api   │     │ openai_compat│     │ openai_api   │   ← Cloud
      │ whisper_local│     │ openclaw     │     │ omnivoice_   │   ← lokal/GPU
      │ (faster-     │     │ (gateway)    │     │  local       │     (lazy)
      │  whisper)    │     │              │     │ (omnivoice)  │
      └──────────────┘     └──────────────┘     └──────────────┘
        STT_BACKEND          LLM_BACKEND          TTS_BACKEND
```

**Lazy imports:** Heavy GPU deps (`faster_whisper`, `torch`, `omnivoice`) are
imported exclusively in `load()` of the respective backend — and only when it is
active. In the cloud default no GPU code is ever loaded.

### Pipeline per turn

1. The browser sends speech frames (VAD live / PTT) and/or text over the WebSocket.
2. `TurnState` collects inputs within the debounce window (`DEBOUNCE_MS`); new input
   aborts in-flight LLM/TTS calls (barge-in) — with wake word or voice lock
   active only AFTER the segment passes the gate, so foreign voices can't
   interrupt a running reply.
3. STT → text; Whisper hallucinations are filtered out.
4. **Voice-lock gate** (only when enrolled): segments not spoken by the owner
   are dropped; mixed segments are trimmed to the owner's part.
5. **Wake-word gate** (only in wake mode, per connection): only segments
   addressed to the AI trigger a turn.
6. `ConversationManager` appends the history and calls the LLM
   (`chat_stream` in streaming mode, otherwise `chat`).
7. `sanitizer` removes emojis/Markdown/links; TTS synthesizes sentence by sentence.
8. Audio goes to the browser as turn-id-tagged frames.

---

## Streaming & Latency

By default (`STREAMING=1`) the latency-critical path is streamed end-to-end:

| Stage | What |
|------|-----|
| **A1** | LLM tokens are streamed (SSE), text appears live (`reply.delta`). |
| **A2** | As soon as a sentence is finished, it is synthesized immediately and played back progressively as PCM chunks (`VCT2`) — sentence 1 plays while sentence 2 is being generated. |
| **B1** | The browser streams microphone frames along live (`segment.stream.*`) instead of sending a blob at the end. |
| **B2** | The server transcribes the growing buffer in a throttled way → live interim transcripts (`transcript.partial`). |

`STREAMING=0` falls back to the classic path (generate completely first,
then one WAV) — useful if an LLM endpoint can't do SSE streaming.
Tuning knobs: `TTS_CHUNK_MS`, `TTS_FIRST_CHUNK_CHARS` (force-flush the first
sentence to TTS earlier), and for B2 `STT_PARTIAL*` (on by default with
`whisper_local`).

The debounce window is anchored at the moment the user stopped speaking: the
silence the client VAD already held before committing (redemption) and the
STT/gate latency count toward `DEBOUNCE_MS` instead of stacking on top of it.

The **statistics card** shows the perceived response time: `audio.start` delivers
`e2eMs` (finished speaking → first playback, incl. the configured
debounce pause — reported separately) as well as the "first / total" times for
the agent and TTS, making it visible that playback starts long before the
complete synthesis finishes.

---

## Wake Word

The wake word is an **input mode** alongside VAD and push-to-talk, selectable in
the **Input mode** box of the UI (per browser connection). In wake mode, only
segments whose transcript **begins** with the wake word (filler words like "Hey/Ok"
in front are allowed) trigger a turn — everything else is discarded. Fuzzy matching
tolerates Whisper mishearings ("Antonja", "Anthonia", "An Tonia"). After a reply
a conversation window stays open, so follow-up questions go through **without** saying
the wake word again. Typed inputs always bypass the gate.

`WAKE_WORD_ENABLED` is only the **startup default** (which mode the UI boots
up in); switching is possible live in the UI at any time. The other variables
configure the matching:

| Variable | Default | Meaning |
|---|---|---|
| `WAKE_WORD_ENABLED` | `0` | Startup default: `1` = UI starts in wake mode |
| `WAKE_WORD` | = `AGENT_NAME` | Wake word (empty = agent name lowercased) |
| `WAKE_MODE` | `conversation` | `conversation` = follow-up window stays open · `alexa` = one-shot, wake word needed every time |
| `WAKE_WORD_WINDOW_S` | `8` | Follow-up question window after a reply (s) |
| `WAKE_WORD_FUZZY` | `1` | Tolerate mishearings |
| `WAKE_WORD_ANYWHERE` | `0` | `1` = wake word anywhere instead of only at the start |
| `WAKE_WORD_STRIP` | `1` | Cut the wake word out of the text before the LLM |
| `WAKE_WORD_RATIO` | `0.78` | Fuzzy threshold (higher = stricter) |

---

## Voice lock

Speaker **verification** so the chat only listens to one enrolled voice. This is
separate from the wake word and from STT: every committed voice segment is turned
into a speaker embedding and compared (cosine similarity) against the enrolled
owner profile. A mismatch is dropped as `transcript.ignored` (`speaker_mismatch`)
**before** it reaches the wake gate or the LLM, and it does not interrupt a running
reply. It is language-independent, so Whisper keeps doing the German ASR.

**Mixed segments are trimmed, sentence-level:** when other voices talk into the
same segment right before/after the owner (a train announcement, kids, a video),
the gate scores **equal-length ~3 s blocks** on a fixed grid — embedding scores
are heavily length-sensitive, so only same-length blocks of the same segment are
comparable — and cuts sustained foreign passages (≥ 2.5 s) relative to the
segment's best block, then re-transcribes the kept audio (one extra STT call in
the mixed case). Single foreign words are deliberately never cut: they don't
falsify the transcript, and short-clip scores are unreliable. The removed text
stays visible in the transcript, red and struck through. Simultaneous
overlapping speech cannot be separated this way — that would need neural
target-speaker extraction. Kill switch: `SPEAKER_TRIM=0`.

**Barge-in follows the lock too:** while the lock is engaged the client does not
stop playback on VAD speech (any voice would trigger that). Playback is only
*ducked* while the server verifies the speaker on the live input stream
(~1.5 s of audio); a confirmed owner voice cancels the reply (`audio.stop`), a
foreign voice lets it keep playing at full volume. The stop button and
push-to-talk always interrupt (deliberate user actions).

**Setup** (optional, off by default):

1. `pip install sherpa-onnx` (bundles the kaldi-fbank the model expects; no torch).
2. Download a speaker-embedding model — recommended:
   `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` (~28 MB) from
   the [sherpa-onnx speaker models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models).
   (Benchmarked on real field audio it separates owner vs. foreign voices far
   better than `campplus_en_voxceleb_16k.onnx` — same size and CPU speed.)
3. In `.env`: `SPEAKER_LOCK_ENABLED=1` and `SPEAKER_MODEL_PATH=/abs/path/model.onnx`.
4. Start the server, open the UI → **Voice lock** card → **Learn my voice**
   (records ~6 s; 3–5 samples with your everyday mic and distance give a solid
   profile). The profile is written to `SPEAKER_PROFILE_PATH` and reused across
   restarts. The strictness slider in the card adjusts the threshold live.

Until a profile is enrolled the gate stays open (fail-open), and any load problem
(missing model/dep) simply disables it rather than blocking the mic. The profile
is **model-specific**: after switching `SPEAKER_MODEL_PATH` a profile enrolled
with another model is ignored (logged) and you re-enroll.

| Variable | Default | Meaning |
|---|---|---|
| `SPEAKER_LOCK_ENABLED` | `0` | Enable the voice gate |
| `SPEAKER_MODEL_PATH` | — | Absolute path to the CAM++/WeSpeaker ONNX model |
| `SPEAKER_PROFILE_PATH` | `./speaker_profile.json` | Where the enrolled profile is stored |
| `SPEAKER_THRESHOLD` | `0.5` | Cosine similarity to accept (higher = stricter; start default — the UI slider adjusts it live) |
| `SPEAKER_MIN_DUR_S` | `0.6` | Segments shorter than this can't be verified → dropped |
| `SPEAKER_TRIM` | `1` | Cut sustained foreign passages out of mixed segments |
| `SPEAKER_DEBUG` | `0` | Log per-block scores per segment (threshold tuning) |
| `SPEAKER_DUMP_DIR` | — | Dump gated segments + enroll takes as WAVs (offline gate debugging; newest ~200 kept) |
| `SPEAKER_PROVIDER` | `cpu` | onnxruntime provider: `cpu` \| `cuda` |

---

## Switching Backends

Three independent switches in the `.env`:

| Variable      | Werte                          | Default         |
|---------------|--------------------------------|-----------------|
| `STT_BACKEND` | `openai` · `whisper_local`     | `openai`        |
| `TTS_BACKEND` | `openai` · `omnivoice_local`   | `openai`        |
| `LLM_BACKEND` | `openai_compat` · `openclaw`   | `openai_compat` |

`cfg.validate()` checks only the *active* backend at startup (keys/required
fields). If a local dependency is missing, `load()` returns a clear error message
instead of an import error.

### Local / GPU

```bash
pip install faster-whisper          # STT_BACKEND=whisper_local
pip install omnivoice torch         # TTS_BACKEND=omnivoice_local (s. k2-fsa/OmniVoice)
```

`.env` for local Whisper (GPU):

```bash
STT_BACKEND=whisper_local
WHISPER_DEVICE=cuda
WHISPER_MODEL=large-v3-turbo
WHISPER_LOCAL_FILES_ONLY=1
```

`faster-whisper` also runs on the **CPU** (`WHISPER_DEVICE=cpu`, small model
like `base`, `WHISPER_LOCAL_FILES_ONLY=0` to fetch on demand) — good for testing without a GPU.

---

## Tests

```bash
.venv/bin/python -m pytest -q                 # voll (alle Backends gemockt, keine API/GPU)
.venv/bin/python -m pytest tests/test_wake.py # ein Modul
```

Each module has its own tests; all backends are mocked — no real API calls,
no GPU. Lazy imports and all backend combinations are covered.

---

## Project Structure

```
plauder/
├── config.py              # .env loading, Config dataclass, validation
├── server.py              # WS handler + turn/streaming orchestration; runtime state
├── app.py                 # app boot/wiring: build_app, init_backends, main/run
├── images.py              # /upload handler + /uploads → data-URL resolution
├── audio.py               # PCM/WAV/numpy, frame formats (VCT1/VCT2), sentence splitter
├── turn_state.py          # debounce + coalescing, VAD parameters
├── sanitizer.py           # emoji/markdown stripping, ghost filter, NO_REPLY
├── wake.py                # wake-word matching (STT prefix, fuzzy)
├── speaker_verify.py      # voice lock: speaker embeddings, enrollment, block scoring
├── session.py             # ConversationManager (history per session key)
├── hermes_history.py      # optional: fetch prior history from a Hermes gateway
├── telegram_bridge.py     # optional Telegram mirroring (legacy, off by default)
└── backends/
    ├── stt/{base,openai_api,whisper_local}.py
    ├── tts/{base,openai_api,omnivoice_local}.py
    └── llm/{base,openai_compat,openclaw}.py
server.py                  # entrypoint shim → plauder.server.run()
static/index.html          # complete browser client (audio, WS, UI)
```

> **Secrets & paths:** API keys and machine-specific paths belong
> exclusively in the `.env` (which is in `.gitignore`), never in the source code.
> An existing minimal `.env` (only `OPENAI_API_KEY` + `FIREWORKS_API_KEY`)
> continues to work unchanged thanks to legacy fallback chains.

---

## License

Copyright (C) 2026 Robert Sachse / Bernd Helm

This program is free software: you can redistribute and/or modify it under
the terms of the **GNU General Public License v3** (or any later version), as
published by the Free Software Foundation. It is distributed in the hope that
it will be useful, but **without any warranty**. See [`LICENSE`](LICENSE) for the
full text.
