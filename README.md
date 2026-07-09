# 🎙️ Plauder

Speech-to-speech chat in the browser: **microphone → STT → LLM → TTS → speaker** —
with turn-taking (debounce/coalescing), barge-in, low latency through end-to-end
streaming, a wake word, optional single-voice locking, text input and image
uploads.

The server is a clean Python package (`plauder/`) with **pluggable backends** for
STT, TTS and LLM. Via a single `.env` you switch between cloud APIs (OpenAI,
Fireworks), local models (faster-whisper, OmniVoice) and gateway backends
(OpenClaw) — without touching any code.

---

## Features

- 🎙️ **Voice-to-voice** in the browser, no app to install — just serve
  `static/index.html`.
- ⚡ **End-to-end streaming for low latency** — LLM tokens are synthesized
  sentence by sentence and played back progressively while the next sentence is
  still generating; the mic streams live and delivers interim transcripts (see
  [Streaming & latency](#streaming--latency)).
- 🔀 **Pluggable backends** — STT / TTS / LLM each selectable independently via
  `.env`, cloud or local/GPU, any combination.
- 🔔 **Wake word** — a selectable input mode (alongside VAD and push-to-talk):
  react only to "Antonia …", with a conversation window for follow-ups. Off by
  default, toggleable live in the UI.
- 🔒 **Voice lock** — optional speaker verification: enroll your voice from the
  UI and only *your* voice is transcribed; other people and background voices
  (TV, kids) are dropped before the LLM (see [Voice lock](#voice-lock)).
- 🗣️ **Voice library / cloning** — record a short sample or upload an audio file
  to clone a speaking voice, name it, and pick which voice the assistant speaks
  in. Persistent and shared across all devices. Needs the OmniVoice wrapper
  behind TTS; off by default (`TTS_CLONE_ENABLED`) (see [Voices](#voices-cloning)).
- 🗣️ **Turn-taking & barge-in** — debounce/coalescing; speaking again while the
  agent is answering discards its reply and starts a new turn.
- 💬 **Text + 🖼️ images** — type alongside the voice and attach images
  (multimodal) via 📎 / drag-and-drop / paste.
- 🎵 **Opus compression** — the mic uplink and TTS downlink transparently switch
  to Opus when both sides support WebCodecs (mainly for mobile links); raw PCM
  otherwise. Kill switch `AUDIO_OPUS`.
- 🔁 **Replay & download** — every reply's audio can be replayed or downloaded as
  WAV from a per-message menu.
- 🌐 **Cross-device session sync** — all open browsers share one conversation;
  inputs and replies mirror across devices, and "New session" resets them all.
- 🗣️ **Pronunciation lexicon** — a JSON file of spoken-form overrides for
  names/brands/acronyms the voice mangles.
- 🌍 **i18n (EN/DE)** — one `APP_LANGUAGE` switch drives the UI locale, the
  assistant's spoken language and the STT default.
- 🔌 **Reverse-proxy ready** — run under a sub-path via `BASE_PATH`.
- 🧠 **Optional Hermes memory** — bind to a Hermes gateway for server-side
  memory/history; plain OpenAI-compatible backends ignore it.

---

## Documentation

| Doc | What's in it |
|---|---|
| **[Getting started](#getting-started)** (below) | Quick start with the cloud default. |
| **[INSTALL.md](INSTALL.md)** | Full setup — local/GPU backends, voice lock, systemd, reverse proxy, troubleshooting. |
| **[docs/configuration.md](docs/configuration.md)** | **Complete parameter reference** — every `.env` variable, grouped, with defaults and fallback chains. |
| **[docs/user-guide.md](docs/user-guide.md)** | The browser client — every control, input mode, the per-message menu, voice lock, stats. |
| **[.env.example](.env.example)** | The annotated config template you copy to `.env`. |
| **[omnivoice-openai-wrapper/README.md](omnivoice-openai-wrapper/README.md)** | Running OmniVoice as a standalone OpenAI-compatible TTS service. |
| **[CLAUDE.md](CLAUDE.md)** | Architecture deep-dive + the wire protocol (for contributors). |

---

## Getting Started

### Requirements

- **Python 3.11+** and a microphone-capable browser (Chrome/Edge/Firefox).
- For the cloud default: an **OpenAI API key** (STT + TTS) and an
  **OpenAI-compatible LLM endpoint** (Fireworks, OpenAI, or a local server). For
  fully local operation see [Local / GPU](#local--gpu).

> **HTTPS or localhost — no way around it.** Browsers only expose the microphone
> in a secure context. `http://localhost` works; a bare `http://<ip>:8319` from
> another machine does **not**. For remote access use HTTPS (see
> [reverse proxy](#sub-path--reverse-proxy-base_path)) or an SSH tunnel.

### Quick start (cloud default, no GPU)

```bash
cp .env.example .env          # 1) fill in keys (at least OPENAI_API_KEY + LLM_*)
./start.sh                    # 2) creates the venv, installs deps, starts the server
```

Then open **http://localhost:8319**, allow the microphone, and speak.

`start.sh` is idempotent (safe to run any number of times) and listens on
`${HOST}:${PORT}` — the `.env.example` template sets `0.0.0.0:8319`; without a
`HOST` entry the server binds `127.0.0.1` only.

You start in **VAD mode** (everything you say is sent). In the **Input mode** box
you can switch to **wake word** — then begin with **"Antonia, …"** and keep
talking for ~8 s afterwards without repeating it. To boot straight into wake
mode, set `WAKE_WORD_ENABLED=1` in the `.env`.

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

Every option is documented in **[docs/configuration.md](docs/configuration.md)**
and inline in [`.env.example`](.env.example).

### Health check

```bash
curl http://localhost:8319/healthz   # 200 + the active backends
```

### Language (`APP_LANGUAGE`)

`APP_LANGUAGE` (`en`/`de`, default `en`) sets **both** the UI language of the
browser client (fully internationalized, EN + DE) **and** the assistant's default
spoken language plus the STT default language. The client renders in the selected
language automatically — no extra setup. `APP_LANGUAGE=de` gives a German UI and a
German-speaking assistant.

### Sub-path / reverse proxy (`BASE_PATH`)

To run under a sub-path instead of the domain root (e.g. behind a shared reverse
proxy), set `BASE_PATH=/voice`. All routes then live under that prefix, and the
server injects it so the client builds its WebSocket, upload and asset URLs
accordingly. Configure the proxy to forward the sub-path **without stripping** it.
Full nginx example in [INSTALL §7](INSTALL.md#7-reverse-proxy--sub-path).

---

## Architecture

```
                         Browser (static/index.html)
                          │   ▲
   16 kHz f32 PCM frames   │   │  PCM chunks (VCT2) / Opus chunks (VCT3) / WAV (VCT1)
   (VAD live / push-to-talk;│  │  + JSON events
    Opus packets when negotiated)
                          ▼   │
   ┌───────────────────────────────────────────────────────────────┐
   │                  plauder/server.py  (HTTP/WS layer)             │
   │   Routes: / , /healthz , /ws , /upload , /uploads/… , /static/… │
   │   ws_handler ─ audio segments/frames & text                    │
   │       ├─► turn_state.TurnState   (debounce + coalescing)        │
   │       ├─► speaker_verify         (voice-lock gate)              │
   │       ├─► wake                   (wake-word gate)               │
   │       ├─► sanitizer              (ghost filter, merge, NO_REPLY)│
   │       ├─► audio                  (PCM/WAV/numpy, sentence split) │
   │       └─► session.ConversationManager (history per session key) │
   └───────────────────────────────────────────────────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
      ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
      │  STTBackend  │     │  LLMBackend  │     │  TTSBackend  │
      │ .transcribe  │     │ .chat        │     │ .synth       │
      │              │     │ .chat_stream │     │ .synth_stream│
      ├──────────────┤     ├──────────────┤     ├──────────────┤
      │ openai       │     │ openai_compat│     │ openai       │   ← cloud
      │ whisper_local│     │ openclaw     │     │ omnivoice_   │   ← local/GPU
      │ (faster-     │     │ (gateway)    │     │  local       │     (lazy)
      │  whisper)    │     │              │     │ (omnivoice)  │
      └──────────────┘     └──────────────┘     └──────────────┘
        STT_BACKEND          LLM_BACKEND          TTS_BACKEND
```

**Lazy imports:** heavy GPU deps (`faster_whisper`, `torch`, `omnivoice`) are
imported exclusively inside the active backend's `load()`. In the cloud default no
GPU code is ever loaded.

### Pipeline per turn

1. The browser sends speech frames (VAD live / PTT) and/or text over the WebSocket.
2. `TurnState` collects inputs within the debounce window (`DEBOUNCE_MS`); new
   input aborts in-flight LLM/TTS calls (barge-in) — with the wake word or voice
   lock active only **after** the segment passes the gate, so foreign voices
   can't interrupt a running reply.
3. STT → text; Whisper hallucinations are filtered out.
4. **Voice-lock gate** (only when enrolled): segments not spoken by the owner are
   dropped; mixed segments are trimmed to the owner's part.
5. **Wake-word gate** (only in wake mode, per connection): only segments
   addressed to the AI trigger a turn.
6. `ConversationManager` appends to the history and calls the LLM (`chat_stream`
   in streaming mode, otherwise `chat`).
7. `sanitizer` strips emojis/markdown/links and applies the pronunciation lexicon;
   TTS synthesizes sentence by sentence.
8. Audio goes to the browser as turn-id-tagged frames.

A full architecture + wire-protocol walkthrough is in [CLAUDE.md](CLAUDE.md).

---

## Streaming & Latency

By default (`STREAMING=1`) the latency-critical path is streamed end to end:

| Stage | What |
|------|-----|
| **A1** | LLM tokens are streamed (SSE); text appears live (`reply.delta`). |
| **A2** | Each finished sentence is synthesized immediately and played back progressively as PCM/Opus chunks — sentence 1 plays while sentence 2 generates. |
| **B1** | The browser streams mic frames live (`segment.stream.*`) instead of sending one blob at the end. |
| **B2** | The server transcribes the growing buffer throttled → interim transcripts (`transcript.partial`). |

`STREAMING=0` falls back to the classic path (generate fully, then one WAV) —
useful when an LLM endpoint can't do SSE. Key tuning knobs: `TTS_CHUNK_MS`,
`TTS_FIRST_CHUNK_CHARS`, and for B2 the `STT_PARTIAL*` group (on by default only
with `whisper_local`, since each partial on a cloud STT is a paid call). The
debounce window is anchored at the moment you stopped speaking, so the client
VAD's redemption silence and the STT/gate latency count toward `DEBOUNCE_MS`
instead of stacking on top of it.

On a congested uplink (mobile, weak Wi-Fi) the client sheds mic frames once the
WebSocket send buffer backs up, so live streaming can't spiral into growing
latency; TTS is prefetched behind a lead gate so the next sentence is ready the
moment the current one finishes. Both are automatic — no configuration.

The **stats footer** shows the perceived response time: `e2eMs` (finished
speaking → first playback, incl. the debounce pause, reported separately) plus
first/total times for the agent and TTS — making it visible that playback starts
long before synthesis finishes. Full details:
[Configuration → Streaming](docs/configuration.md#streaming--latency).

---

## Wake Word

The wake word is an **input mode** alongside VAD and push-to-talk, selectable in
the UI (per browser connection). In wake mode, only segments whose transcript
**begins** with the wake word (fillers like "Hey/Ok" in front are allowed)
trigger a turn — everything else is discarded. Fuzzy matching tolerates Whisper
mishearings ("Antonja", "Anthonia", "An Tonia"). After a reply a conversation
window stays open so follow-ups go through **without** repeating the wake word.
Saying a stop word ("stop", "halt", …) as a whole utterance inside an open window
cancels the reply. Typed input always bypasses the gate.

`WAKE_WORD_ENABLED` is only the **startup default** (which mode the UI boots in);
switching is live. All matching parameters are in
[Configuration → Wake word](docs/configuration.md#wake-word).

---

## Voice lock

Speaker **verification** so the chat only listens to one enrolled voice. Separate
from the wake word and from STT: every committed voice segment is turned into a
speaker embedding and compared (cosine similarity) against the enrolled owner
profile. A mismatch is dropped as `transcript.ignored` (`speaker_mismatch`)
**before** it reaches the wake gate or the LLM, and it does not interrupt a
running reply. It is language-independent, so Whisper keeps doing the ASR.

**Mixed segments are trimmed, sentence-level:** when other voices talk into the
same segment right before/after the owner (a train announcement, kids, a video),
the gate scores **equal-length ~3 s blocks** on a fixed grid — embedding scores
are heavily length-sensitive, so only same-length blocks of the same segment are
comparable — and cuts sustained foreign passages (≥ 2.5 s) relative to the
segment's best block, then re-transcribes the kept audio (one extra STT call in
the mixed case). Single foreign words are deliberately never cut: they don't
falsify the transcript and short-clip scores are unreliable. The removed text
stays visible, red and struck through. Simultaneous overlapping speech can't be
separated this way (that needs neural target-speaker extraction). Kill switch:
`SPEAKER_TRIM=0`.

**Barge-in follows the lock too:** while the lock is engaged the client does not
stop playback on VAD speech (any voice would trigger that). Playback is only
*ducked* while the server verifies the speaker on the live input (~1.5 s); a
confirmed owner voice cancels the reply, a foreign voice lets it keep playing.
The stop button and push-to-talk always interrupt.

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
   restarts. The strictness slider adjusts the threshold live.

Until a profile is enrolled the gate stays open (fail-open), and any load problem
(missing model/dep) simply disables it rather than blocking the mic. The profile
is **model-specific**: after switching `SPEAKER_MODEL_PATH` an old profile is
ignored (logged) and you re-enroll. All parameters:
[Configuration → Speaker lock](docs/configuration.md#speaker-lock-voice-gate).

---

## Voices (cloning)

Give the assistant a **custom speaking voice** — cloned from a few seconds of
audio — and manage a whole library of them from the browser. Enable with
`TTS_CLONE_ENABLED=1`; a **"Voices"** card then appears in the settings panel.

- **Add a voice** — 🎙️ record a ~15 s sample, or ⬆️ upload an audio file (any
  format). The reference transcript is filled in automatically (Whisper); for
  uploads you can also type it if auto-detection fails. Samples are **auto-
  cleaned** before cloning (`TTS_CLONE_TRIM`, default on): half words cut off
  at the recording edges — which the model would reproduce as noise — are
  removed and edge silence is trimmed, so talking over the start/end of the
  recording window no longer ruins the clone.
- **Manage** — a compact dropdown picks the voice; icon buttons next to it
  🔊 preview, ✏️ rename and 🗑 delete the selected one. A built-in default
  voice always exists and can't be renamed or deleted.
- **Pick the active voice** — the dropdown choice is the active voice: the
  assistant speaks in it for **every** connected device, and it **persists
  across restarts and new sessions** (stored in `ACTIVE_VOICE_STATE_PATH`,
  default `./.active_voice`).
- **🔁 Echo mode** — a toggle in the Voices card to try a cloned voice live:
  while on, the assistant skips the LLM and simply **repeats whatever you say**
  in the active voice. Echo turns don't touch the conversation history and
  aren't mirrored to other devices or Telegram. Per browser tab, and off again
  after a reload — so a forgotten toggle can't look like a broken assistant.

**Requires the OmniVoice wrapper behind TTS** — voice cloning is an OmniVoice
capability, so `TTS_BACKEND=openai` must point at
[`omnivoice-openai-wrapper`](omnivoice-openai-wrapper/README.md) (plain OpenAI
TTS can't clone; the flag is ignored with a warning otherwise). The library
itself — reference WAVs + metadata — lives **on the wrapper/GPU box**
(`OMNIVOICE_VOICES_DIR`); the app mediates all record/upload/manage actions so
the browser never talks to the GPU box directly. Parameters:
[Configuration → Voice cloning](docs/configuration.md#voice-cloning--voice-library).

---

## Switching Backends

Three independent switches in the `.env`; any combination works. `cfg.validate()`
checks only the *active* backend at startup (keys/required fields), and a missing
local dependency yields a clear message from `load()` instead of an import error.

| Variable | Values | Default |
|---|---|---|
| `STT_BACKEND` | `openai` · `whisper_local` | `openai` |
| `TTS_BACKEND` | `openai` · `omnivoice_local` | `openai` |
| `LLM_BACKEND` | `openai_compat` · `openclaw` | `openai_compat` |

### Local / GPU

```bash
pip install faster-whisper          # STT_BACKEND=whisper_local
pip install omnivoice torch         # TTS_BACKEND=omnivoice_local (see k2-fsa/OmniVoice)
```

`.env` for local Whisper (GPU):

```bash
STT_BACKEND=whisper_local
WHISPER_DEVICE=cuda
WHISPER_MODEL=large-v3-turbo
WHISPER_LOCAL_FILES_ONLY=1
```

`faster-whisper` also runs on **CPU** (`WHISPER_DEVICE=cpu`, a small model like
`base`, `WHISPER_LOCAL_FILES_ONLY=0` to fetch on demand) — good for testing
without a GPU. For OmniVoice on a shared GPU box, prefer the standalone
[OpenAI-compatible wrapper](omnivoice-openai-wrapper/README.md) (see
[INSTALL §4](INSTALL.md#4-choose-your-backends)).

---

## Tests

```bash
.venv/bin/python -m pytest -q                 # full suite (all backends mocked; no API/GPU)
.venv/bin/python -m pytest tests/test_wake.py # one module
```

Each module has its own tests; all backends are mocked — no real API calls, no
GPU. Lazy imports and all backend combinations are covered. There is no lint step
and no pytest config file; tests rely on `tests/conftest.py`.

---

## Project Structure

```
plauder/
├── config.py              # .env loading, Config dataclass, validation
├── server.py              # WS handler + turn/streaming orchestration; runtime state
├── app.py                 # app boot/wiring: build_app, init_backends, main/run
├── images.py              # /upload handler + /uploads → data-URL resolution
├── audio.py               # PCM/WAV/numpy, frame formats (VCT1/VCT2/VCT3), sentence splitter
├── opus_codec.py          # optional opus encode/decode (lazy opuslib/libopus)
├── turn_state.py          # debounce + coalescing, VAD parameters
├── sanitizer.py           # emoji/markdown stripping, pronunciations, ghost filter, NO_REPLY
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

> **Secrets & paths** live exclusively in `.env` (gitignored), never in the
> source code. An existing minimal `.env` (only `OPENAI_API_KEY` +
> `FIREWORKS_API_KEY`) keeps working thanks to the legacy fallback chains.

---

## License

Copyright (C) 2026 Robert Sachse / Bernd Helm

This program is free software: you can redistribute and/or modify it under the
terms of the **GNU General Public License v3** (or any later version), as
published by the Free Software Foundation. It is distributed in the hope that it
will be useful, but **without any warranty**. See [`LICENSE`](LICENSE) for the
full text.
