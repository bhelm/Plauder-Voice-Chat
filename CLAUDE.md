# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Browser speech-to-speech chat: mic → STT → LLM → TTS → speaker, with turn-taking
(debounce/coalescing), barge-in, low-latency streaming, and a wake word. The
server is a `plauder/` package with **pluggable STT/TTS/LLM backends** chosen
entirely via `.env` (no code changes). Comments, docs, and `.env.example` are in
English — match that when editing.

**Language / i18n.** `APP_LANGUAGE` (`en`/`de`, default `en`) is the single
language switch. It drives the assistant's default spoken language (the
`_DEFAULT_VOICE_HINTS` entry in `config.py`) and the STT default language, and it
is the UI locale handed to the browser. The client (`static/index.html`) is i18n
with an in-file `I18N = { en, de }` dictionary plus a tiny `t()` /
`applyI18n()` layer; UI text lives under `data-i18n*` attributes or `t('key')`
calls — never hard-code user-facing strings. The server injects the language at
serve time by replacing the `__APP_LANG__` placeholder in `index.html` (so the
page renders in the right locale with no flash) and also advertises it as
`hello.lang`. When adding UI strings, add a key to **both** `I18N.en` and
`I18N.de`.

**Sub-path / `BASE_PATH`.** `BASE_PATH` (default `''` = root, e.g. `/voice`) lets
the app run behind a reverse proxy under a sub-path. `build_app()` registers every
route under the prefix; `index()` injects it into `index.html` via the
`__BASE_PATH__` placeholder (same mechanism as `__APP_LANG__`), and `hello`
carries `basePath`. The client reads it into a `BASE` const and builds **all**
WS / `/upload` / `/uploads` / `/static` URLs with it (head `<script src>` and the
ort `wasmPaths` use the injected `__BASE_PATH__` directly, since they run before
`BASE` exists). Image URLs sent back to the server stay canonical (`/uploads/…`,
prefixed only for display via `mediaUrl()`). Any new client URL or absolute asset
path must go through `BASE`/`__BASE_PATH__`, or it breaks under a sub-path.

## Commands

```bash
./start.sh                       # idempotent: venv + deps + run server (port 8319)
.venv/bin/python -m pytest -q            # full suite (all backends mocked; no API/GPU)
.venv/bin/python -m pytest tests/test_wake.py -q          # one module
.venv/bin/python -m pytest tests/test_server.py::test_ws_hello_frame   # one test
curl http://127.0.0.1:8319/healthz       # 200 + active backends
```

There is **no lint step** and no pytest config file; tests rely on `tests/conftest.py`.

The browser client is a single hand-written file, `static/index.html` (~2200 lines
of inline JS — no build step). After editing its JS, syntax-check by extracting the
`<script>` blocks and running `node --check`; it cannot be unit-tested here.

Restarting the running server: **do not** `pkill -f "start.sh"` or
`pkill -f server.py` — the pattern matches the shell running the command and kills
it (exit 144). Kill by explicit PID (`pgrep -f 'server\.py'` → `kill -9 <pid>` plus
its `start.sh` parent), confirm port 8319 is free, then
`nohup ./start.sh > /tmp/voice.log 2>&1 &`. The server does not auto-reload.

## Architecture

`server.py` (repo root) is a shim → `plauder.server.run()`. Inside `plauder/`:

- **`config.py`** — single source of truth. `Config.from_env()` builds a frozen
  dataclass from `os.environ` (filled by a built-in `.env` parser). Has extensive
  **legacy fallback chains** (e.g. `STT_OPENAI_API_KEY` → `OPENAI_API_KEY`,
  `LLM_API_KEY` → `FIREWORKS_API_KEY`). `validate()` checks only the *active* backend.
- **`server.py`** — aiohttp HTTP/WS transport + turn orchestration only. Runtime
  state (CFG/STT/TTS/CONV/…) lives in module globals set via `configure()`; tests
  inject mock backends through it. Connection-owned tasks (debounce, agent,
  tracked segment/partial handlers) are cancelled on close via
  `_cancel_connection_tasks`.
- **`app.py`** — app boot/wiring: `build_app()` (routes + middleware),
  `init_backends()`, `main()`/`run()`. Reads `server.*` at call time (re-exported
  from `server` so the entrypoint shims and `srv.build_app` still resolve); kept
  out of `server.py` to separate process lifecycle from request handling.
- **`images.py`** — self-contained `/upload` handler + `/uploads/...` → data-URL
  resolution for the multimodal LLM call (no runtime backend state).
- **`backends/{stt,tts,llm}/`** — each has a `base.py` abstract class with a
  `from_config()` factory dispatching on `STT_BACKEND`/`TTS_BACKEND`/`LLM_BACKEND`.
  **Heavy GPU deps (`faster_whisper`, `torch`, `omnivoice`) are imported only inside
  `load()` of their backend** — never at module level. The cloud default never
  loads GPU code. Keep this invariant when touching backends.
- **`session.py`** (`ConversationManager`) holds conversation history per
  `user_key`; LLM backends are stateless (receive the full message list).
- **`turn_state.py`** (`TurnState`) — per-connection debounce/coalescing state.
- **`sanitizer.py`** — TTS text cleanup, NO_REPLY detection, Whisper hallucination
  filter, transcript merging. `audio.py` — PCM/WAV/numpy + sentence chunking.

### Turn lifecycle

Browser (16 kHz f32 PCM via Silero VAD or push-to-talk) → WS → `TurnState`
(debounce `DEBOUNCE_MS`, new input cancels in-flight = barge-in) → STT → ghost
filter → wake-word gate → `ConversationManager.chat*` → sanitize → TTS → browser.

### Streaming pipeline (`STREAMING=1`, default on)

The latency-critical path is streamed end to end; `STREAMING=0` restores the
classic "generate fully → one WAV" path (the fallback, used when an LLM endpoint
can't do SSE).

- **A1** LLM token streaming: `LLMBackend.chat_stream()` (SSE) →
  `ConversationManager.chat_stream()` → `reply.delta` events.
- **A2** sentence-wise TTS + progressive playback: `_stream_reply_and_tts()` in
  server.py splits the token stream into sentences (`audio.split_stream_sentences`),
  pipelines each through `TTSBackend.synth_stream()`, and ships PCM as **`VCT2`**
  binary chunks; the client plays them gaplessly via Web Audio. NO_REPLY is detected
  across deltas before any audio is emitted.
- **B1** input streaming: VAD-mode client streams mic frames live
  (`segment.stream.start` → binary frames → `segment.stream.commit`); the server
  buffers and transcribes on commit.
- **B2** streaming STT: while a segment streams in, the server throttle-runs STT on
  the growing buffer and emits `transcript.partial` (default on for `whisper_local`).

### Wake word (`WAKE_WORD_ENABLED=1`, default on)

`plauder/wake.py` gates **voice** turns on an STT-prefix match (fuzzy, tolerates
Whisper mishearings; default word = `AGENT_NAME` lowercased). Non-matching segments
emit `transcript.ignored` and do **not** start or cancel a turn — the in-flight
cancel in `_handle_audio_segment` is deliberately deferred until a segment passes
the gate, so background speech can't interrupt mid-answer. After a reply a
conversation window (`WAKE_WORD_WINDOW_S`, refreshed on `playback.done`) lets
follow-ups through without repeating the word. Typed input always bypasses the gate.

### Client ↔ server protocol

JSON event frames over WS (`hello`, `transcript`/`transcript.partial`/
`transcript.ignored`, `turn.pending`/`turn.commit`, `reply`/`reply.delta`,
`audio.start`/`audio.end`,
`wake.armed`/`wake.detected`/`wake.window`/`wake.closed`, …) plus
two binary framings:
`VCT1` = full WAV (classic path), `VCT2` = streamed PCM chunk (`audio.py` defines
both; the client parses them in `static/index.html`). `hello` advertises
capabilities (`streaming`, `streamInput`, `sttPartial`, `wakeWord`) so the client
adapts. Adding a server event/flag means handling it in `static/index.html` too.

**Latency stats:** `audio.start` carries the time-to-first-audio breakdown —
`e2eMs` (last segment received → first audio sent; **includes** the debounce
wait, `debounceMs` is sent alongside so the client can split pause vs. system
time), `llmFirstMs` (to first LLM token), `ttsFirstMs` (to first PCM chunk).
The full-phase totals stay on `reply` (`llmMs`) and `audio.end`/`audio.meta`
(`ttsMs`). The anchor lives on `TurnState.speech_end_ts`. **Wake-word** is a
per-connection input mode: the client sends `settings.wakeWordEnabled` (echoed
in `settings.ack`) → `TurnState.wake_word_enabled`; `CFG.wake_word_enabled` is
only the start-default advertised in `hello.wakeWord.enabled`
(`hello.wakeWord.available` is always true). **Acoustic feedback:** `wake.detected`
fires once per segment the moment the wake word matches — including in a B2
*partial* transcript, before the user stops talking (`_emit_wake_detected`,
deduped via `TurnState.wake_detected_seg`) → client plays a rising cue.
`wake.window` carries `windowS` + a `reason` on every open/refresh of the
conversation window (`_open_wake_window`): `command` = a request came in, a reply
is coming → client does NOT run the idle timer (so the window can't lapse mid-
answer); `armed` = only the wake word was said, Antonia waits; `done` = reply
fully spoken (emitted on `playback.done`). The client only arms the falling-cue
timer for `armed`/`done`, so the window closes a full `windowS` AFTER the answer
finishes, not when it starts. Saying a stop word (`wake.is_stop_command`:
"stop"/"ok stopp"/"halt"/… — must be the whole utterance) inside an open window
cancels any in-flight reply, closes the window (`wake_until=0`) and emits
`wake.closed`. Both cues are client-synthesized (Web Audio, no assets), scaled by
the persisted `cueVolume` slider (0 = off), and only sound in wake mode.

## Conventions & gotchas

- **`.env` is the only place for config and secrets** (gitignored). No credentials
  or machine-specific absolute paths in code, `.env.example`, or README — defaults
  use `Path.home()`/project-relative paths. New options: add field + `from_env()`
  wiring in `config.py` and document in `.env.example`.
- Backend test doubles in tests are **duck-typed** (don't subclass the bases), so
  orchestration code falls back via `getattr(..., "chat_stream"/"synth_stream", None)`
  — preserve that when adding new streaming hooks.
- `whisper_local` runs CPU (`WHISPER_DEVICE=cpu`, small model) or GPU
  (`cuda`, `large-v3-turbo`); on a GPU box also set `WHISPER_LOCAL_FILES_ONLY=1`.
- Speaker-ID (House Mode) is wired but the `speaker_id` module is **not in the repo**;
  it stays silently disabled unless that module + a CAM++ ONNX model + `speakers.json`
  are present.
- See `README.md` for the full backend-switching matrix and config reference.
