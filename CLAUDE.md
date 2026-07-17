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
is the UI locale handed to the browser. The client is i18n with an
`I18N = { en, de }` dictionary (`static/js/i18n.js`) plus a tiny `t()` /
`applyI18n()` layer in `index.html`; UI text lives under `data-i18n*` attributes
or `t('key')` calls — never hard-code user-facing strings. The server injects
the language at serve time by replacing the `__APP_LANG__` placeholder in
`index.html` (so the page renders in the right locale with no flash) and also
advertises it as `hello.lang`. When adding UI strings, add a key to **both**
`I18N.en` and `I18N.de` — `tests/client/pure_modules.test.mjs` fails on
mismatched key sets.

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
The split CSS/JS asset URLs additionally carry `?v=__ASSET_VER__`, replaced at
serve time with the newest client-file mtime (`_asset_version()` in server.py)
— cache busting without a build step.

## Commands

```bash
./start.sh                       # idempotent: venv + deps + run server (port 8319)
.venv/bin/python -m pytest -q            # full suite (all backends mocked; no API/GPU)
.venv/bin/python -m pytest tests/test_wake.py -q          # one module
.venv/bin/python -m pytest tests/test_server.py::test_ws_hello_frame   # one test
curl http://127.0.0.1:8319/healthz       # 200 + active backends
```

There is **no lint step** and no pytest config file; tests rely on `tests/conftest.py`.

The browser client is hand-written with **no build step**: `static/index.html`
(markup + chat/protocol/UI logic as inline JS) plus classic scripts under
`static/js/` — `i18n.js` (string table), `markdown.js` (renderer + escapeHtml),
`vct.js` (binary frame parsers), `playback.js` (audio OUTPUT: stream player,
WAV playback, replay cache, volume/duck) and `mic.js` (audio INPUT: mic/VAD
capture, opus uplink, timed enrollment/clone recorder, PTT) — and
`static/style.css`. The split files are plain classic scripts loaded before
the main inline block — top-level `const`/`let`/`function` bindings are shared
through the global lexical scope (call-time resolution), so no
import/export/window wiring; each cluster's state lives at the top of its
file and must not be redeclared elsewhere. After editing client JS run
`.venv/bin/python -m pytest tests/test_client_js.py -q`: it `node --check`s
every file and inline block and unit-tests the pure modules
(`tests/client/pure_modules.test.mjs`).

Restarting the running server: it is managed by systemd
(`voice-chat.service`, `Restart=always`) — use `systemctl restart voice-chat`;
logs via `journalctl -u voice-chat`. Do **not** `pkill -f "start.sh"` /
`pkill -f server.py` (the pattern matches the shell running the command and kills
it, exit 144), and do not start `./start.sh` by hand while the unit is active —
the port is taken and systemd would race a manual instance. The server does not
auto-reload.

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
- **`voice_clone.py`** — voice library / cloning handlers: `/voice-upload`,
  recorded-sample commit, hello capability block, active-voice lookup. Reads
  `server.*` at call time (same pattern as `app.py`); the wrapper CRUD client
  is `voices.py` (`VoiceLibrary`).
- **`speaker_gate.py`** — speaker-lock gating POLICY (commit gate incl.
  foreign-passage trim, early barge-in check, owner-watch auto-commit) plus
  all its field-calibrated tunables. Reads `server.*` at call time; the
  scoring itself (embeddings/thresholds) is `speaker_verify.py`.
- **`backends/{stt,tts,llm}/`** — each has a `base.py` abstract class with a
  `from_config()` factory dispatching on `STT_BACKEND`/`TTS_BACKEND`/`LLM_BACKEND`.
  **Heavy GPU deps (`faster_whisper`, `torch`, `omnivoice`) are imported only inside
  `load()` of their backend** — never at module level. The cloud default never
  loads GPU code. Keep this invariant when touching backends.
  The `hermes_gateway` LLM backend is special: a **persistent WebSocket** to the
  Hermes gateway's `voice_chat` platform adapter (plugin source lives in
  `hermes_plugin/` in this repo, symlink-installed into
  `~/.hermes/plugins/platforms/` — after changing `hermes_plugin/` code, run
  `hermes gateway restart`, not just the voice-chat service). It sends only
  the LAST user message incl. image data URLs (the gateway keeps the history
  and caches images as vision attachments) and receives **unsolicited pushes**
  (background task results, cron) which `server.handle_gateway_push` speaks
  through the streaming machinery in push mode (`_StreamingReply(push=True)` —
  like echo, but the bubble is a regular reply marked `push`). A push claims
  the `agent_task` slot under its own `push-<id>`, tracked on
  `TurnState.agent_turn_id` so a barge-in's `turn.discarded` names the CANCELLED
  push (not the incoming user turn — else the client suppresses the new turn's
  bubbles). **Nothing unspoken is dropped** (`_handle_cancelled_push`), and a
  barged-into push is never re-spoken verbatim. The cancel matrix:
  **barge-in** decides on how much actually played — the browser reports it via
  `playback.stopped` (`{turnId, playedS}`, sent by `bargeInStop` for VAD/manual
  stops) and the threshold is `PUSH_HEARD_THRESHOLD_S` (default 3 s): *heard*
  (played ≥ threshold) → persisted text bubble only; *unheard* (< threshold, or
  no report) → text bubble **and** `push.undelivered` to the gateway
  (`notify_push_undelivered` hook → per-chat note appended to the next turn's
  `channel_prompt`, so the agent weaves the content into its answer to the
  interrupting utterance). **stop-word** → text bubble; **connection close** →
  re-queue onto `_PENDING_PUSHES` for the next connect; **session reset** → drop
  (fresh session). The surfaced text bubble is an `external.message`
  `source:"push", persist:true` (persisted client-side, unlike the ephemeral
  `source:"system"` notices). Gateway token
  streaming arrives as `agent.partial` frames carrying the full accumulated
  text; the backend yields suffix deltas per message and ignores reformatted
  finalize rewrites (no double-speak). Wire protocol + setup:
  `hermes_plugin/README.md`; cross-side protocol tests:
  `tests/test_hermes_bridge.py`, push tests: `tests/test_server_push.py`.
- **`session.py`** (`ConversationManager`) holds conversation history per
  `user_key`; LLM backends are stateless (receive the full message list).
- **`turn_state.py`** (`TurnState`) — per-connection debounce/coalescing state.
- **`sanitizer.py`** — TTS text cleanup, NO_REPLY detection, Whisper hallucination
  filter, transcript merging. `audio.py` — PCM/WAV/numpy + sentence chunking.

### Turn lifecycle

Browser (16 kHz f32 PCM via Silero VAD or push-to-talk) → WS → `TurnState`
(debounce `DEBOUNCE_MS`, new input cancels in-flight = barge-in) → STT → ghost
filter → wake-word gate → `ConversationManager.chat*` → sanitize → TTS → browser.
The debounce timer is anchored at speech end (`TurnState.debounce_anchor`): the
client VAD's redemption silence (~0.8×debounce, credited for non-PTT segments)
and the STT/gate latency count toward the `DEBOUNCE_MS` pause window instead of
stacking on top of it (floor `DEBOUNCE_MIN_WAIT_S`).

### Streaming pipeline (`STREAMING=1`, default on)

The latency-critical path is streamed end to end; `STREAMING=0` restores the
classic "generate fully → one WAV" path (the fallback, used when an LLM endpoint
can't do SSE).

- **A1** LLM token streaming: `LLMBackend.chat_stream()` (SSE) →
  `ConversationManager.chat_stream()` → `reply.delta` events.
- **A2** sentence-wise TTS + progressive playback: `_stream_reply_and_tts()` in
  server.py splits the token stream into sentences (`audio.split_stream_sentences`),
  pipelines each through `TTSBackend.synth_stream()`, and ships PCM as **`VCT2`**
  binary chunks (or opus-encoded **`VCT3`** chunks when the client negotiated
  `audioCodec:"opus"` — see the protocol section); the client plays them gaplessly
  via Web Audio. NO_REPLY is detected across deltas before any audio is emitted.
- **B1** input streaming: VAD-mode client streams mic frames live
  (`segment.stream.start` → binary frames → `segment.stream.commit`); the server
  buffers and transcribes on commit.
- **B2** streaming STT: while a segment streams in, the server throttle-runs STT on
  the growing buffer and emits `transcript.partial` (default on for `whisper_local`).

### Wake word (`WAKE_WORD_ENABLED`, start-default off)

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
three binary framings. **Cross-device sync:** all browsers share ONE session
(`WS_CLIENTS` registry in server.py); committed user inputs and final replies
are mirrored to the other connections as `chat.remote`, and a `session.reset`
rotates the persisted Hermes session ID (`.hermes_session_id`, read fresh by
`_apply_hermes_headers` before every LLM call), cancels the peers' in-flight
turns and clears their UIs via `session.reset.remote`.
`VCT1` = full WAV (classic path), `VCT2` = streamed PCM chunk, `VCT3` = streamed
opus chunk (`audio.py` defines all three; the client parses them in
`static/index.html`). `hello` advertises capabilities (`streaming`,
`streamInput`, `sttPartial`, `wakeWord`, `audio`) so the client adapts. Adding a
server event/flag means handling it in `static/index.html` too.

**Opus compression (`AUDIO_OPUS`, default on).** `plauder/opus_codec.py` wraps
opuslib/system-libopus (imported ONLY lazily; when missing, `hello.audio`
advertises `{opusIn:false, opusOut:false}` and everything stays raw — same
invariant as the GPU backends). The client feature-detects WebCodecs
(`AudioEncoder`/`AudioDecoder`, codec `opus`) and uses opus automatically when
both sides support it (no UI toggle; the negotiated codecs are logged to the
console). **Uplink** (B1 streamed segments only): `segment.stream.start`
carries `codec:"opus"`, each binary message holds `0x4F ('O') + u16 BE len +
packet` records (client-encoded @16 kHz ~24 kbit/s; the preroll goes through
the same encoder, and the encoder is flushed before the commit JSON — all opus
start/commit/abort work is serialized on one promise chain so wire order can't
interleave across segments). The server decodes packets ON ARRIVAL into
`seg["buf"]` as 16 kHz f32, so partials/speaker gates/owner-watch/commit STT
keep seeing plain PCM; corrupt packets are logged + dropped. Enrollment and
PTT/full segments stay raw f32. **Downlink:** the client opts in via
`settings.audioCodec:"opus"` (echoed in `settings.ack`, stored as
`TurnState.audio_codec`, re-sent after `hello` once caps are known);
`_tts_worker` then encodes the TTS PCM (one encoder per turn at the TTS sample
rate, ~48 kbit/s, flushed at stream end) and ships `VCT3` frames =
`"VCT3" + idLen + turnId + u16 seq + repeated(u16 len + packet)`, one frame per
~`tts_chunk_ms`; `audio.start` carries `codec:"opus"|"pcm"`. The client decodes
via `AudioDecoder` into the existing streamPlayer path (gapless scheduling,
barge-in stop, replay cache, `playback.done`), gates decoded output on
`streamPlayer.turnId` at decode-output time, coalesces the decoded 20 ms
packets into ~300 ms buffers before scheduling (one WebAudio node per packet
would pile up thousands of scheduled nodes on long replies → audio-thread
crackle; tail flushed in `endStreamPlayback`) and flushes the decoder before
finalizing on `audio.end`. Defense in depth: an opus `segment.stream.start`
without a usable server codec is answered with `transcript.error` and the
segment is dropped.

**Echo mode** (voice-clone playground): `settings.echoMode` (echoed in
`settings.ack`) → `TurnState.echo_mode`, toggled from the Voices card
(client-side state, deliberately not persisted — a reload always starts in
normal mode). While on, `_run_turn_inner` short-circuits after `turn.commit`:
no LLM/CONV call, no history, no peer/Telegram mirroring — the committed user
text is fed through the streaming TTS machinery as-is (works with
`STREAMING=0` too, since no LLM stream is needed) and `reply.start`/`reply`
carry `echo:true` (the client styles the bubble with a 🔁 marker).

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

### Waifu / VTuber avatar (`WAIFU_MODE`, start-default off)

An **optional, fully isolated** 3D avatar (VRM 1.0 via three-vrm) rendered in the
settings drawer — a client-only feature that never touches the STT/TTS/turn
pipeline (if it breaks, the voice chat runs on unaffected; every entry point is
wrapped in try/catch). `WAIFU_MODE` (env, `config.py`) is only the server-side
start-default, injected into `index.html` via the `__WAIFU_MODE__` placeholder
(same mechanism as `__APP_LANG__`) and read into `window.__WAIFU_MODE__`; a
per-user `localStorage` toggle (`waifuEnabled`) overrides it. `start-joy.sh`
exports `WAIFU_MODE=1` by default.

- **`static/js/waifu.js`** — ES module (loaded via the `<script type="importmap">`
  in `index.html`: `three`, `three/addons/`, `three-vrm`, `three-vrm-animation`,
  all vendored under `static/vendor/`). Dynamically imported **only when the
  toggle turns on** — zero three.js overhead while off. Owns the render loop,
  scene, and a layered animation system (all additive per frame, damped): rest
  pose (arms down — the VRM default is a T-pose), breathing, procedural idle
  randomness (head glances, arm sway, body weight-shift), mode overlays
  (`idle`/`listening`/`speaking`/`thinking` — thinking is staged by duration:
  head tilt → squint → wandering eyes via VRM lookAt), and one-shot emotes.
  `Waifu.emote(name)` plays a matching **VRMA clip** (`static/anims/*.vrma`,
  from `tk256ailab/vrm-viewer`, MIT) blended over the procedural pose via an
  AnimationMixer + per-bone quaternion slerp (rotation tracks only — no root
  motion, no expression-track conflicts with lip-sync); facial expression runs
  in parallel. Public API: `mount/unmount/setMouth/setState/emote/isReady/resize`.
- **`static/js/waifu_ui.js`** — classic script (global scope like the others).
  Owns the toggle, the docked stage, the pop-out window (physically **moves** the
  canvas into a `window.open()` doc — renderer/state survive; double-click =
  fullscreen), lazy loading, and the non-invasive hooks: it wraps the global
  `setMicUi` for body-language state, and wraps `appendAgentBubbleDelta` to (a)
  set `thinking` on the first reply delta and (b) scan reply text for nonverbal
  tags (`[laughter]`/`[sigh]`/`*lacht*`/…) → queued emotes. **Lip-sync** is a
  standalone rAF driver that reads the real playback state (`anyAudioPlaying()`
  from playback.js, plus `replayPlaying`) — deliberately NOT `setMicUi` events,
  which fire on VAD/transcript mid-playback and used to kill the mouth. The
  emote queue only fires while audio actually plays (text streams ahead of
  speech) and is discarded after real silence.
- The `joy.vrm` model lives in `static/models/` (gitignored like the other
  models). `static/waifu_test.html` is a standalone dev harness for the avatar.
  Avatar UI strings use `waifu.*` / `settings.sec_waifu` i18n keys (both locales).

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
- Speaker-ID (House Mode): `speaker_id.py` (CAM++ embedder + multi-register
  fingerprint store) ships in the repo, but the feature stays silently disabled
  unless a CAM++ ONNX model + `speakers.json` are present (both gitignored;
  enroll via `enroll_register.py`).
- See `README.md` for the full backend-switching matrix and config reference.
