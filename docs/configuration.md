# Configuration Reference

Every runtime option Plauder understands, grouped by area. **All configuration
and all secrets live in a single `.env` file** (gitignored) next to the project
root; there is no other config file and no CLI flags. `.env.example` is the
annotated template — copy it to `.env` and edit.

- The `.env` is parsed by the server itself (no `python-dotenv` needed). A real
  environment variable that is already exported **wins** over the `.env` value,
  so shell exports and systemd `Environment=` lines override the file.
- Booleans accept `1/0`, `true/false`, `yes/no`, `on/off` (case-insensitive);
  empty means "use the default".
- `Config.from_env()` in [`plauder/config.py`](../plauder/config.py) is the single
  source of truth — this page mirrors it. `cfg.validate()` checks **only the
  active backend** at startup, so unused backends never need keys or deps.
- One meta variable is read before the config is built: `VOICE_ENV_FILE` sets an
  alternate path to the `.env` (default: `.env` next to the project root).

> **Legacy fallbacks.** Older `.env` files keep working: many new variables fall
> back to an older name if unset (shown as *falls back to …* below). A minimal
> `.env` with just `OPENAI_API_KEY` + `FIREWORKS_API_KEY` still runs the cloud
> default unchanged.

---

## Contents

- [General](#general)
- [Persona / system prompt](#persona--system-prompt)
- [Backend selection](#backend-selection)
- [STT — speech to text](#stt--speech-to-text)
- [TTS — text to speech](#tts--text-to-speech)
- [LLM](#llm)
- [Turn-taking](#turn-taking)
- [Streaming & latency](#streaming--latency)
- [Wake word](#wake-word)
- [Speaker lock (voice gate)](#speaker-lock-voice-gate)
- [Warmups](#warmups)
- [Pronunciation lexicon](#pronunciation-lexicon)
- [Hermes memory integration](#hermes-memory-integration)
- [House Mode](#house-mode)
- [Shared / legacy keys](#shared--legacy-keys)

---

## General

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address. `.env.example` ships `0.0.0.0`; without a `HOST` entry the server binds localhost only. |
| `PORT` | `8319` | Listen port. |
| `AGENT_NAME` | `Antonia` | Assistant name; also the **default wake word** (lowercased). |
| `APP_LANGUAGE` | `en` | UI locale **and** the assistant's + STT default language (`en`/`de`). Also accepts the alias `APP_LANG`. Drives the browser i18n, the voice-mode hint language, and `STT_LANGUAGE`'s default. |
| `BASE_PATH` | `` (root) | Sub-path to serve under, for a reverse proxy — e.g. `/voice` serves everything under `/voice/…`. `voice`, `/voice`, `/voice/` all normalize the same. The proxy must forward the prefix **without stripping** it. |
| `LOG_LEVEL` | `INFO` | Python log level (upper-cased). |

## Persona / system prompt

The assistant runs with a terse **voice-mode hint** by default (be concise, no
markdown/emoji, it is read aloud — language follows `APP_LANGUAGE`). An optional
persona is prepended to that hint. Resolution order: `SOUL_PATH` file →
`SYSTEM_PROMPT` → no persona.

| Variable | Default | Meaning |
|---|---|---|
| `SYSTEM_PROMPT` | `` | Persona text prepended to the voice hint. Most useful with `LLM_BACKEND=openai_compat`; leave empty if the backend already ships its own persona (e.g. an agent gateway). |
| `SOUL_PATH` | `` | Path to a persona file (`SOUL.md`); overrides `SYSTEM_PROMPT`. Unreadable file → warning + fall back to `SYSTEM_PROMPT`. |
| `VOICE_MODE_HINT` | *(built-in)* | Replace **only** the voice-hint part (persona still prepended). |
| `VOICE_MODE_SYSTEM` | `` | Full override of the **entire** system prompt (persona + hint ignored). |

## Backend selection

Three independent switches; any combination is allowed. `validate()` checks only
the active one.

| Variable | Values | Default |
|---|---|---|
| `STT_BACKEND` | `openai` · `whisper_local` | `openai` |
| `TTS_BACKEND` | `openai` · `omnivoice_local` | `openai` |
| `LLM_BACKEND` | `openai_compat` · `openclaw` | `openai_compat` |

Heavy GPU dependencies (`faster_whisper`, `torch`, `omnivoice`) are imported
**only** inside the active backend's `load()` — the cloud default never touches
GPU code.

## STT — speech to text

**OpenAI Whisper API** (`STT_BACKEND=openai`):

| Variable | Default | Meaning |
|---|---|---|
| `STT_OPENAI_API_KEY` | `` | *Falls back to `OPENAI_API_KEY`.* |
| `STT_OPENAI_MODEL` | `whisper-1` | Transcription model. |
| `STT_OPENAI_BASE_URL` | `` (OpenAI cloud) | Point at a self-hosted OpenAI-compatible Whisper endpoint. *Falls back to `OPENAI_BASE_URL`.* |
| `STT_LANGUAGE` | *(= `APP_LANGUAGE`)* | ISO-639-1 code; empty follows `APP_LANGUAGE`. Also accepts the alias `WHISPER_LANGUAGE`. |

**faster-whisper** (`STT_BACKEND=whisper_local`, needs `pip install faster-whisper`):

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | Model id. CPU: use `base`/`small`; GPU: `large-v3-turbo`. |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu`. |
| `WHISPER_COMPUTE_TYPE` | `int8` | e.g. `int8`, `float16`. |
| `WHISPER_BEAM_SIZE` | `5` | Beam search width. |
| `WHISPER_LOCAL_FILES_ONLY` | `1` | `1` = never hit the network (air-gapped/GPU box with pre-downloaded weights); `0` = fetch on demand. |

**Hallucination ("ghost") filter** — drops Whisper's phantom phrases ("Thank
you", "Thanks for watching") that appear on near-silence:

| Variable | Default | Meaning |
|---|---|---|
| `STT_HALLUCINATION_FILTER` | `1` | Master switch for the ghost filter. |
| `STT_GHOST_NO_SPEECH_PROB` | `0.6` | Drop a known ghost phrase when Whisper's no-speech probability exceeds this. |
| `STT_GHOST_USE_DURATION` | `0` | `1` = also treat very short segments as ghost candidates. |
| `STT_GHOST_MAX_DUR_S` | `1.5` | Segment-length cutoff for the duration heuristic above. |
| `STT_GHOST_EXTRA_PHRASES` | `` | Comma-separated phrases to add to the built-in ghost list. |

**Streaming STT / interim transcripts (B2)** — see [Streaming & latency](#streaming--latency).

## TTS — text to speech

**OpenAI TTS API** (`TTS_BACKEND=openai`). This backend also drives any
OpenAI-compatible TTS server — including the bundled
[OmniVoice wrapper](../omnivoice-openai-wrapper/README.md): keep
`TTS_BACKEND=openai` and point `TTS_OPENAI_BASE_URL` at it.

| Variable | Default | Meaning |
|---|---|---|
| `TTS_OPENAI_API_KEY` | `` | *Falls back to `OPENAI_API_KEY`.* |
| `TTS_OPENAI_MODEL` | `tts-1` | `tts-1` / `tts-1-hd`, or a self-hosted model name (e.g. `omnivoice`). Alias: `OPENAI_TTS_MODEL`. |
| `TTS_OPENAI_VOICE` | `nova` | `nova`/`shimmer`/`alloy`/`echo`/`fable`/`onyx`, or a self-hosted voice id. Alias: `OPENAI_TTS_VOICE`. |
| `TTS_OPENAI_BASE_URL` | `` (OpenAI cloud) | Self-hosted endpoint. *Falls back to `OPENAI_BASE_URL`.* |
| `TTS_OPENAI_SAMPLE_RATE` | `24000` | Sample rate of the returned raw PCM. `24000` = OpenAI cloud / Kokoro / XTTS. Only change it if the server demonstrably delivers something else (otherwise playback sounds too fast/slow). |
| `TTS_OPENAI_LOCAL_SPEED` | `0` | `1` = apply the speed slider client-side via pitch-preserving time-stretch instead of sending `speed` to the server. For servers that ignore `speed` (e.g. XTTS). |
| `TTS_SPEED` | `1.0` | Default playback speed (server clamps to `0.25`–`4.0`; the UI slider ranges `0.7`–`3.0×`). |

**OmniVoice in-process** (`TTS_BACKEND=omnivoice_local`, needs `pip install
omnivoice torch`). For a shared GPU box prefer the standalone wrapper instead
(see above / INSTALL §4).

| Variable | Default | Meaning |
|---|---|---|
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | Model id. |
| `OMNIVOICE_DEVICE` | `cuda` | `cuda`/`cpu`. |
| `OMNIVOICE_MODE` | `clone` | `clone` needs `OMNIVOICE_REF_AUDIO`. |
| `OMNIVOICE_REF_AUDIO` | `` | Reference voice WAV (**required** in `clone` mode). |
| `OMNIVOICE_REF_TEXT` | `` | Transcript of the reference audio (optional). |
| `OMNIVOICE_LANGUAGE` | *(= `APP_LANGUAGE`)* | Spoken language; empty follows `APP_LANGUAGE`. |

## LLM

**OpenAI-compatible** (`LLM_BACKEND=openai_compat`) — Fireworks, OpenAI, vLLM,
llama.cpp server, LM Studio, a gateway, anything speaking `/v1/chat/completions`:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEY` | `` | *Falls back to `FIREWORKS_API_KEY`, then `OPENCLAW_GATEWAY_TOKEN`.* |
| `LLM_BASE_URL` | `https://api.fireworks.ai/inference/v1` | Endpoint. Alias: `FIREWORKS_BASE_URL`. |
| `LLM_MODEL` | `accounts/fireworks/models/glm-5p2` | Model name. Alias: `FIREWORKS_MODEL`. |
| `LLM_MAX_TOKENS` | `4096` | Max completion tokens. Alias: `OPENCLAW_MAX_TOKENS`. |
| `LLM_TIMEOUT` | `300` | Request timeout (s). Alias: `OPENCLAW_TIMEOUT`. |
| `LLM_HISTORY_TURNS` | `20` | Past turns (user+assistant pairs) kept in the context window. |
| `OPENCLAW_RETRY_TIMEOUT` | `1` | `1` = retry once on an idle/timeout error (helps cold gateways). Applies to both LLM backends. |

**OpenClaw gateway** (`LLM_BACKEND=openclaw`, legacy):

| Variable | Default | Meaning |
|---|---|---|
| `OPENCLAW_GATEWAY_URL` | `http://localhost:8080` | Gateway base URL. |
| `OPENCLAW_GATEWAY_TOKEN` | `` | Auth token (**required** for this backend). *Falls back to `FIREWORKS_API_KEY`.* |
| `OPENCLAW_AGENT_ID` | `antonia` | Agent id. |
| `OPENCLAW_USER_ID` | `voice-user` | User id. |

## Turn-taking

The debounce window is anchored at the moment the user stops speaking; the
client VAD's redemption silence and STT/gate latency count toward it instead of
stacking on top. See [Streaming & latency](#streaming--latency) and the
[turn lifecycle](../CLAUDE.md).

| Variable | Default | Meaning |
|---|---|---|
| `DEBOUNCE_MS` | `1200` | Silence after speech before a turn is dispatched. New input inside the window coalesces; new input after dispatch is a barge-in. |
| `DEBOUNCE_MS_MIN` | `200` | Lower clamp for the UI pause slider. |
| `DEBOUNCE_MS_MAX` | `5000` | Upper clamp for the UI pause slider. |
| `TTS_SENTENCE_SPLIT` | `0` | `1` = synthesize sentence by sentence (lowers first-word latency in the non-streaming path). |
| `TTS_SENTENCE_GAP_MS` | `120` | Silence inserted between synthesized sentences. |
| `TTS_MAX_CHARS_PER_CHUNK` | `220` | Max chars per TTS chunk after the first sentence. |

## Streaming & latency

`STREAMING=1` (default) streams the whole latency-critical path end to end;
`STREAMING=0` restores the classic "generate fully → one WAV" path (use when an
LLM endpoint can't do SSE).

| Variable | Default | Meaning |
|---|---|---|
| `STREAMING` | `1` | End-to-end streaming (A1 LLM tokens + A2 sentence-wise TTS + B1 live mic upload). |
| `TTS_CHUNK_MS` | `400` | Target length of each progressive PCM/opus audio chunk. |
| `TTS_FIRST_CHUNK_CHARS` | `100` | Force-flush the **first** sentence to TTS after this many chars (cut at a clause boundary if possible) so a long opening sentence doesn't delay first audio. `0` = off. |
| `AUDIO_OPUS` | `1` | Opus compression for the browser audio links (kill switch). When on **and** the opus codec is usable (`opuslib` + system libopus) **and** the browser has WebCodecs, the mic uplink (~24 kbit/s vs. raw f32 ~512 kbit/s) and the TTS downlink (VCT3, ~48 kbit/s vs. raw PCM16 ~384 kbit/s) switch to opus automatically. No support on either side → raw paths, unchanged. `0` = force raw everywhere. |

**Interim transcripts (B2)** — while a segment streams in, periodically
re-transcribe the growing buffer and emit `transcript.partial`:

| Variable | Default | Meaning |
|---|---|---|
| `STT_PARTIAL` | *(on only for `whisper_local`)* | Enable interim transcripts. **Off by default with a cloud STT backend — each partial is a paid API call.** |
| `STT_PARTIAL_MIN_INTERVAL_MS` | `700` | Minimum gap between partial STT runs. |
| `STT_PARTIAL_MIN_NEW_MS` | `500` | Minimum new audio before another partial run. |

## Wake word

The wake word is a per-connection **input mode** (alongside VAD & push-to-talk),
switchable live in the UI. `WAKE_WORD_ENABLED` is only the **startup default** —
which mode the UI boots into.

| Variable | Default | Meaning |
|---|---|---|
| `WAKE_WORD_ENABLED` | `0` | `1` = UI starts in wake mode. |
| `WAKE_WORD` | *(= `AGENT_NAME` lowercased)* | The wake word. |
| `WAKE_MODE` | `conversation` | `conversation` = follow-up window stays open after a reply · `alexa` = one-shot, wake word needed every time. |
| `WAKE_WORD_WINDOW_S` | `8` | Follow-up window (s) after a reply. |
| `WAKE_WORD_FUZZY` | `1` | Tolerate Whisper mishearings ("Antonja", "Anthonia", "An Tonia"). |
| `WAKE_WORD_ANYWHERE` | `0` | `1` = match the wake word anywhere, not only at the start. |
| `WAKE_WORD_STRIP` | `1` | Cut the wake word out of the text before the LLM. |
| `WAKE_WORD_RATIO` | `0.78` | Fuzzy threshold (higher = stricter). |

Typed input always bypasses the gate. Saying a stop word ("stop", "halt", …) as
a whole utterance inside an open window cancels the reply and closes the window.

## Speaker lock (voice gate)

Optional speaker **verification** so the chat only listens to one enrolled
voice; background voices are dropped before the LLM. Independent of the wake word
and of STT (works with any STT backend). Needs `pip install sherpa-onnx` + a
CAM++/WeSpeaker ONNX embedding model. Fail-open: any missing model/dep/profile
disables the gate rather than blocking the mic. See the
[voice-lock guide](../README.md#voice-lock) for behavior details and enrollment.

| Variable | Default | Meaning |
|---|---|---|
| `SPEAKER_LOCK_ENABLED` | `0` | Enable the gate. |
| `SPEAKER_MODEL_PATH` | `` | Absolute path to the `.onnx` embedding model (**required** when enabled). |
| `SPEAKER_PROFILE_PATH` | `<project>/speaker_profile.json` | Where the enrolled profile is stored. Model-specific — switching models means re-enrolling. |
| `SPEAKER_THRESHOLD` | `0.5` | Cosine similarity to accept (higher = stricter; startup default — the UI slider adjusts it live). |
| `SPEAKER_MIN_DUR_S` | `0.6` | Segments shorter than this can't be verified → dropped. |
| `SPEAKER_TRIM` | `1` | Trim sustained foreign passages out of mixed segments (3 s block scoring + one extra STT call when mixed). |
| `SPEAKER_DEBUG` | `0` | Log per-block scores per segment (threshold tuning). |
| `SPEAKER_DUMP_DIR` | `` | Dump gated segments + enroll takes as WAVs here for offline analysis (newest ~200 kept). |
| `SPEAKER_PROVIDER` | `cpu` | onnxruntime provider: `cpu` / `cuda`. |
| `SPEAKER_NUM_THREADS` | `1` | onnxruntime thread count. |

## Warmups

Fire one throwaway call at startup so the first **real** user turn doesn't pay
the cold-start cost. Most useful for self-hosted endpoints that lazy-load a model
on first request (local Whisper, the OmniVoice wrapper).

| Variable | Default | Meaning |
|---|---|---|
| `STT_WARMUP` | `0` | One 0.5 s silent transcription at startup. **No-op with the OpenAI cloud STT backend** — it only fires for a self-hosted endpoint (`STT_OPENAI_BASE_URL` set / `whisper_local`). |
| `TTS_WARMUP` | `0` | One short synthesis at startup. |

## Pronunciation lexicon

| Variable | Default | Meaning |
|---|---|---|
| `VOICE_PRONUNCIATIONS` | `<project>/pronunciations.json` if present | JSON file of `{word: spoken-replacement}` applied to the reply text right before TTS (whole-word, case-insensitive) — fixes names/brands/acronyms the voice mangles. Keys starting with `_` are ignored (use them for comments). Reloaded automatically when the file's mtime changes. |

Example `pronunciations.json`:

```json
{
  "_comment": "spoken-form overrides applied before TTS",
  "GdW": "Ge de We",
  "Helm & Walter": "Helm und Walter"
}
```

## Hermes memory integration

Optional binding to a Hermes gateway that provides server-side memory/history. A
stable session id is sent as `X-Hermes-Session-Id`; plain OpenAI-compatible
backends ignore it. The "New session" reset in the UI rotates the persisted id.

| Variable | Default | Meaning |
|---|---|---|
| `HERMES_SESSION_KEY_SEPARATE` | `` | Stable session key for the voice chat's own Hermes session. Empty = no Hermes binding. |
| `HERMES_SESSION_STATE_PATH` | `<project>/.hermes_session_id` | Where the active session id is persisted after a reset, so reconnects/reloads stay in the fresh session. |

## House Mode

Umbrella switch for the multi-user "house" features. **Speaker
_identification_ (`HOUSE_SPEAKER_ID`) requires an external `speaker_id` module
that is not shipped in this repo** — it stays silently disabled until that module
+ a CAM++ model + `speakers.json` are present. For single-owner voice gating use
[Speaker lock](#speaker-lock-voice-gate) instead.

| Variable | Default | Meaning |
|---|---|---|
| `HOUSE_MODE` | `0` | Master switch; the three sub-flags default to its value. |
| `HOUSE_DATA_DIR` | `<project>/house_data` | Data directory. |
| `HOUSE_SPEAKER_ID` | *(= `HOUSE_MODE`)* | Multi-speaker identification (needs the external module). |
| `HOUSE_WAKE_WORD` | *(= `HOUSE_MODE`)* | Per-house wake-word handling. |
| `HOUSE_AUTH` | *(= `HOUSE_MODE`)* | House authentication. |

## Shared / legacy keys

Provided so existing minimal `.env` files keep working. Any backend key left
empty falls back to these.

| Variable | Used as fallback for |
|---|---|
| `OPENAI_API_KEY` | `STT_OPENAI_API_KEY`, `TTS_OPENAI_API_KEY` |
| `OPENAI_BASE_URL` | `STT_OPENAI_BASE_URL`, `TTS_OPENAI_BASE_URL` |
| `FIREWORKS_API_KEY` | `LLM_API_KEY`, `OPENCLAW_GATEWAY_TOKEN` |
| `FIREWORKS_BASE_URL` / `FIREWORKS_MODEL` | `LLM_BASE_URL` / `LLM_MODEL` |
