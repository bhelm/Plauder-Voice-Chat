# Installation Guide

Step-by-step setup of Plauder — from a bare Linux box to a running
speech-to-speech chat, including the optional features (voice lock, wake word,
local GPU backends, systemd, reverse proxy).

The short version for the impatient is in the [README](README.md#getting-started);
this guide covers the full path with all the options.

---

## 1. Prerequisites

- **Linux** (any distro with systemd if you want the service setup; macOS works
  for development too).
- **Python 3.11+** with `venv` (`sudo apt install python3 python3-venv` on
  Debian/Ubuntu).
- **git**.
- A **microphone-capable browser** (Chrome/Edge/Firefox).
- For the cloud default: an **OpenAI API key** (STT + TTS) and an
  **OpenAI-compatible LLM endpoint** (OpenAI itself, Fireworks, or any local
  server speaking the `/v1/chat/completions` protocol).

> **HTTPS or localhost — no way around it.** Browsers only expose the
> microphone in a *secure context*: `http://localhost` works, a bare
> `http://server-ip:8319` from another machine does **not**. For any remote
> access put the app behind HTTPS (see [reverse proxy](#7-reverse-proxy--sub-path))
> or tunnel it (`ssh -L 8319:localhost:8319 yourserver`).

No ffmpeg, no system audio libraries, no GPU are needed for the cloud default —
audio is handled as raw PCM in Python and in the browser.

## 2. Get the code and start it

```bash
git clone <your-repo-url> plauder
cd plauder
cp .env.example .env        # then edit — see next section
./start.sh
```

`start.sh` is **idempotent**: it creates `.venv/` on first run, installs
`requirements.txt`, and starts the server. Run it as often as you like; it never
destroys anything. The server listens on `${HOST}:${PORT}` — the `.env.example`
template sets `0.0.0.0:8319`; without a `HOST` entry it binds `127.0.0.1` only.

Sanity check:

```bash
curl http://127.0.0.1:8319/healthz     # 200 + JSON with the active backends
```

Then open **http://localhost:8319**, allow the microphone, and talk.

## 3. Configure `.env` (the only config file)

All configuration and all secrets live in `.env` (gitignored). Every option is
documented inline in [`.env.example`](.env.example). The minimal cloud setup:

```bash
# STT + TTS via OpenAI
OPENAI_API_KEY=sk-...

# LLM via any OpenAI-compatible endpoint
LLM_BACKEND=openai_compat
LLM_BASE_URL=https://api.fireworks.ai/inference/v1
LLM_API_KEY=fw_...
LLM_MODEL=accounts/fireworks/models/glm-5p2

# Language of UI + assistant + STT: en (default) or de
APP_LANGUAGE=de
```

Useful basics:

| Variable | Default | Meaning |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8319` | Bind address (`.env.example` sets `HOST=0.0.0.0`) |
| `AGENT_NAME` | `Antonia` | Assistant name; doubles as the default wake word |
| `APP_LANGUAGE` | `en` | UI + assistant + STT language (`en`/`de`) |
| `DEBOUNCE_MS` | `1200` | Pause after speech before the turn is submitted |
| `STREAMING` | `1` | End-to-end streaming; `0` = classic generate-then-play |

The server validates only the *active* backends at startup and prints a clear
error if a key or path is missing.

## 4. Choose your backends

Three independent switches; any combination works:

| Variable | Values | Default |
|---|---|---|
| `STT_BACKEND` | `openai` · `whisper_local` | `openai` |
| `TTS_BACKEND` | `openai` · `omnivoice_local` | `openai` |
| `LLM_BACKEND` | `openai_compat` · `openclaw` | `openai_compat` |

### Local STT (faster-whisper)

```bash
.venv/bin/pip install faster-whisper
```

```bash
STT_BACKEND=whisper_local
WHISPER_DEVICE=cpu          # or cuda on a GPU box
WHISPER_MODEL=base          # cpu: base/small; gpu: large-v3-turbo
WHISPER_LOCAL_FILES_ONLY=0  # 1 on air-gapped/GPU boxes with pre-downloaded models
```

### Local TTS (OmniVoice)

Two ways to run OmniVoice; pick one.

**A) In-process backend** — simplest, GPU deps live in the app venv:

```bash
.venv/bin/pip install omnivoice torch    # see k2-fsa/OmniVoice for details
```

```bash
TTS_BACKEND=omnivoice_local
```

Heavy GPU dependencies are imported **only** when their backend is active — the
cloud default never loads them.

**B) OpenAI wrapper as a standalone service (recommended for a shared GPU box)** —
run OmniVoice behind a small **OpenAI-compatible HTTP wrapper** on port 8880 and
let the app talk to it as a plain cloud TTS. One warm model on one pinned GPU,
decoupled from the app's env and restarts, and **reusable by other harnesses** —
the same endpoint drives Hermes / OpenClaw or anything else that speaks OpenAI TTS.

The wrapper (and **only** the wrapper — OmniVoice itself is a pip dependency it
installs) lives in [`omnivoice-openai-wrapper/`](omnivoice-openai-wrapper/): the
server, `textnorm.py`, pinned `requirements.txt`, systemd unit and a
voice-bootstrap helper. Full walkthrough in
[`omnivoice-openai-wrapper/README.md`](omnivoice-openai-wrapper/README.md).
In short, on the GPU box:

```bash
cp -r omnivoice-openai-wrapper /opt/omnivoice-openai-wrapper && cd /opt/omnivoice-openai-wrapper
python3.11 -m venv ../omnivoice-env
../omnivoice-env/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
../omnivoice-env/bin/pip install -r requirements.txt
# provide a voice: ref/ref.wav + matching ref/ref.txt (or run bootstrap_voice.py)
sudo cp omnivoice.service /etc/systemd/system/ && sudo $EDITOR /etc/systemd/system/omnivoice.service
sudo systemctl daemon-reload && sudo systemctl enable --now omnivoice
```

Then point the app (the app venv needs **no** GPU deps) at it:

```bash
TTS_BACKEND=openai
TTS_OPENAI_BASE_URL=http://<gpu-box>:8880/v1
TTS_OPENAI_MODEL=omnivoice
TTS_OPENAI_VOICE=de_female
TTS_OPENAI_SAMPLE_RATE=24000
```

### Pointing the LLM at your own server

Anything that speaks the OpenAI chat-completions protocol works (vLLM, llama.cpp
server, LM Studio, a gateway):

```bash
LLM_BACKEND=openai_compat
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_API_KEY=whatever-your-server-expects
LLM_MODEL=your-model-name
```

## 5. Optional: voice lock (speaker verification)

Lets the chat listen to **one enrolled voice only** — background voices, kids,
and TV are dropped before the LLM. Full behavior description in the
[README](README.md#voice-lock).

```bash
# 1) dependency (no torch needed)
.venv/bin/pip install sherpa-onnx

# 2) embedding model (~28 MB) — this one benchmarked best on real field audio
mkdir -p models
curl -L -o models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
```

(Yes, `speaker-recongition-models` — the typo is part of the upstream URL.)

```bash
# 3) .env
SPEAKER_LOCK_ENABLED=1
SPEAKER_MODEL_PATH=/abs/path/to/models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
```

4) Restart, open the UI → **Voice lock** card → **Learn my voice**. Record 3–5
samples with the microphone and distance you actually use. The profile lands in
`speaker_profile.json` (gitignored) and survives restarts.

The profile is **model-specific**: if you ever change `SPEAKER_MODEL_PATH`, the
old profile is ignored (logged at startup) and you enroll again. For threshold
tuning set `SPEAKER_DEBUG=1` (per-block scores in the log) and optionally
`SPEAKER_DUMP_DIR=/path/to/dumps` to keep the gated audio as WAV files for
offline analysis.

## 6. Optional: wake word

Off by default; switchable live in the UI (input mode: VAD / push-to-talk /
wake word). To boot straight into wake mode:

```bash
WAKE_WORD_ENABLED=1
# WAKE_WORD=antonia        # default: AGENT_NAME lowercased
```

## 7. Reverse proxy / sub-path

To serve the app under HTTPS and/or a sub-path, set

```bash
BASE_PATH=/voice
```

and forward the prefix **without stripping it** (WebSockets need the upgrade
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

Health check then lives at `https://your.domain/voice/healthz`.

## 8. Run as a systemd service

`/etc/systemd/system/voice-chat.service`:

```ini
[Unit]
Description=Plauder Voice Chat
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
WorkingDirectory=/opt/plauder          # wherever you cloned it
ExecStart=/opt/plauder/start.sh
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voice-chat.service
journalctl -u voice-chat.service -f        # logs
```

The server does **not** auto-reload; after a `git pull` or `.env` change:
`sudo systemctl restart voice-chat.service`.

## 9. Update

```bash
git pull
sudo systemctl restart voice-chat.service   # or re-run ./start.sh
```

`start.sh` re-installs requirements on every start, so dependency bumps are
picked up automatically. Client changes in `static/index.html` only need a
browser reload — the page is read fresh per request.

## 10. Tests

```bash
.venv/bin/python -m pytest -q     # all backends mocked — no API keys, no GPU
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Browser shows no mic prompt | Not a secure context — use `localhost`, HTTPS, or an SSH tunnel (see §1). |
| `healthz` fails / startup error names a variable | The active backend is missing a key/path in `.env` — the message says which. |
| Replies never stream (long silence, then all at once) | Your LLM endpoint can't do SSE — set `STREAMING=0`. |
| Voice lock rejects you | Re-enroll with your everyday mic/distance; lower the strictness slider; `SPEAKER_DEBUG=1` shows the scores. |
| Voice lock silently inactive | Model path wrong or `sherpa-onnx` missing — the startup log names the problem; the gate fails open by design. |
| Port 8319 busy | Another instance is running: `pgrep -f 'server\.py'`, kill that PID (not via `pkill -f start.sh` — it matches your own shell). |
