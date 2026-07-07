# OmniVoice → OpenAI TTS wrapper

**This is only the wrapper, not OmniVoice itself.** It's a tiny HTTP server that
adapts [k2-fsa **OmniVoice**](https://github.com/k2-fsa/OmniVoice) to the OpenAI
`POST /v1/audio/speech` protocol. OmniVoice (the model + the `omnivoice` pip
package) is a **dependency** installed into the venv here — see `requirements.txt`
and the setup steps below.

It runs on a GPU box and exposes a Kokoro-style endpoint on port **8880** — so
**any** OpenAI-TTS client can use it with no code changes: the voice-chat app
(`TTS_BACKEND=openai`), and equally well the **Hermes / OpenClaw harness** or
anything else that already talks OpenAI TTS. Point the client's TTS base URL at
this wrapper and you're done.

Why a separate service instead of the in-process `omnivoice_local` backend? One
warm model on one pinned GPU, shared by every app on the network, decoupled from
the app's Python env and restarts. The app stays pure-cloud/CPU.

```
voice-chat app ──HTTP──►  :8880 /v1/audio/speech  ──►  OmniVoice (GPU, in-process)
Hermes / OpenClaw ──HTTP─►      (this wrapper)
```

## What's here

| File | Purpose |
|---|---|
| `omnivoice_openai_server.py` | FastAPI server: `/v1/audio/speech`, `/v1/models`, `/v1/audio/voices`, `/health` |
| `textnorm.py` | German number/date/currency → spoken words (OmniVoice mis-reads bare digits) |
| `bootstrap_voice.py` | Optional: generate voice-design candidates to freeze as a reference |
| `requirements.txt` | Pinned deps (the versions running in production) |
| `omnivoice.service` | systemd unit template |
| `ref/ref.txt.example` | Sample reference transcript |

The real voice reference (`ref/ref.wav` + `ref/ref.txt`) is **not** in git — it's a
personal recording. You provide your own (see step 3).

## Setup (GPU box)

Tested with **Python 3.11 + CUDA**, `omnivoice==0.1.5`, `torch==2.12.1`.

### 1. Dedicated venv

Use a **separate** venv, not the voice-chat `.venv` — this pulls in torch + the
model stack.

```bash
cd /opt/omnivoice-openai-wrapper                 # <- wherever you copy this folder
python3.11 -m venv ../omnivoice-env
# torch/torchaudio must match your CUDA — install them FIRST from the right index:
../omnivoice-env/bin/pip install torch==2.12.1 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu124
../omnivoice-env/bin/pip install -r requirements.txt
```

`pydub` shells out to **ffmpeg** for mp3/opus/aac output — make sure `ffmpeg` is
on PATH (`apt install ffmpeg`). WAV/FLAC/PCM need no ffmpeg.

### 2. Grab the model (first run pulls it from Hugging Face)

`k2-fsa/OmniVoice` downloads on first load into `HF_HOME`. On an air-gapped box,
pre-download it and set `HF_HOME` accordingly.

### 3. Pick a voice — the clone reference

The server freezes **one** voice at startup from a reference WAV + its exact
transcript. Two ways to get one:

**A) Bring your own** (best quality): a clean ~10–20 s mono recording of the
target voice.

```bash
cp /path/to/my_voice.wav ref/ref.wav
# put the EXACT words spoken in that clip into ref/ref.txt:
$EDITOR ref/ref.txt
```

**B) Synthesize a candidate** with OmniVoice voice-design:

```bash
../omnivoice-env/bin/python bootstrap_voice.py     # writes ref/cand_*.wav
# listen, pick one, then freeze it:
cp ref/cand_2.wav ref/ref.wav
cp ref/ref.txt.example ref/ref.txt                 # bootstrap speaks this exact text
```

> The transcript in `ref/ref.txt` **must** match what's spoken in `ref/ref.wav`
> word for word, or cloning quality drops.

### 4. Smoke-test in the foreground

```bash
CUDA_VISIBLE_DEVICES=0 ../omnivoice-env/bin/uvicorn omnivoice_openai_server:app \
    --host 0.0.0.0 --port 8880
# wait for ">> ready", then in another shell:
curl -s http://127.0.0.1:8880/health
curl -s http://127.0.0.1:8880/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"Guten Tag, hier spricht OmniVoice.","response_format":"wav"}' \
  -o /tmp/test.wav && echo "wrote /tmp/test.wav"
```

### 5. Run it as a service

```bash
sudo cp omnivoice.service /etc/systemd/system/omnivoice.service
sudo $EDITOR /etc/systemd/system/omnivoice.service   # set WorkingDirectory, GPU, ref paths
sudo systemctl daemon-reload
sudo systemctl enable --now omnivoice
systemctl status omnivoice
journalctl -u omnivoice -f                            # watch it load ("... >> ready")
```

## Configuration (env vars)

All optional; defaults are project-relative. In production they're set in the
systemd unit.

| Var | Default | Meaning |
|---|---|---|
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | Hugging Face model id |
| `OMNIVOICE_REF_WAV` | `./ref/ref.wav` | built-in (`default`) voice reference audio |
| `OMNIVOICE_REF_TXT` | `./ref/ref.txt` | exact transcript of that audio |
| `OMNIVOICE_VOICES_DIR` | `./voices` | persistent voice library (added voices: `{id}.wav` + `{id}.json`) |
| `OMNIVOICE_LANG` | `de` | default synthesis language |
| `OMNIVOICE_VOICE` | `de_female` | display name of the built-in `default` voice |
| `OMNIVOICE_NUM_STEP` | `32` | diffusion steps — lower = faster/rougher (16 is a good live default) |
| `OMNIVOICE_NORMALIZE` | `1` | expand German digits/dates/currency before synth |
| `OMNIVOICE_PROMPT_CACHE` | `8` | max cloned prompts kept warm in VRAM (LRU) |
| `CUDA_VISIBLE_DEVICES` | — | pin to one GPU |
| `HF_HOME` | — | model cache location |

## Connecting clients

### Voice-chat app

In the app's `.env` (this repo), just point the OpenAI TTS backend at the server:

```bash
TTS_BACKEND=openai
TTS_OPENAI_BASE_URL=http://<gpu-box>:8880/v1
TTS_OPENAI_API_KEY=                 # ignored by this server; any value is fine
TTS_OPENAI_MODEL=omnivoice
TTS_OPENAI_VOICE=de_female
TTS_OPENAI_SAMPLE_RATE=24000        # OmniVoice output rate
```

No `omnivoice`/`torch` in the app venv — the app stays a thin OpenAI client.

### Hermes / OpenClaw harness

Same story: it's a standard OpenAI TTS endpoint, so wherever the harness
configures a TTS/speech base URL and model, use:

```
base URL:  http://<gpu-box>:8880/v1
model:     omnivoice
voice:     de_female
```

and it speaks with the same German voice as the voice-chat app — one shared GPU
service behind all of them.

## Endpoints

- `POST /v1/audio/speech` — body `{input, voice?, response_format?, speed?, language?}`.
  `response_format` ∈ `mp3` (default) `wav` `flac` `pcm` `opus` `aac`.
  `voice` selects a library voice by **id** (`default` or an id from
  `GET /v1/audio/voices`); an unknown id falls back to `default` so audio never
  breaks. `language` is a non-standard convenience override (defaults to `OMNIVOICE_LANG`).
- `GET /v1/models` — advertises the single `omnivoice` model.
- `GET /v1/audio/voices` — the voice library: `{voices: [{id, name, created, isDefault}]}`.
- `POST /v1/audio/voices` — **register a new cloned voice.** `multipart/form-data`
  with `file` (any audio format — decoded via ffmpeg to a 24 kHz mono reference),
  `name`, and `ref_text` (exact transcript of the sample). Returns `{id, name, …}`.
- `PATCH /v1/audio/voices/{id}` — rename: body `{name}` (built-in `default` is immutable).
- `DELETE /v1/audio/voices/{id}` — remove a cloned voice (built-in `default` can't be deleted).
- `GET /health` — `{status: ok|loading, voice, voices}`.

### Voice library

Beyond the frozen built-in voice, the wrapper keeps a **persistent library** under
`OMNIVOICE_VOICES_DIR`: each added voice is a reference WAV + a small JSON of
`{id, name, ref_text, created}`, reloaded on restart. Clone prompts are built
lazily and cached in VRAM (LRU, `OMNIVOICE_PROMPT_CACHE`). The voice-chat app
drives all of this from the browser (record/upload/rename/delete/pick-active) —
see the app's `TTS_CLONE_ENABLED`.

Requests are serialized (single GPU, one `threading.Lock`); the model and the
built-in clone prompt are loaded once at startup with a warmup pass. Registering
a voice runs its decode + clone in a threadpool so concurrent speech isn't blocked.
