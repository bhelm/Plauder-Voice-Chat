#!/usr/bin/env python
"""OpenAI-compatible /v1/audio/speech server backed by OmniVoice (k2-fsa).

Drop-in replacement for a Kokoro/OpenAI TTS endpoint: any client that speaks
the OpenAI `POST /v1/audio/speech` protocol (the voice-chat app with
``TTS_BACKEND=openai``, but also Hermes / OpenClaw or any other harness) can
point at it unchanged.

Loads the OmniVoice model once, freezes a fixed voice-clone prompt at startup,
and serves TTS. GPU is pinned via CUDA_VISIBLE_DEVICES (see the systemd unit).

Config via environment (all optional; defaults are project-relative):
  OMNIVOICE_MODEL     HF model id                (default k2-fsa/OmniVoice)
  OMNIVOICE_REF_WAV   clean voice-clone reference WAV   (default ./ref/ref.wav)
  OMNIVOICE_REF_TXT   exact transcript of that WAV      (default ./ref/ref.txt)
  OMNIVOICE_LANG      default synthesis language        (default de)
  OMNIVOICE_VOICE     advertised voice name             (default de_female)
  OMNIVOICE_NUM_STEP  diffusion steps (quality/latency) (default 32)
  OMNIVOICE_NORMALIZE 1=expand German digits/dates/etc. (default 1)
"""
import io
import os
import threading

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from pydub import AudioSegment

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from textnorm import normalize_de

_HERE = os.path.dirname(os.path.abspath(__file__))
SR = 24000
NORMALIZE = os.environ.get("OMNIVOICE_NORMALIZE", "1") != "0"

# perf: allow TF32 matmuls on Ampere. NOTE: do NOT enable cudnn.benchmark —
# it re-autotunes per input shape, and TTS uses variable-length inputs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
REF_WAV = os.environ.get("OMNIVOICE_REF_WAV", os.path.join(_HERE, "ref", "ref.wav"))
REF_TXT = os.environ.get("OMNIVOICE_REF_TXT", os.path.join(_HERE, "ref", "ref.txt"))
DEFAULT_LANG = os.environ.get("OMNIVOICE_LANG", "de")
VOICE_NAME = os.environ.get("OMNIVOICE_VOICE", "de_female")
NUM_STEP = int(os.environ.get("OMNIVOICE_NUM_STEP", "32"))

_lock = threading.Lock()
_model = None
_clone_prompt = None
_gen_cfg = None


def _load():
    global _model, _clone_prompt, _gen_cfg
    print(f">> loading {MODEL_ID} ...", flush=True)
    _model = OmniVoice.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    _gen_cfg = OmniVoiceGenerationConfig(num_step=NUM_STEP, guidance_scale=2.0)
    with open(REF_TXT, "r", encoding="utf-8") as fh:
        ref_text = fh.read().strip()
    print(f">> freezing voice clone prompt from {REF_WAV}", flush=True)
    _clone_prompt = _model.create_voice_clone_prompt(ref_audio=REF_WAV, ref_text=ref_text)
    # warmup so the first real request isn't slowed by CUDA/cudnn autotune
    try:
        _synth("Guten Tag.", DEFAULT_LANG, None)
        print(">> warmup done", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f">> warmup skipped: {e}", flush=True)
    print(">> ready", flush=True)


def _synth(text: str, language: str, speed):
    with _lock, torch.inference_mode():
        out = _model.generate(
            text=text,
            language=language,
            voice_clone_prompt=_clone_prompt,
            speed=speed,
            generation_config=_gen_cfg,
        )
    return np.asarray(out[0], dtype=np.float32)


def _encode(wav: np.ndarray, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "mp3").lower()
    if fmt == "wav":
        buf = io.BytesIO(); sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"
    if fmt == "flac":
        buf = io.BytesIO(); sf.write(buf, wav, SR, format="FLAC")
        return buf.getvalue(), "audio/flac"
    if fmt == "pcm":  # OpenAI: 24kHz 16-bit mono signed LE, headerless
        return (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes(), "audio/pcm"
    # mp3 / opus / aac via ffmpeg (pydub)
    pcm16 = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
    seg = AudioSegment(pcm16, frame_rate=SR, sample_width=2, channels=1)
    codec = {"mp3": ("mp3", "audio/mpeg"),
             "opus": ("opus", "audio/ogg"),
             "aac": ("adts", "audio/aac")}.get(fmt)
    if codec is None:
        raise HTTPException(400, f"unsupported response_format: {fmt}")
    ffmt, mime = codec
    buf = io.BytesIO(); seg.export(buf, format=ffmt)
    return buf.getvalue(), mime


app = FastAPI(title="OmniVoice OpenAI TTS")


class SpeechReq(BaseModel):
    model: str | None = "omnivoice"
    input: str
    voice: str | None = VOICE_NAME
    response_format: str | None = "mp3"
    speed: float | None = 1.0
    language: str | None = None  # non-standard convenience override


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "voice": VOICE_NAME}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "omnivoice", "object": "model", "owned_by": "k2-fsa"}]}


@app.get("/v1/audio/voices")
def voices():
    return {"voices": [VOICE_NAME]}


@app.post("/v1/audio/speech")
def speech(req: SpeechReq):
    if _model is None:
        raise HTTPException(503, "model still loading")
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(400, "input is empty")
    lang = req.language or DEFAULT_LANG
    if NORMALIZE and lang.lower().startswith("de"):
        text = normalize_de(text)
    speed = req.speed if (req.speed and req.speed != 1.0) else None
    try:
        wav = _synth(text, lang, speed)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"synthesis failed: {e}")
    data, mime = _encode(wav, req.response_format)
    return Response(content=data, media_type=mime)


@app.exception_handler(HTTPException)
def _http_exc(_req, exc):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"message": exc.detail, "type": "invalid_request_error"}})


_load()
