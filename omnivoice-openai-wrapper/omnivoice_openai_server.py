#!/usr/bin/env python
"""OpenAI-compatible /v1/audio/speech server backed by OmniVoice (k2-fsa).

Drop-in replacement for a Kokoro/OpenAI TTS endpoint: any client that speaks
the OpenAI `POST /v1/audio/speech` protocol (the voice-chat app with
``TTS_BACKEND=openai``, but also Hermes / OpenClaw or any other harness) can
point at it unchanged.

Loads the OmniVoice model once and serves TTS. On top of the plain OpenAI
surface it keeps a **persistent voice library**: reference samples (recorded or
uploaded) are stored under OMNIVOICE_VOICES_DIR as ``{id}.wav`` + ``{id}.json``
and turned into reusable voice-clone prompts (cached in RAM, LRU-bounded to keep
VRAM in check). Callers pick a voice by putting its id in the OpenAI ``voice``
field; a frozen built-in voice (id ``default``) is always available and can't be
deleted. CRUD lives under ``/v1/audio/voices``. GPU is pinned via
CUDA_VISIBLE_DEVICES (see the systemd unit).

Config via environment (all optional; defaults are project-relative):
  OMNIVOICE_MODEL     HF model id                (default k2-fsa/OmniVoice)
  OMNIVOICE_REF_WAV   clean voice-clone reference WAV   (default ./ref/ref.wav)
  OMNIVOICE_REF_TXT   exact transcript of that WAV      (default ./ref/ref.txt)
  OMNIVOICE_VOICES_DIR persistent voice-library dir     (default ./voices)
  OMNIVOICE_LANG      default synthesis language        (default de)
  OMNIVOICE_VOICE     display name of the built-in voice (default de_female)
  OMNIVOICE_NUM_STEP  diffusion steps (quality/latency) (default 32)
  OMNIVOICE_NORMALIZE 1=expand German digits/dates/etc. (default 1)
  OMNIVOICE_PROMPT_CACHE  max cloned prompts kept in VRAM (default 8)
"""
import asyncio
import io
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
VOICES_DIR = os.environ.get("OMNIVOICE_VOICES_DIR", os.path.join(_HERE, "voices"))
DEFAULT_LANG = os.environ.get("OMNIVOICE_LANG", "de")
VOICE_NAME = os.environ.get("OMNIVOICE_VOICE", "de_female")
NUM_STEP = int(os.environ.get("OMNIVOICE_NUM_STEP", "32"))
PROMPT_CACHE_MAX = max(1, int(os.environ.get("OMNIVOICE_PROMPT_CACHE", "8")))

DEFAULT_ID = "default"
_ID_RE = re.compile(r"[0-9a-f]{32}")  # minted voice ids only (uuid4().hex)

# Everything below is guarded by the single GPU lock: the model, the frozen
# default prompt, the voice metadata registry and the LRU prompt cache. One lock
# keeps prompt-building, generation and registry mutation from racing on the GPU
# or on the dicts (the FastAPI sync endpoints run in a threadpool).
_lock = threading.Lock()
_model = None
_gen_cfg = None
_default_prompt = None
_voices: dict = {}                 # id -> {id, name, ref_text, created, isDefault}
_prompts: "OrderedDict" = OrderedDict()  # id -> clone prompt (LRU; default NOT stored here)


def _wav_path(voice_id: str) -> str:
    return os.path.join(VOICES_DIR, f"{voice_id}.wav")


def _meta_path(voice_id: str) -> str:
    return os.path.join(VOICES_DIR, f"{voice_id}.json")


def _load_voices() -> None:
    """Populate the in-RAM registry: the synthetic built-in default entry plus
    every persisted voice on disk. Prompts are built lazily on first use."""
    _voices.clear()
    try:
        with open(REF_TXT, "r", encoding="utf-8") as fh:
            default_ref = fh.read().strip()
    except OSError:
        default_ref = ""
    _voices[DEFAULT_ID] = {
        "id": DEFAULT_ID, "name": VOICE_NAME, "ref_text": default_ref,
        "created": 0, "isDefault": True,
    }
    if not os.path.isdir(VOICES_DIR):
        return
    for fn in os.listdir(VOICES_DIR):
        if not fn.endswith(".json"):
            continue
        vid = fn[:-5]
        if not _ID_RE.fullmatch(vid) or not os.path.isfile(_wav_path(vid)):
            continue
        try:
            with open(_meta_path(vid), "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        _voices[vid] = {
            "id": vid,
            "name": str(meta.get("name") or vid),
            "ref_text": str(meta.get("ref_text") or ""),
            "created": float(meta.get("created") or 0),
            "isDefault": False,
        }


def _resolve_prompt_locked(voice_id: str):
    """Return the clone prompt for a voice id. Assumes _lock is held. Unknown or
    ``default`` id → the frozen built-in prompt (audio never breaks on a stale
    id, e.g. after a restart that dropped an in-flight cloned voice)."""
    if not voice_id or voice_id == DEFAULT_ID or voice_id not in _voices:
        return _default_prompt
    if voice_id in _prompts:
        _prompts.move_to_end(voice_id)
        return _prompts[voice_id]
    meta = _voices[voice_id]
    prompt = _model.create_voice_clone_prompt(
        ref_audio=_wav_path(voice_id), ref_text=meta["ref_text"])
    _prompts[voice_id] = prompt
    while len(_prompts) > PROMPT_CACHE_MAX:
        _prompts.popitem(last=False)  # evict least-recently-used
    return prompt


def _load() -> None:
    global _model, _gen_cfg, _default_prompt
    os.makedirs(VOICES_DIR, exist_ok=True)
    print(f">> loading {MODEL_ID} ...", flush=True)
    _model = OmniVoice.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    _gen_cfg = OmniVoiceGenerationConfig(num_step=NUM_STEP, guidance_scale=2.0)
    with open(REF_TXT, "r", encoding="utf-8") as fh:
        ref_text = fh.read().strip()
    print(f">> freezing built-in voice clone prompt from {REF_WAV}", flush=True)
    _default_prompt = _model.create_voice_clone_prompt(ref_audio=REF_WAV, ref_text=ref_text)
    _load_voices()
    print(f">> voice library: {len(_voices)} voice(s) ({VOICES_DIR})", flush=True)
    # warmup so the first real request isn't slowed by CUDA/cudnn autotune
    try:
        _synth("Guten Tag.", DEFAULT_LANG, None, DEFAULT_ID)
        print(">> warmup done", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f">> warmup skipped: {e}", flush=True)
    print(">> ready", flush=True)


def _synth(text: str, language: str, speed, voice_id: str) -> np.ndarray:
    with _lock, torch.inference_mode():
        prompt = _resolve_prompt_locked(voice_id)
        out = _model.generate(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            speed=speed,
            generation_config=_gen_cfg,
        )
    return np.asarray(out[0], dtype=np.float32)


def _decode_to_ref_wav(data: bytes, out_path: str) -> None:
    """Decode arbitrary uploaded/recorded audio (wav/mp3/m4a/ogg/webm/flac …)
    to a clean 24 kHz mono 16-bit WAV reference via ffmpeg (pydub)."""
    seg = AudioSegment.from_file(io.BytesIO(data))
    seg = seg.set_frame_rate(SR).set_channels(1).set_sample_width(2)
    seg.export(out_path, format="wav")


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
    voice: str | None = DEFAULT_ID
    response_format: str | None = "mp3"
    speed: float | None = 1.0
    language: str | None = None  # non-standard convenience override


class RenameReq(BaseModel):
    name: str


def _voice_public(meta: dict) -> dict:
    return {"id": meta["id"], "name": meta["name"],
            "created": meta["created"], "isDefault": meta["isDefault"]}


def _voices_sorted() -> list:
    # default first, then newest-created first
    return sorted(_voices.values(),
                  key=lambda m: (not m["isDefault"], -m["created"]))


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading",
            "voice": VOICE_NAME, "voices": len(_voices)}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "omnivoice", "object": "model", "owned_by": "k2-fsa"}]}


@app.get("/v1/audio/voices")
def list_voices():
    with _lock:
        return {"voices": [_voice_public(m) for m in _voices_sorted()]}


def _register_voice_sync(data: bytes, name: str, ref_text: str) -> dict:
    """Decode → clone → persist a new voice (blocking; runs in a threadpool).
    Rolls back partial files on any failure. Raises HTTPException on bad input."""
    vid = uuid.uuid4().hex
    wav_path = _wav_path(vid)
    try:
        _decode_to_ref_wav(data, wav_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not decode audio: {e}")
    meta = {"id": vid, "name": name, "ref_text": ref_text,
            "created": time.time(), "isDefault": False}
    try:
        with _lock, torch.inference_mode():
            prompt = _model.create_voice_clone_prompt(ref_audio=wav_path, ref_text=ref_text)
            _prompts[vid] = prompt
            while len(_prompts) > PROMPT_CACHE_MAX:
                _prompts.popitem(last=False)
            _voices[vid] = meta
        with open(_meta_path(vid), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        _voices.pop(vid, None); _prompts.pop(vid, None)
        for p in (wav_path, _meta_path(vid)):
            try:
                os.remove(p)
            except OSError:
                pass
        raise HTTPException(500, f"clone failed: {e}")
    return meta


@app.post("/v1/audio/voices")
async def create_voice(file: UploadFile = File(...),
                       name: str = Form(...),
                       ref_text: str = Form(...)):
    if _model is None:
        raise HTTPException(503, "model still loading")
    name = (name or "").strip() or "Stimme"
    ref_text = (ref_text or "").strip()
    if not ref_text:
        raise HTTPException(400, "ref_text is required (transcript of the sample)")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    # Offload the blocking decode + GPU clone so the event loop stays responsive.
    meta = await asyncio.to_thread(_register_voice_sync, data, name, ref_text)
    return _voice_public(meta)


@app.patch("/v1/audio/voices/{voice_id}")
def rename_voice(voice_id: str, req: RenameReq):
    if not _ID_RE.fullmatch(voice_id) or voice_id not in _voices:
        raise HTTPException(404, "unknown voice")
    if _voices[voice_id]["isDefault"]:
        raise HTTPException(400, "cannot modify the built-in voice")
    new_name = (req.name or "").strip()
    if not new_name:
        raise HTTPException(400, "name is required")
    with _lock:
        _voices[voice_id]["name"] = new_name
        meta = dict(_voices[voice_id])
    with open(_meta_path(voice_id), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    return _voice_public(meta)


@app.delete("/v1/audio/voices/{voice_id}")
def delete_voice(voice_id: str):
    if not _ID_RE.fullmatch(voice_id) or voice_id not in _voices:
        raise HTTPException(404, "unknown voice")
    if _voices[voice_id]["isDefault"]:
        raise HTTPException(400, "cannot delete the built-in voice")
    with _lock:
        _voices.pop(voice_id, None)
        _prompts.pop(voice_id, None)
    for p in (_wav_path(voice_id), _meta_path(voice_id)):
        try:
            os.remove(p)
        except OSError:
            pass
    return {"ok": True, "id": voice_id}


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
        wav = _synth(text, lang, speed, req.voice or DEFAULT_ID)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"synthesis failed: {e}")
    data, mime = _encode(wav, req.response_format)
    return Response(content=data, media_type=mime)


@app.exception_handler(HTTPException)
def _http_exc(_req, exc):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"message": exc.detail, "type": "invalid_request_error"}})


_load()
