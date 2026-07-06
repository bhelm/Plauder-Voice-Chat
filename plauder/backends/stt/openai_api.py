"""STT backend: OpenAI Whisper API (whisper-1).

The browser delivers 16 kHz float32 PCM; we pack each segment into an in-memory
WAV and send it to /v1/audio/transcriptions. No GPU, no local model.
"""

from __future__ import annotations

import asyncio
import io
import time

import numpy as np

from ... import audio as audio_utils
from .base import STTBackend


class OpenAISTTBackend(STTBackend):
    def __init__(self, *, api_key: str, model: str = "whisper-1",
                 language: str | None = "de", base_url: str | None = None):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.base_url = base_url
        self._client = None
        self.last_no_speech_prob = None

    @classmethod
    def from_config(cls, cfg) -> "OpenAISTTBackend":
        return cls(
            api_key=cfg.stt_openai_api_key,
            model=cfg.stt_openai_model,
            language=cfg.stt_language,
            base_url=cfg.stt_openai_base_url,
        )

    async def load(self) -> None:
        if not self.api_key:
            from ..base import BackendError
            raise BackendError(
                "OpenAI STT needs an API key (STT_OPENAI_API_KEY / OPENAI_API_KEY).")
        from openai import OpenAI  # cloud SDK, no GPU dep
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    @property
    def loaded(self) -> bool:
        return self._client is not None

    def describe(self) -> dict:
        return {
            "engine": "openai-whisper",
            "model": self.model,
            "device": "openai-api",
            "language": self.language,
            "loaded": self.loaded,
        }

    def _transcribe_sync(self, audio_pcm: bytes, sample_rate: int) -> str:
        samples = audio_utils.pcm_bytes_to_float32_array(audio_pcm)
        wav_bytes = audio_utils.float32_to_pcm16_wav_bytes(samples, sample_rate)
        buf = io.BytesIO(wav_bytes)
        buf.name = "audio.wav"  # OpenAI infers the format from the name
        req = {"model": self.model, "file": buf, "response_format": "json"}
        if self.language:
            req["language"] = self.language
        resp = self._client.audio.transcriptions.create(**req)
        # OpenAI does not provide no_speech_prob.
        self.last_no_speech_prob = None
        return (getattr(resp, "text", None) or "").strip()

    async def transcribe(self, audio_pcm: bytes, sample_rate: int) -> str:
        if self._client is None:
            raise RuntimeError("STT not initialized (load() did not run)")
        # No lock: the SDK's httpx client is thread-safe and pools connections.
        # Serializing here made the FINAL transcript (which gates the whole
        # turn) wait behind an in-flight B2 partial of the same segment.
        return await asyncio.to_thread(self._transcribe_sync, audio_pcm, sample_rate)
