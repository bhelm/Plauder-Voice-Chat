"""STT-Backend: OpenAI Whisper API (whisper-1).

Browser liefert 16 kHz float32-PCM; wir packen jedes Segment in ein In-Memory-WAV
und schicken es an /v1/audio/transcriptions. Keine GPU, kein lokales Modell.
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
        self._lock = asyncio.Lock()
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
                "OpenAI-STT braucht einen API-Key (STT_OPENAI_API_KEY / OPENAI_API_KEY).")
        from openai import OpenAI  # cloud SDK, kein GPU-Dep
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
        buf.name = "audio.wav"  # OpenAI leitet das Format aus dem Namen ab
        req = {"model": self.model, "file": buf, "response_format": "json"}
        if self.language:
            req["language"] = self.language
        resp = self._client.audio.transcriptions.create(**req)
        # OpenAI liefert kein no_speech_prob.
        self.last_no_speech_prob = None
        return (getattr(resp, "text", None) or "").strip()

    async def transcribe(self, audio_pcm: bytes, sample_rate: int) -> str:
        if self._client is None:
            raise RuntimeError("STT nicht initialisiert (load() nicht gelaufen)")
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, audio_pcm, sample_rate)
