"""STT-Backend: faster_whisper (lokal, GPU).

KRITISCH: ``faster_whisper`` (zieht ctranslate2/torch nach sich) wird NUR in
load() importiert — niemals auf Modul-Ebene. Wenn STT_BACKEND≠whisper_local,
wird dieses Modul gar nicht erst importiert (siehe STTBackend.from_config).
"""

from __future__ import annotations

import asyncio

import numpy as np

from ... import audio as audio_utils
from .base import STTBackend


class WhisperLocalSTTBackend(STTBackend):
    def __init__(self, *, model: str = "large-v3-turbo", device: str = "cuda",
                 compute_type: str = "int8", beam_size: int = 5,
                 language: str | None = "de", local_files_only: bool = True):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.local_files_only = local_files_only
        self._model = None
        self._lock = asyncio.Lock()
        self.last_no_speech_prob = None

    @classmethod
    def from_config(cls, cfg) -> "WhisperLocalSTTBackend":
        return cls(
            model=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            beam_size=cfg.whisper_beam_size,
            language=cfg.stt_language,
            local_files_only=cfg.whisper_local_files_only,
        )

    async def load(self) -> None:
        try:
            from faster_whisper import WhisperModel  # lazy! GPU-Dep
        except ImportError as exc:
            from ..base import BackendError
            raise BackendError(
                "STT_BACKEND=whisper_local, aber faster_whisper ist nicht "
                "installiert. `pip install faster-whisper` (GPU-Setup) oder "
                "STT_BACKEND=openai verwenden."
            ) from exc

        def _build():
            return WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=self.local_files_only,
            )

        self._model = await asyncio.to_thread(_build)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> dict:
        return {
            "engine": "faster-whisper",
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "loaded": self.loaded,
        }

    def _transcribe_sync(self, audio_pcm: bytes, sample_rate: int) -> str:
        samples = audio_utils.pcm_bytes_to_float32_array(audio_pcm)
        # faster_whisper erwartet float32-Mono @16kHz numpy-Array direkt.
        segments, info = self._model.transcribe(
            np.asarray(samples, dtype=np.float32),
            language=self.language,
            beam_size=self.beam_size,
        )
        parts = []
        no_speech = []
        for seg in segments:
            parts.append(seg.text)
            nsp = getattr(seg, "no_speech_prob", None)
            if isinstance(nsp, (int, float)):
                no_speech.append(nsp)
        self.last_no_speech_prob = (sum(no_speech) / len(no_speech)) if no_speech else None
        return " ".join(p.strip() for p in parts if p and p.strip()).strip()

    async def transcribe(self, audio_pcm: bytes, sample_rate: int) -> str:
        if self._model is None:
            raise RuntimeError("STT nicht initialisiert (load() nicht gelaufen)")
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, audio_pcm, sample_rate)
