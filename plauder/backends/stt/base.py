"""STT-Backend-Abstraktion."""

from __future__ import annotations

import abc


class STTBackend(abc.ABC):
    """Spracherkennung. Implementierungen importieren schwere Deps (faster_whisper,
    torch) NUR lazy in load()/__init__, niemals auf Modul-Ebene.
    """

    #: no_speech_prob des zuletzt transkribierten Segments (None, wenn das
    #: Backend keins liefert — z.B. OpenAI). Direkt nach ``await transcribe()``
    #: lesen (race-frei dank kooperativem Scheduling).
    last_no_speech_prob: float | None = None

    @abc.abstractmethod
    async def load(self) -> None:
        """Initialisiert das Backend (Client/Modell laden)."""

    @abc.abstractmethod
    async def transcribe(self, audio_pcm: bytes, sample_rate: int) -> str:
        """Transkribiert rohe float32-PCM-Bytes (Browser-Format) zu Text."""

    @property
    def loaded(self) -> bool:  # pragma: no cover - trivial default
        return True

    def describe(self) -> dict:
        """Healthz-Info für dieses Backend."""
        return {"engine": self.__class__.__name__}

    @staticmethod
    def from_config(cfg) -> "STTBackend":
        """Factory: wählt anhand von cfg.stt_backend. Importiert NUR das
        gewählte Backend-Modul (lazy)."""
        name = cfg.stt_backend
        if name == "openai":
            from .openai_api import OpenAISTTBackend
            return OpenAISTTBackend.from_config(cfg)
        if name == "whisper_local":
            from .whisper_local import WhisperLocalSTTBackend
            return WhisperLocalSTTBackend.from_config(cfg)
        from ..base import BackendError
        raise BackendError(f"Unbekanntes STT_BACKEND: {name!r}")
