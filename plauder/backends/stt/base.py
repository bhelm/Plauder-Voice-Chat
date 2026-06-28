"""STT backend abstraction."""

from __future__ import annotations

import abc


class STTBackend(abc.ABC):
    """Speech recognition. Implementations import heavy deps (faster_whisper,
    torch) ONLY lazily in load()/__init__, never at module level.
    """

    #: no_speech_prob of the most recently transcribed segment (None if the
    #: backend does not provide one — e.g. OpenAI). Read directly after
    #: ``await transcribe()`` (race-free thanks to cooperative scheduling).
    last_no_speech_prob: float | None = None

    @abc.abstractmethod
    async def load(self) -> None:
        """Initializes the backend (load client/model)."""

    @abc.abstractmethod
    async def transcribe(self, audio_pcm: bytes, sample_rate: int) -> str:
        """Transcribes raw float32 PCM bytes (browser format) to text."""

    @property
    def loaded(self) -> bool:  # pragma: no cover - trivial default
        return True

    def describe(self) -> dict:
        """Healthz info for this backend."""
        return {"engine": self.__class__.__name__}

    @staticmethod
    def from_config(cfg) -> "STTBackend":
        """Factory: selects based on cfg.stt_backend. Imports ONLY the
        chosen backend module (lazily)."""
        name = cfg.stt_backend
        if name == "openai":
            from .openai_api import OpenAISTTBackend
            return OpenAISTTBackend.from_config(cfg)
        if name == "whisper_local":
            from .whisper_local import WhisperLocalSTTBackend
            return WhisperLocalSTTBackend.from_config(cfg)
        from ..base import BackendError
        raise BackendError(f"Unknown STT_BACKEND: {name!r}")
