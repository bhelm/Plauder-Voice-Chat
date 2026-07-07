"""TTS backend abstraction."""

from __future__ import annotations

import abc


class TTSBackend(abc.ABC):
    """Speech synthesis. Implementations import heavy deps (omnivoice,
    torch) ONLY lazily in load()/__init__, never at module level.
    """

    #: Default sample rate of the backend (best-effort estimate before load()).
    sample_rate: int = 24000

    @abc.abstractmethod
    async def load(self) -> None:
        """Initializes the backend (load client/model)."""

    @abc.abstractmethod
    async def synth(self, text: str, *, speed: float = 1.0,
                    voice: str | None = None) -> tuple[bytes, int]:
        """Synthesizes text → (16-bit mono PCM bytes, sample_rate).

        ``voice`` optionally overrides the backend's default voice for this one
        call (e.g. a cloned-voice id from the voice library). Backends that
        don't support per-call voices ignore it.
        """

    async def synth_stream(self, text: str, *, speed: float = 1.0,
                           voice: str | None = None):
        """Async generator: yields ``(pcm_bytes, sample_rate)`` chunks as soon
        as they are ready. The server sends each chunk progressively to the
        client (gapless playback). ``voice`` overrides the default voice for
        this call (see ``synth``).

        Default fallback: a single chunk with the complete ``synth`` result.
        Backends with native audio streaming override this.
        """
        pcm, sr = await self.synth(text, speed=speed, voice=voice)
        if pcm:
            yield pcm, sr

    @property
    def loaded(self) -> bool:  # pragma: no cover - trivial default
        return True

    def describe(self) -> dict:
        return {"engine": self.__class__.__name__, "sample_rate": self.sample_rate}

    @staticmethod
    def from_config(cfg) -> "TTSBackend":
        name = cfg.tts_backend
        if name == "openai":
            from .openai_api import OpenAITTSBackend
            return OpenAITTSBackend.from_config(cfg)
        if name == "omnivoice_local":
            from .omnivoice_local import OmniVoiceLocalTTSBackend
            return OmniVoiceLocalTTSBackend.from_config(cfg)
        from ..base import BackendError
        raise BackendError(f"Unknown TTS_BACKEND: {name!r}")
