"""TTS-Backend-Abstraktion."""

from __future__ import annotations

import abc


class TTSBackend(abc.ABC):
    """Sprachsynthese. Implementierungen importieren schwere Deps (omnivoice,
    torch) NUR lazy in load()/__init__, niemals auf Modul-Ebene.
    """

    #: Default-Sample-Rate des Backends (vor load() bestmögliche Schätzung).
    sample_rate: int = 24000

    @abc.abstractmethod
    async def load(self) -> None:
        """Initialisiert das Backend (Client/Modell laden)."""

    @abc.abstractmethod
    async def synth(self, text: str, *, speed: float = 1.0) -> tuple[bytes, int]:
        """Synthetisiert Text → (16-bit-Mono-PCM-Bytes, sample_rate)."""

    async def synth_stream(self, text: str, *, speed: float = 1.0):
        """Async-Generator: liefert ``(pcm_bytes, sample_rate)``-Häppchen, sobald
        sie bereitstehen. Der Server schickt jedes Häppchen progressiv an den
        Client (lückenlose Wiedergabe).

        Default-Fallback: ein einzelnes Häppchen mit dem kompletten ``synth``-
        Ergebnis. Backends mit nativem Audio-Streaming überschreiben das.
        """
        pcm, sr = await self.synth(text, speed=speed)
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
        raise BackendError(f"Unbekanntes TTS_BACKEND: {name!r}")
