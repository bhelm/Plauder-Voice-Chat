"""TTS backend: OmniVoice (local, GPU, k2-fsa/OmniVoice).

CRITICAL: ``omnivoice``/``torch`` are imported ONLY in load() — never at module
level. If TTS_BACKEND≠omnivoice_local, this module is not imported.

OmniVoice returns float32 samples @24 kHz; we convert them to 16-bit PCM bytes
to satisfy the TTSBackend contract (pcm_bytes, sample_rate).
"""

from __future__ import annotations

import asyncio

import numpy as np

from ... import audio as audio_utils
from .base import TTSBackend


class OmniVoiceLocalTTSBackend(TTSBackend):
    def __init__(self, *, model: str = "k2-fsa/OmniVoice", device: str = "cuda",
                 mode: str = "clone", ref_audio: str | None = None,
                 ref_text: str | None = None, language: str | None = None,
                 sentence_split: bool = False, max_chars: int = 220, gap_ms: int = 120):
        self.model = model
        self.device = device
        self.mode = mode
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.sentence_split = sentence_split
        self.max_chars = max_chars
        self.gap_ms = gap_ms
        self.sample_rate = 24000
        self._tts = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, cfg) -> "OmniVoiceLocalTTSBackend":
        return cls(
            model=cfg.omnivoice_model,
            device=cfg.omnivoice_device,
            mode=cfg.omnivoice_mode,
            ref_audio=cfg.omnivoice_ref_audio,
            ref_text=cfg.omnivoice_ref_text,
            language=cfg.omnivoice_language,
            sentence_split=cfg.tts_sentence_split,
            max_chars=cfg.tts_max_chars_per_chunk,
            gap_ms=cfg.tts_sentence_gap_ms,
        )

    async def load(self) -> None:
        try:
            from omnivoice import OmniVoice  # lazy! GPU dep
        except ImportError as exc:
            from ..base import BackendError
            raise BackendError(
                "TTS_BACKEND=omnivoice_local, but omnivoice is not installed. "
                "GPU setup required, or use TTS_BACKEND=openai."
            ) from exc

        def _build():
            engine = OmniVoice(self.model, device=self.device)
            self.sample_rate = getattr(engine, "sample_rate", 24000)
            return engine

        self._tts = await asyncio.to_thread(_build)

    @property
    def loaded(self) -> bool:
        return self._tts is not None

    def describe(self) -> dict:
        return {
            "engine": "omnivoice",
            "model": self.model,
            "device": self.device,
            "mode": self.mode,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "loaded": self.loaded,
        }

    def _synth_one(self, text: str, speed: float) -> np.ndarray:
        # OmniVoice API: generate(text, ...) -> (np.float32, sample_rate)
        out = self._tts.generate(
            text,
            mode=self.mode,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            language=self.language,
            speed=speed,
        )
        samples, sr = out if isinstance(out, tuple) else (out, self.sample_rate)
        self.sample_rate = sr
        return np.asarray(samples, dtype=np.float32)

    def _synth_sync(self, text: str, speed: float) -> bytes:
        if not self.sentence_split:
            samples = self._synth_one(text, speed)
        else:
            pieces = audio_utils.split_text_for_tts(text, self.max_chars)
            if len(pieces) <= 1:
                samples = self._synth_one(pieces[0] if pieces else text, speed)
            else:
                gap = np.zeros(int(max(0, self.gap_ms) / 1000.0 * self.sample_rate),
                               dtype=np.float32)
                parts: list[np.ndarray] = []
                for i, piece in enumerate(pieces):
                    parts.append(self._synth_one(piece, speed))
                    if gap.size and i < len(pieces) - 1:
                        parts.append(gap)
                samples = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        return audio_utils.float32_to_pcm16_bytes(samples)

    async def synth(self, text: str, *, speed: float = 1.0,
                    voice: str | None = None) -> tuple[bytes, int]:
        # `voice` (per-call clone-voice override) is not supported by the
        # in-process backend — the reference is fixed at construction. Ignored.
        if self._tts is None:
            raise RuntimeError("TTS not initialized (load() did not run)")
        async with self._lock:
            pcm = await asyncio.to_thread(self._synth_sync, text, speed)
        return pcm, self.sample_rate
