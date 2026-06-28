"""TTS-Backend: OpenAI TTS (tts-1 / tts-1-hd, Cloud).

Liefert raw PCM (16-bit signed LE, mono) bei response_format="pcm". Die
Sample-Rate ist fix 24 kHz. Optionales satzweises Splitting (gegen niedrige
Erst-Latenz) wird hier umgesetzt.
"""

from __future__ import annotations

import asyncio

import numpy as np

from ... import audio as audio_utils
from ...config import OPENAI_TTS_SAMPLE_RATE
from .base import TTSBackend


class OpenAITTSBackend(TTSBackend):
    def __init__(self, *, api_key: str, model: str = "tts-1", voice: str = "nova",
                 base_url: str | None = None, sentence_split: bool = False,
                 max_chars: int = 220, gap_ms: int = 120,
                 sample_rate: int = OPENAI_TTS_SAMPLE_RATE,
                 local_speed: bool = False):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.base_url = base_url
        self.sentence_split = sentence_split
        self.max_chars = max_chars
        self.gap_ms = gap_ms
        self.sample_rate = sample_rate
        # local_speed=True: `speed` wird NICHT an den Server geschickt, sondern
        # lokal per tonhöhen-erhaltender Zeitdehnung umgesetzt — für Server, die
        # den OpenAI-`speed`-Parameter ignorieren (z.B. lokales XTTS).
        self.local_speed = local_speed
        self._client = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, cfg) -> "OpenAITTSBackend":
        return cls(
            api_key=cfg.tts_openai_api_key,
            model=cfg.tts_openai_model,
            voice=cfg.tts_openai_voice,
            base_url=cfg.tts_openai_base_url,
            sentence_split=cfg.tts_sentence_split,
            max_chars=cfg.tts_max_chars_per_chunk,
            gap_ms=cfg.tts_sentence_gap_ms,
            sample_rate=getattr(cfg, "tts_openai_sample_rate", OPENAI_TTS_SAMPLE_RATE),
            local_speed=getattr(cfg, "tts_openai_local_speed", False),
        )

    async def load(self) -> None:
        if not self.api_key:
            from ..base import BackendError
            raise BackendError(
                "OpenAI-TTS braucht einen API-Key (TTS_OPENAI_API_KEY / OPENAI_API_KEY).")
        from openai import OpenAI
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    @property
    def loaded(self) -> bool:
        return self._client is not None

    def describe(self) -> dict:
        return {
            "engine": "openai-tts",
            "model": self.model,
            "voice": self.voice,
            "device": "openai-api",
            "sample_rate": self.sample_rate,
            "loaded": self.loaded,
        }

    @staticmethod
    def _clamp_speed(speed) -> float:
        try:
            s = float(speed)
        except (TypeError, ValueError):
            return 1.0
        return max(0.25, min(4.0, s))

    def _synth_one(self, text: str, speed: float) -> np.ndarray:
        # Bei local_speed das Tempo lokal (in _synth_sync) anwenden → Server mit
        # speed=1.0 ansprechen, sonst würde es doppelt wirken (bzw. ignoriert).
        api_speed = 1.0 if self.local_speed else self._clamp_speed(speed)
        resp = self._client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
            speed=api_speed,
        )
        pcm_bytes = resp.read() if hasattr(resp, "read") else bytes(resp.content)
        return audio_utils.pcm16_bytes_to_float32(pcm_bytes)

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
        # local_speed: Tempo tonhöhen-erhaltend lokal umsetzen (Server ignoriert speed).
        if self.local_speed:
            rate = self._clamp_speed(speed)
            if abs(rate - 1.0) > 1e-3:
                samples = audio_utils.time_stretch(samples, rate, self.sample_rate)
        # float32 [-1,1] → int16-PCM-Bytes
        clipped = np.clip(samples, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()

    async def synth(self, text: str, *, speed: float = 1.0) -> tuple[bytes, int]:
        if self._client is None:
            raise RuntimeError("TTS nicht initialisiert (load() nicht gelaufen)")
        async with self._lock:
            pcm = await asyncio.to_thread(self._synth_sync, text, speed)
        return pcm, self.sample_rate
