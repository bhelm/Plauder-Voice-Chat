"""TTS backend: OpenAI TTS (tts-1 / tts-1-hd, cloud).

Returns raw PCM (16-bit signed LE, mono) with response_format="pcm". The
sample rate is fixed at 24 kHz. Optional sentence-wise splitting (to reduce
first-audio latency) is implemented here.
"""

from __future__ import annotations

import asyncio
import threading

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
        # local_speed=True: `speed` is NOT sent to the server, but applied
        # locally via pitch-preserving time stretching — for servers that
        # ignore the OpenAI `speed` parameter (e.g. local XTTS).
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
                "OpenAI TTS needs an API key (TTS_OPENAI_API_KEY / OPENAI_API_KEY).")
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

    def _synth_one(self, text: str, speed: float, voice: str | None = None) -> np.ndarray:
        # With local_speed, apply the tempo locally (in _synth_sync) → call the
        # server with speed=1.0, otherwise it would apply twice (or be ignored).
        api_speed = 1.0 if self.local_speed else self._clamp_speed(speed)
        resp = self._client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=text,
            response_format="pcm",
            speed=api_speed,
        )
        pcm_bytes = resp.read() if hasattr(resp, "read") else bytes(resp.content)
        return audio_utils.pcm16_bytes_to_float32(pcm_bytes)

    def _synth_sync(self, text: str, speed: float, voice: str | None = None) -> bytes:
        if not self.sentence_split:
            samples = self._synth_one(text, speed, voice)
        else:
            pieces = audio_utils.split_text_for_tts(text, self.max_chars)
            if len(pieces) <= 1:
                samples = self._synth_one(pieces[0] if pieces else text, speed, voice)
            else:
                gap = np.zeros(int(max(0, self.gap_ms) / 1000.0 * self.sample_rate),
                               dtype=np.float32)
                parts: list[np.ndarray] = []
                for i, piece in enumerate(pieces):
                    parts.append(self._synth_one(piece, speed, voice))
                    if gap.size and i < len(pieces) - 1:
                        parts.append(gap)
                samples = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        # local_speed: apply tempo locally, pitch-preserving (server ignores speed).
        if self.local_speed:
            rate = self._clamp_speed(speed)
            if abs(rate - 1.0) > 1e-3:
                samples = audio_utils.time_stretch(samples, rate, self.sample_rate)
        # float32 [-1,1] → int16 PCM bytes
        return audio_utils.float32_to_pcm16_bytes(samples)

    async def synth(self, text: str, *, speed: float = 1.0,
                    voice: str | None = None) -> tuple[bytes, int]:
        if self._client is None:
            raise RuntimeError("TTS not initialized (load() did not run)")
        async with self._lock:
            pcm = await asyncio.to_thread(self._synth_sync, text, speed, voice)
        return pcm, self.sample_rate

    # ~120 ms of PCM per streamed chunk — small enough for fast first audio,
    # large enough to keep the WS frame count sane.
    _STREAM_CHUNK_S = 0.12

    async def synth_stream(self, text: str, *, speed: float = 1.0,
                           voice: str | None = None):
        """True chunked streaming: PCM is yielded while the server is still
        synthesizing (OpenAI-compatible servers like Kokoro-FastAPI stream the
        response body). Falls back to the buffered ``synth`` when local time
        stretching is on (needs the complete clip) or the SDK lacks streaming.
        """
        if self._client is None:
            raise RuntimeError("TTS not initialized (load() did not run)")
        streaming_api = getattr(self._client.audio.speech, "with_streaming_response", None)
        if self.local_speed or streaming_api is None:
            pcm, sr = await self.synth(text, speed=speed, voice=voice)
            if pcm:
                yield pcm, sr
            return

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        chunk_bytes = max(2, int(self.sample_rate * self._STREAM_CHUNK_S) * 2)
        api_speed = self._clamp_speed(speed)
        req_voice = voice or self.voice

        def _produce():
            try:
                with streaming_api.create(
                        model=self.model, voice=req_voice, input=text,
                        response_format="pcm", speed=api_speed) as resp:
                    carry = b""  # keep 16-bit sample alignment across chunks
                    for chunk in resp.iter_bytes(chunk_size=chunk_bytes):
                        if stop.is_set():
                            return
                        data = carry + chunk
                        keep = len(data) - (len(data) % 2)
                        carry = data[keep:]
                        if keep:
                            loop.call_soon_threadsafe(q.put_nowait, ("pcm", data[:keep]))
                    if carry and not stop.is_set():
                        loop.call_soon_threadsafe(q.put_nowait, ("pcm", carry + b"\x00"))
            except Exception as exc:  # surfaced in the async generator below
                if not stop.is_set():
                    loop.call_soon_threadsafe(q.put_nowait, ("err", exc))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("end", None))

        producer = asyncio.to_thread(_produce)
        producer_task = asyncio.ensure_future(producer)
        try:
            while True:
                kind, payload = await q.get()
                if kind == "pcm":
                    yield payload, self.sample_rate
                elif kind == "err":
                    raise payload
                else:
                    break
        finally:
            stop.set()
            await asyncio.gather(producer_task, return_exceptions=True)
