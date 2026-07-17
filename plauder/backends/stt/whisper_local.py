"""STT backend: faster_whisper (local, GPU).

CRITICAL: ``faster_whisper`` (which pulls in ctranslate2/torch) is imported ONLY
in load() — never at module level. If STT_BACKEND≠whisper_local, this module is
not even imported in the first place (see STTBackend.from_config).
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from ... import audio as audio_utils
from .base import STTBackend

LOG = logging.getLogger("voice-chat")


class WhisperLocalSTTBackend(STTBackend):
    def __init__(self, *, model: str = "large-v3-turbo", device: str = "cuda",
                 compute_type: str = "int8", beam_size: int = 5,
                 language: str | None = "de", local_files_only: bool = True,
                 condition_on_previous_text: bool = False):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.local_files_only = local_files_only
        self.condition_on_previous_text = condition_on_previous_text
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
            condition_on_previous_text=cfg.whisper_condition_on_previous_text,
        )

    async def load(self) -> None:
        try:
            from faster_whisper import WhisperModel  # lazy! GPU dep
        except ImportError as exc:
            from ..base import BackendError
            raise BackendError(
                "STT_BACKEND=whisper_local, but faster_whisper is not "
                "installed. `pip install faster-whisper` (GPU setup) or "
                "use STT_BACKEND=openai."
            ) from exc

        def _build():
            # "Download only when not already local" — the requested behavior:
            # ALWAYS try an offline load first (from the HF cache or a local
            # path). Only if the model is genuinely missing do we fall back to
            # a network download. This avoids the per-start HuggingFace API
            # revision check that `local_files_only=False` triggers even when
            # the weights are already cached.
            #
            # An explicit WHISPER_LOCAL_FILES_ONLY=1 (self.local_files_only)
            # stays a hard offline lock: never touch the network, even to
            # fetch a missing model — the startup then fails loudly, as before.
            try:
                return WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    local_files_only=True,
                )
            except Exception as exc:
                if self.local_files_only:
                    # Hard offline lock requested — do not download; re-raise.
                    raise
                LOG.info(
                    "whisper: %r not in local cache (%s) — downloading once",
                    self.model_name, type(exc).__name__)
                return WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    local_files_only=False,
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
        # faster_whisper expects a float32 mono @16kHz numpy array directly.
        segments, info = self._model.transcribe(
            np.asarray(samples, dtype=np.float32),
            language=self.language,
            beam_size=self.beam_size,
            condition_on_previous_text=self.condition_on_previous_text,
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
            raise RuntimeError("STT not initialized (load() did not run)")
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, audio_pcm, sample_rate)
