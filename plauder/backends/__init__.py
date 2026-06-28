"""Pluggable backends (STT/TTS/LLM). Concrete implementations are imported
lazily via their respective ``from_config`` factory — inactive backends
(especially GPU backends) are never loaded."""
from .base import BackendError, LLMBackend, STTBackend, TTSBackend, UpstreamTimeoutError

__all__ = ["BackendError", "UpstreamTimeoutError", "STTBackend", "TTSBackend", "LLMBackend"]
