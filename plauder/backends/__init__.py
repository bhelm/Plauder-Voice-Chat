"""Pluggable Backends (STT/TTS/LLM). Konkrete Implementierungen werden lazy
über die jeweilige ``from_config``-Factory importiert — inaktive Backends
(insb. GPU-Backends) werden nie geladen."""
from .base import BackendError, LLMBackend, STTBackend, TTSBackend, UpstreamTimeoutError

__all__ = ["BackendError", "UpstreamTimeoutError", "STTBackend", "TTSBackend", "LLMBackend"]
