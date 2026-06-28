"""Shared backend abstractions.

Re-exports the three abstract backend base classes and defines
``UpstreamTimeoutError`` (raised by the LLM layer, used by the server for retry).
"""

from __future__ import annotations


class BackendError(RuntimeError):
    """Backend could not be loaded (missing deps, key, model …)."""


class UpstreamTimeoutError(RuntimeError):
    """LLM gateway returned 408 / upstream provider timeout."""


from .stt.base import STTBackend  # noqa: E402
from .tts.base import TTSBackend  # noqa: E402
from .llm.base import LLMBackend  # noqa: E402

__all__ = [
    "BackendError",
    "UpstreamTimeoutError",
    "STTBackend",
    "TTSBackend",
    "LLMBackend",
]
