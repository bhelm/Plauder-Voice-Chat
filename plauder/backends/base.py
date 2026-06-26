"""Gemeinsame Backend-Abstraktionen.

Re-exportiert die drei abstrakten Backend-Basisklassen und definiert
``UpstreamTimeoutError`` (vom LLM-Layer geworfen, vom Server für Retry genutzt).
"""

from __future__ import annotations


class BackendError(RuntimeError):
    """Backend konnte nicht geladen werden (fehlende Deps, Key, Modell …)."""


class UpstreamTimeoutError(RuntimeError):
    """LLM-Gateway hat 408 / upstream provider timeout zurückgegeben."""


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
