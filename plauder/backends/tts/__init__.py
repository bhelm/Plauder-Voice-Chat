"""TTS backends. Concrete modules are loaded lazily via TTSBackend.from_config."""
from .base import TTSBackend

__all__ = ["TTSBackend"]
