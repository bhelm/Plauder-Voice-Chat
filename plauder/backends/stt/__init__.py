"""STT backends. Concrete modules are loaded lazily via STTBackend.from_config."""
from .base import STTBackend

__all__ = ["STTBackend"]
