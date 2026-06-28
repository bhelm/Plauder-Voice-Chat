"""Shared test setup.

Hermetic, deterministic ENV values (they win over .env thanks to
override=False in the loader) and path setup so the ``plauder`` package
is importable.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Deterministic test ENV.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("FIREWORKS_API_KEY", "fw-test-dummy")
os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
os.environ.setdefault("FIREWORKS_MODEL", "accounts/fireworks/models/glm-5p2")
os.environ.setdefault("HOUSE_MODE", "0")
os.environ.setdefault("AGENT_NAME", "Antonia")
# Backends default to cloud (no GPU in tests).
os.environ.setdefault("STT_BACKEND", "openai")
os.environ.setdefault("TTS_BACKEND", "openai")
os.environ.setdefault("LLM_BACKEND", "openai_compat")
