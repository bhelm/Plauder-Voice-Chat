"""Gemeinsames Test-Setup.

Hermetische, deterministische ENV-Werte (gewinnen gegen .env wegen
override=False im Loader) und Pfad-Setup, damit das ``plauder``-Package
importierbar ist.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Deterministische Test-ENV.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("FIREWORKS_API_KEY", "fw-test-dummy")
os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
os.environ.setdefault("FIREWORKS_MODEL", "accounts/fireworks/models/glm-5p2")
os.environ.setdefault("HOUSE_MODE", "0")
os.environ.setdefault("AGENT_NAME", "Antonia")
# Backends standardmäßig Cloud (keine GPU im Test).
os.environ.setdefault("STT_BACKEND", "openai")
os.environ.setdefault("TTS_BACKEND", "openai")
os.environ.setdefault("LLM_BACKEND", "openai_compat")
