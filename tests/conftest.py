"""Shared test setup.

Hermetic, deterministic ENV values (they win over .env thanks to
override=False in the loader) and path setup so the ``plauder`` package
is importable.

This module also holds the SHARED test doubles and WS drain helper so the
individual test modules don't cross-import each other or re-implement them.
The fake backends are intentionally **duck-typed** (they do NOT subclass the
abstract backend bases) so that the orchestration code's ``getattr`` fallbacks
get exercised — keep it that way.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

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


# --------------------------------------------------------------------------- #
# State-leak guards: restore the module globals that ``configure()`` sets (and
# the ``WAKE_CLOSE_GUARD_S`` module-level tunable) around every test, so an
# injected backend or a poked tunable can't bleed into a later test.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _restore_server_state():
    from plauder import server as srv

    saved_globals = {
        name: getattr(srv, name)
        for name in ("CFG", "STT", "TTS", "CONV", "BRIDGE", "GHOST", "SPEAKER", "VOICES")
    }
    saved_guard = srv.WAKE_CLOSE_GUARD_S
    try:
        yield
    finally:
        for name, value in saved_globals.items():
            setattr(srv, name, value)
        srv.WAKE_CLOSE_GUARD_S = saved_guard


# --------------------------------------------------------------------------- #
# Shared mock backends (duck-typed — do NOT subclass the abstract bases).
# --------------------------------------------------------------------------- #
class FakeSTT:
    last_no_speech_prob = None

    async def transcribe(self, audio_pcm, sample_rate):
        return "hallo welt"

    def describe(self):
        return {"engine": "fake-stt", "loaded": True}


class FakeTTS:
    sample_rate = 24000

    async def synth(self, text, *, speed=1.0):
        # 4 int16-Samples
        return b"\x00\x00\x01\x00\x02\x00\x03\x00", 24000

    def describe(self):
        return {"engine": "fake-tts", "sample_rate": 24000, "loaded": True}


class FakeLLM:
    loaded = True
    last_meta = {"finish_reason": "stop", "usage": {"total_tokens": 5}}

    def __init__(self, reply="Hallo, ich bin Antonia."):
        self.reply = reply

    async def chat(self, messages, system_hint=None):
        return self.reply

    def describe(self):
        return {"engine": "fake-llm", "model": "fake", "ready": True}


# --------------------------------------------------------------------------- #
# Shared WS drain helper.
# --------------------------------------------------------------------------- #
async def _drain_until(ws, want_type, *, timeout=3.0):
    """Reads WS frames until a JSON frame with type==want_type arrives.
    Collects all seen types + the first binary, if any."""
    seen = []
    binary = None
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=max(0.05, deadline - loop.time()))
        except asyncio.TimeoutError:
            break
        if msg.type.name == "BINARY":
            binary = msg.data
            seen.append("__binary__")
            continue
        if msg.type.name in ("CLOSE", "CLOSING", "CLOSED", "ERROR"):
            break
        data = msg.json()
        seen.append(data.get("type"))
        if data.get("type") == want_type:
            return data, seen, binary
    return None, seen, binary
