"""Voice library: per-call TTS voice override, active-voice persistence, and the
server-side clone helpers (plauder.voice_clone). All network-free (fake OpenAI client / fake library)."""
import asyncio
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from plauder import server as srv
from plauder import ai_voice as vc
from plauder.backends.tts.openai_api import OpenAITTSBackend
from plauder.voices import DEFAULT_VOICE_ID, VoiceLibrary


@pytest.fixture(autouse=True)
def _isolate_ai_voice_state(tmp_path, monkeypatch):
    """Keep the module globals of plauder.ai_voice out of the tests.

    ``_STATE_PATH`` points at the project root, so without this a real
    ``.ai_voice.json`` written by a running server leaks into the assertions
    (and a test could overwrite the user's actual choice).
    """
    monkeypatch.setattr(vc, "_STATE_PATH", tmp_path / ".ai_voice.json")
    monkeypatch.setattr(vc, "_state", None)
    monkeypatch.setattr(vc, "_store", vc.LocalVoiceStore(tmp_path / "voices"))
    yield
    vc._state = None
    vc._store = None


class _Cfg:
    tts_backend = "openai"
    tts_openai_api_key = "sk-test"
    tts_openai_model = "tts-1"
    tts_openai_voice = "nova"
    tts_openai_base_url = None
    tts_sentence_split = False
    tts_max_chars_per_chunk = 220
    tts_sentence_gap_ms = 120


def _openai_with_mock():
    eng = OpenAITTSBackend.from_config(_Cfg())
    resp = MagicMock()
    resp.read.return_value = np.asarray([1, 2, 3], dtype=np.int16).tobytes()
    client = MagicMock()
    client.audio.speech.create.return_value = resp
    eng._client = client
    return eng, client


# --- per-call voice override -------------------------------------------------
def test_synth_voice_override_wins_over_default():
    eng, client = _openai_with_mock()
    asyncio.run(eng.synth("hi", voice="clone-123"))
    assert client.audio.speech.create.call_args.kwargs["voice"] == "clone-123"


def test_synth_without_voice_uses_default():
    eng, client = _openai_with_mock()
    asyncio.run(eng.synth("hi"))
    assert client.audio.speech.create.call_args.kwargs["voice"] == "nova"


# --- active-voice persistence ------------------------------------------------
def test_active_voice_defaults_to_builtin(tmp_path):
    lib = VoiceLibrary("http://x/v1", state_path=str(tmp_path / ".active_voice"))
    assert lib.get_active() == DEFAULT_VOICE_ID


def test_active_voice_persists_across_instances(tmp_path):
    path = str(tmp_path / ".active_voice")
    VoiceLibrary("http://x/v1", state_path=path).set_active("abc123")
    # a fresh instance (fresh cache) reads the persisted id back
    assert VoiceLibrary("http://x/v1", state_path=path).get_active() == "abc123"


def test_set_active_empty_falls_back_to_default(tmp_path):
    lib = VoiceLibrary("http://x/v1", state_path=str(tmp_path / ".active_voice"))
    assert lib.set_active("") == DEFAULT_VOICE_ID


def test_voice_url_built_under_v1():
    lib = VoiceLibrary("http://box:8880/v1/", state_path="/tmp/none")
    assert lib._url("") == "http://box:8880/v1/audio/voices"
    assert lib._url("/abc") == "http://box:8880/v1/audio/voices/abc"


# --- server helpers ----------------------------------------------------------
class _FakeVoices:
    def __init__(self, active="clone-1"):
        self._active = active

    def get_active(self):
        return self._active

    async def list(self):
        return [
            {"id": DEFAULT_VOICE_ID, "name": "Built-in", "isDefault": True},
            {"id": "clone-1", "name": "Me", "isDefault": False},
        ]


def test_clone_active_and_active_voice_id(monkeypatch):
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(tts_clone_enabled=True, app_language="en"))
    monkeypatch.setattr(srv, "VOICES", _FakeVoices("clone-1"))
    assert vc.library_active() is True
    assert vc.active_voice_id() == "clone-1"


def test_clone_inactive_when_disabled(monkeypatch):
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(tts_clone_enabled=False, app_language="en"))
    monkeypatch.setattr(srv, "VOICES", _FakeVoices())
    assert vc.library_active() is False
    assert vc.active_voice_id() is None


def test_clone_inactive_when_no_library(monkeypatch):
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(tts_clone_enabled=True, app_language="en"))
    monkeypatch.setattr(srv, "VOICES", None)
    assert vc.library_active() is False
    assert vc.active_voice_id() is None


def test_state_block_wrapper_source(monkeypatch):
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(
        tts_clone_enabled=True, app_language="en", ai_voice_source="auto",
        omnivoice_mode="clone", omnivoice_instruct=""))
    monkeypatch.setattr(srv, "VOICES", _FakeVoices("clone-1"))
    monkeypatch.setattr(srv, "TTS", object())
    vc._state = None
    block = asyncio.run(vc.state_block())
    vc._state = None
    assert block["available"] is True and block["source"] == "wrapper"
    assert block["active"] == "clone-1"
    assert {v["id"] for v in block["voices"]} == {DEFAULT_VOICE_ID, "clone-1"}
    # Voice design is an in-process feature — the wrapper only serves voices.
    assert block["canDesign"] is False and block["canManageSamples"] is False


def test_state_block_without_any_source(monkeypatch):
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(
        tts_clone_enabled=False, app_language="en", ai_voice_source="auto",
        omnivoice_mode="clone", omnivoice_instruct=""))
    monkeypatch.setattr(srv, "VOICES", None)
    monkeypatch.setattr(srv, "TTS", object())
    vc._state = None
    block = asyncio.run(vc.state_block())
    vc._state = None
    # The section still renders — it just reports that nothing is wired.
    assert block["available"] is False and block["source"] == "none"
    assert block["voices"] == []


# --- clone commit (recording → cleanup → STT → register) ----------------------
def _f32(parts):
    return np.concatenate(parts).astype(np.float32).tobytes()


def _tone(dur_s, amp=0.25, sr=16000):
    n = int(sr * dur_s)
    return (np.sin(2 * np.pi * 220.0 * np.arange(n) / sr) * amp).astype(np.float32)


def _sil(dur_s, sr=16000):
    return np.zeros(int(sr * dur_s), dtype=np.float32)


class _FakeSTT:
    def __init__(self, text="hallo welt"):
        self.text = text
        self.buffers = []

    async def transcribe(self, buf, sr):
        self.buffers.append(buf)
        return self.text


class _FakeRegVoices:
    def __init__(self):
        self.calls = []

    async def register(self, data, *, filename, content_type, name, ref_text):
        self.calls.append({"data": data, "name": name, "ref_text": ref_text})
        return {"id": "clone-9", "name": name, "isDefault": False}


def _wire(monkeypatch, *, trim=True, stt=None, voices=None):
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(
        tts_clone_enabled=True, tts_clone_trim=trim, app_language="en"))
    monkeypatch.setattr(srv, "STT", stt or _FakeSTT())
    monkeypatch.setattr(srv, "VOICES", voices or _FakeRegVoices())


def test_clone_commit_trims_edge_fragment_before_stt(monkeypatch):
    stt, voices = _FakeSTT(), _FakeRegVoices()
    _wire(monkeypatch, stt=stt, voices=voices)
    buf = _f32([_tone(0.4), _sil(0.6), _tone(3.0), _sil(0.5)])  # half word at start
    ack = asyncio.run(vc.clone_commit(buf, "Me"))
    assert ack["ok"] is True and ack["refText"] == "hallo welt"
    # STT saw the CLEANED buffer (fragment + gap removed), not the raw recording
    assert len(stt.buffers[0]) < len(buf)
    assert voices.calls[0]["ref_text"] == "hallo welt"
    assert voices.calls[0]["data"][:4] == b"RIFF"


def test_clone_commit_rejects_edge_only_speech(monkeypatch):
    voices = _FakeRegVoices()
    _wire(monkeypatch, voices=voices)
    buf = _f32([_tone(1.0), _sil(0.5), _tone(1.0)])  # speech touches both edges
    ack = asyncio.run(vc.clone_commit(buf, "Me"))
    assert ack == {"ok": False, "error": "edge_speech"}
    assert voices.calls == []


def test_clone_commit_trim_disabled_passes_raw_buffer(monkeypatch):
    stt = _FakeSTT()
    _wire(monkeypatch, trim=False, stt=stt)
    buf = _f32([_tone(0.4), _sil(0.6), _tone(3.0), _sil(0.5)])
    ack = asyncio.run(vc.clone_commit(buf, "Me"))
    assert ack["ok"] is True
    assert len(stt.buffers[0]) == len(buf)


def test_clone_commit_too_short_raw(monkeypatch):
    _wire(monkeypatch)
    ack = asyncio.run(vc.clone_commit(_f32([_sil(0.5)]), "Me"))
    assert ack == {"ok": False, "error": "too_short"}


# --- AI-Voice source selection (merged module) -------------------------------
class _FakeLocalTts:
    """Duck-typed local backend: only `set_voice_mode` marks it switchable."""
    def __init__(self):
        self.calls = []

    async def set_voice_mode(self, mode, *, instruct=None, ref_audio=None, ref_text=None):
        self.calls.append((mode, instruct))


def _src(monkeypatch, *, want="auto", library=False, local=False):
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(
        tts_clone_enabled=True, app_language="en", ai_voice_source=want))
    monkeypatch.setattr(srv, "VOICES", _FakeVoices("clone-1") if library else None)
    monkeypatch.setattr(srv, "TTS", _FakeLocalTts() if local else object())
    return vc.source()


def test_source_auto_prefers_wrapper_over_local(monkeypatch):
    assert _src(monkeypatch, library=True, local=True) == "wrapper"


def test_source_auto_falls_back_to_local(monkeypatch):
    assert _src(monkeypatch, library=False, local=True) == "local"


def test_source_auto_none_when_nothing_wired(monkeypatch):
    assert _src(monkeypatch, library=False, local=False) is None


def test_pinned_source_does_not_silently_fall_back(monkeypatch):
    """A pinned source that isn't wired disables the feature — falling back
    would hide the misconfiguration behind a working-looking UI."""
    assert _src(monkeypatch, want="wrapper", library=False, local=True) is None
    assert _src(monkeypatch, want="local", library=True, local=False) is None


def test_library_active_only_for_wrapper_source(monkeypatch):
    _src(monkeypatch, want="local", library=True, local=True)
    assert vc.library_active() is False       # local pinned → no HTTP library
    assert vc.active_voice_id() is None


def test_state_block_reports_local_source(monkeypatch, tmp_path):
    _src(monkeypatch, want="local", library=False, local=True)
    monkeypatch.setattr(srv.CFG, "omnivoice_mode", "design", raising=False)
    monkeypatch.setattr(srv.CFG, "omnivoice_instruct", "warm voice", raising=False)
    monkeypatch.setattr(vc, "_store", vc.LocalVoiceStore(tmp_path))
    vc._state = None                           # re-read from the stub CFG
    block = asyncio.run(vc.state_block())
    vc._state = None
    assert block["available"] is True and block["source"] == "local"
    assert block["canDesign"] is True and block["canManageSamples"] is True
    assert block["mode"] == "design" and block["instruct"] == "warm voice"


def test_state_block_field_set_is_stable(monkeypatch):
    """The client copies this block field by field (applyAiVoiceState in
    index.html, guarded by tests/client/pure_modules.test.mjs). A new key here
    must be added there too, or it silently never reaches the UI — which is
    exactly how the voice list once went missing."""
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(
        tts_clone_enabled=False, app_language="en", ai_voice_source="auto",
        omnivoice_mode="clone", omnivoice_instruct=""))
    monkeypatch.setattr(srv, "VOICES", None)
    monkeypatch.setattr(srv, "TTS", object())
    vc._state = None
    block = asyncio.run(vc.state_block())
    vc._state = None
    assert set(block) == {
        "available", "source", "canDesign", "canClone", "canManageSamples",
        "mode", "instruct", "active", "voices",
    }


def test_create_voice_adds_and_selects_it(monkeypatch, tmp_path):
    """'New voice' makes an EMPTY voice active so the next recording lands in
    it — and leaves the backend on its current reference until a sample exists,
    so the assistant does not lose its voice in between."""
    tts = _FakeLocalTts()
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(
        tts_clone_enabled=False, app_language="en", ai_voice_source="local",
        omnivoice_mode="clone", omnivoice_instruct="", omnivoice_ref_audio=None,
        omnivoice_ref_text=None))
    monkeypatch.setattr(srv, "VOICES", None)
    monkeypatch.setattr(srv, "TTS", tts)
    monkeypatch.setattr(vc, "_store", vc.LocalVoiceStore(tmp_path))
    monkeypatch.setattr(vc, "_state", {"mode": "clone", "instruct": ""})

    voice = asyncio.run(vc.create_voice("Neue"))
    assert voice["name"] == "Neue" and voice["samples"] == []
    assert vc.store().get_active() == voice["id"]
    # applied in clone mode, but WITHOUT a reference (no sample yet)
    mode, _instruct = tts.calls[-1]
    assert mode == "clone"
    assert vc.store().reference() is None
    vc._state = None


def test_create_voice_is_local_only(monkeypatch):
    monkeypatch.setattr(srv, "CFG", types.SimpleNamespace(
        tts_clone_enabled=True, app_language="en", ai_voice_source="wrapper"))
    monkeypatch.setattr(srv, "VOICES", _FakeVoices("clone-1"))
    monkeypatch.setattr(srv, "TTS", object())
    assert asyncio.run(vc.create_voice("Neue")) is None
