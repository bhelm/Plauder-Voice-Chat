"""TTS-Backends (openai + omnivoice_local), gemockt — keine API-Calls, keine GPU."""
import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from plauder.backends.tts.base import TTSBackend
from plauder.backends.tts.openai_api import OpenAITTSBackend


class _Cfg:
    tts_backend = "openai"
    tts_openai_api_key = "sk-test"
    tts_openai_model = "tts-1"
    tts_openai_voice = "nova"
    tts_openai_base_url = None
    tts_sentence_split = False
    tts_max_chars_per_chunk = 220
    tts_sentence_gap_ms = 120


def _openai_with_mock(pcm_int16):
    eng = OpenAITTSBackend.from_config(_Cfg())
    pcm_bytes = np.asarray(pcm_int16, dtype=np.int16).tobytes()
    resp = MagicMock()
    resp.read.return_value = pcm_bytes
    client = MagicMock()
    client.audio.speech.create.return_value = resp
    eng._client = client
    return eng, client


def test_factory_selects_openai():
    assert isinstance(TTSBackend.from_config(_Cfg()), OpenAITTSBackend)


def test_synth_returns_pcm_bytes_and_rate():
    eng, _ = _openai_with_mock([0, 16384, -16384, 32767])
    pcm, sr = asyncio.run(eng.synth("Hallo", speed=1.0))
    assert sr == eng.sample_rate == 24000
    # 4 int16-Samples -> 8 Bytes
    assert len(pcm) == 8
    assert isinstance(pcm, (bytes, bytearray))


def test_synth_roundtrip_values():
    eng, _ = _openai_with_mock([0, 16384, -16384])
    pcm, _ = asyncio.run(eng.synth("x", speed=1.0))
    out = np.frombuffer(pcm, dtype=np.int16)
    # Dekodieren (/32768) + Re-Enkodieren (*32767) ist bewusst minimal verlustig.
    assert abs(int(out[1]) - 16384) <= 1


def test_passes_voice_and_format():
    eng, client = _openai_with_mock([1, 2, 3])
    eng._synth_sync("Text", 1.0)
    kwargs = client.audio.speech.create.call_args.kwargs
    assert kwargs["model"] == "tts-1"
    assert kwargs["voice"] == "nova"
    assert kwargs["input"] == "Text"
    assert kwargs["response_format"] == "pcm"


def test_speed_clamped():
    assert OpenAITTSBackend._clamp_speed(10.0) == 4.0
    assert OpenAITTSBackend._clamp_speed(0.1) == 0.25
    assert OpenAITTSBackend._clamp_speed("nonsense") == 1.0
    assert OpenAITTSBackend._clamp_speed(1.5) == 1.5


def test_synth_requires_load():
    eng = OpenAITTSBackend.from_config(_Cfg())
    eng._client = None
    with pytest.raises(RuntimeError):
        asyncio.run(eng.synth("x", speed=1.0))


def test_describe():
    eng, _ = _openai_with_mock([0])
    d = eng.describe()
    assert d["engine"] == "openai-tts"
    assert d["voice"] == "nova"


# --- omnivoice_local (lazy) -------------------------------------------------
def test_omnivoice_not_imported_for_cloud_backend():
    assert "omnivoice" not in sys.modules
    assert "torch" not in sys.modules


def test_omnivoice_load_raises_clear_error_without_dep():
    from plauder.backends.tts.omnivoice_local import OmniVoiceLocalTTSBackend
    from plauder.backends.base import BackendError

    class _LocalCfg:
        omnivoice_model = "k2-fsa/OmniVoice"; omnivoice_device = "cuda"
        omnivoice_mode = "clone"; omnivoice_ref_audio = "/x.wav"
        omnivoice_ref_text = None; omnivoice_language = "de"
        tts_sentence_split = False; tts_max_chars_per_chunk = 220; tts_sentence_gap_ms = 120

    eng = OmniVoiceLocalTTSBackend.from_config(_LocalCfg())
    if "omnivoice" in sys.modules:
        pytest.skip("omnivoice installiert — Fehlerpfad nicht testbar")
    with pytest.raises(BackendError):
        asyncio.run(eng.load())


def test_omnivoice_synth_with_mock_engine():
    from plauder.backends.tts.omnivoice_local import OmniVoiceLocalTTSBackend

    class _LocalCfg:
        omnivoice_model = "x"; omnivoice_device = "cpu"; omnivoice_mode = "instruct"
        omnivoice_ref_audio = None; omnivoice_ref_text = None; omnivoice_language = "de"
        tts_sentence_split = False; tts_max_chars_per_chunk = 220; tts_sentence_gap_ms = 120

    eng = OmniVoiceLocalTTSBackend.from_config(_LocalCfg())
    fake = MagicMock()
    fake.generate.return_value = (np.array([0.0, 0.5, -0.5], dtype=np.float32), 24000)
    eng._tts = fake
    pcm, sr = asyncio.run(eng.synth("Hallo", speed=1.0))
    assert sr == 24000
    assert len(pcm) == 6  # 3 int16
