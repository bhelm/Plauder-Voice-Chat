"""STT backends (openai + whisper_local), mocked — no API calls, no GPU."""
import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from plauder.backends.stt.base import STTBackend
from plauder.backends.stt.openai_api import OpenAISTTBackend
from plauder.config import SAMPLE_RATE


class _Cfg:
    stt_backend = "openai"
    stt_openai_api_key = "sk-test"
    stt_openai_model = "whisper-1"
    stt_openai_base_url = None
    stt_language = "de"


def _pcm(seconds=0.5):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32).tobytes()


def _openai_with_mock(text):
    eng = OpenAISTTBackend.from_config(_Cfg())
    client = MagicMock()
    resp = MagicMock()
    resp.text = text
    client.audio.transcriptions.create.return_value = resp
    eng._client = client
    return eng, client


def test_factory_selects_openai():
    eng = STTBackend.from_config(_Cfg())
    assert isinstance(eng, OpenAISTTBackend)


def test_openai_transcribe_returns_text():
    eng, client = _openai_with_mock("Hallo Antonia")
    text = asyncio.run(eng.transcribe(_pcm(), SAMPLE_RATE))
    assert text == "Hallo Antonia"
    client.audio.transcriptions.create.assert_called_once()


def test_openai_sends_wav_and_language():
    eng, client = _openai_with_mock("egal")
    eng._transcribe_sync(_pcm(1.0), SAMPLE_RATE)
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["language"] == "de"
    f = kwargs["file"]
    assert f.name.endswith(".wav")
    assert f.getvalue()[:4] == b"RIFF"


def test_openai_strips_whitespace():
    eng, _ = _openai_with_mock("  text mit rand  ")
    assert eng._transcribe_sync(_pcm(), SAMPLE_RATE) == "text mit rand"


def test_openai_no_speech_prob_is_none():
    eng, _ = _openai_with_mock("x")
    eng._transcribe_sync(_pcm(), SAMPLE_RATE)
    assert eng.last_no_speech_prob is None


def test_openai_transcribe_requires_load():
    eng = OpenAISTTBackend.from_config(_Cfg())
    eng._client = None
    with pytest.raises(RuntimeError):
        asyncio.run(eng.transcribe(_pcm(), SAMPLE_RATE))


def test_describe_reports_engine():
    eng, _ = _openai_with_mock("x")
    d = eng.describe()
    assert d["engine"] == "openai-whisper"
    assert d["loaded"] is True


# --- whisper_local (lazy import) --------------------------------------------
def test_whisper_local_not_imported_for_cloud_backend():
    """As long as whisper_local is not active, faster_whisper stays unloaded."""
    assert "faster_whisper" not in sys.modules


def test_whisper_local_load_raises_clear_error_without_dep():
    """Without faster_whisper installed, load() yields a clear error."""
    from plauder.backends.stt.whisper_local import WhisperLocalSTTBackend
    from plauder.backends.base import BackendError

    class _LocalCfg:
        whisper_model = "large-v3-turbo"
        whisper_device = "cuda"
        whisper_compute_type = "int8"
        whisper_beam_size = 5
        stt_language = "de"
        whisper_local_files_only = True

    import importlib.util
    eng = WhisperLocalSTTBackend.from_config(_LocalCfg())
    # Only meaningful if faster_whisper is NOT installable (otherwise the error
    # path is not hit — then it continues to the model download/build).
    if importlib.util.find_spec("faster_whisper") is not None:
        pytest.skip("faster_whisper installiert — Fehlerpfad nicht testbar")
    with pytest.raises(BackendError):
        asyncio.run(eng.load())


def test_whisper_local_transcribe_with_mock_model():
    from plauder.backends.stt.whisper_local import WhisperLocalSTTBackend

    class _LocalCfg:
        whisper_model = "x"; whisper_device = "cpu"; whisper_compute_type = "int8"
        whisper_beam_size = 1; stt_language = "de"; whisper_local_files_only = True

    eng = WhisperLocalSTTBackend.from_config(_LocalCfg())
    seg = MagicMock()
    seg.text = "lokaler text"
    seg.no_speech_prob = 0.2
    model = MagicMock()
    model.transcribe.return_value = ([seg], MagicMock())
    eng._model = model
    text = asyncio.run(eng.transcribe(_pcm(), SAMPLE_RATE))
    assert text == "lokaler text"
    assert eng.last_no_speech_prob == 0.2
