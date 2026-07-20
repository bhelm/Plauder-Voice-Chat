"""Config: .env loading, defaults, legacy fallbacks, validation, backend combos."""
import os
import tempfile
from pathlib import Path

import pytest

from plauder import config
from plauder.config import Config, ConfigError, load_dotenv


def _cfg(**env):
    """Builds a Config from an explicit ENV dict (isolated from os.environ)."""
    saved = dict(os.environ)
    try:
        # Set only the passed keys + the mandatory defaults.
        for k in list(os.environ):
            if k.startswith(("STT_", "TTS_", "LLM_", "WHISPER_", "OMNIVOICE_",
                             "OPENAI_", "FIREWORKS_", "OPENCLAW_", "HOUSE_",
                             "DEBOUNCE_", "AGENT_", "HOST", "PORT",
                             "APP_", "WAKE_", "HERMES_", "SPEAKER_",
                             "BASE_PATH", "STREAMING", "SYSTEM_PROMPT",
                             "SOUL_PATH", "VOICE_")):
                del os.environ[k]
        os.environ.update(env)
        return Config.from_env()
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_dotenv_loader_populates_environ():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / ".env"
        p.write_text('UNIT_TEST_KEY=hallo123\nexport QUOTED="mit space"\n')
        os.environ.pop("UNIT_TEST_KEY", None)
        os.environ.pop("QUOTED", None)
        load_dotenv(p)
        assert os.environ["UNIT_TEST_KEY"] == "hallo123"
        assert os.environ["QUOTED"] == "mit space"
        os.environ.pop("UNIT_TEST_KEY", None)
        os.environ.pop("QUOTED", None)


def test_defaults_when_minimal_env():
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.stt_backend == "openai"
    assert cfg.tts_backend == "openai"
    assert cfg.llm_backend == "openai_compat"
    assert cfg.port == 8319
    assert cfg.agent_name == "Antonia"
    assert cfg.tts_openai_voice == "nova"


def test_legacy_fallback_keys():
    """Old .env (OPENAI_API_KEY + FIREWORKS_*) fills the new fields."""
    cfg = _cfg(
        OPENAI_API_KEY="sk-legacy",
        FIREWORKS_API_KEY="fw-legacy",
        FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1",
        FIREWORKS_MODEL="accounts/fireworks/models/glm-5p2",
        WHISPER_LANGUAGE="de",
    )
    assert cfg.stt_openai_api_key == "sk-legacy"
    assert cfg.tts_openai_api_key == "sk-legacy"
    assert cfg.llm_api_key == "fw-legacy"
    assert cfg.llm_base_url == "https://api.fireworks.ai/inference/v1"
    assert cfg.llm_model.startswith("accounts/fireworks/models/")
    assert cfg.stt_language == "de"


def test_stt_partial_defaults_off_for_openai():
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.stt_partial is False


def test_stt_partial_defaults_on_for_whisper_local():
    cfg = _cfg(STT_BACKEND="whisper_local", OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.stt_partial is True


def test_stt_partial_explicit_override():
    cfg = _cfg(STT_BACKEND="whisper_local", STT_PARTIAL="0",
               OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.stt_partial is False


def test_streaming_defaults_on():
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.streaming is True
    assert cfg.tts_chunk_ms == 400


def test_wake_word_defaults_off_and_follows_agent_name():
    # Wake word is a selectable input mode; start default is OFF.
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.wake_word_enabled is False
    assert cfg.wake_word == "antonia"            # = AGENT_NAME (default) lowercased


def test_wake_word_start_default_via_env():
    cfg = _cfg(WAKE_WORD_ENABLED="1", OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.wake_word_enabled is True


def test_wake_word_follows_custom_agent_name():
    cfg = _cfg(AGENT_NAME="Xena", OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.wake_word == "xena"


def test_wake_word_explicit_override_wins():
    cfg = _cfg(AGENT_NAME="Xena", WAKE_WORD="computer",
               OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.wake_word == "computer"


def test_wake_word_can_be_disabled():
    cfg = _cfg(WAKE_WORD_ENABLED="0", OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.wake_word_enabled is False


def test_new_keys_win_over_legacy():
    cfg = _cfg(
        OPENAI_API_KEY="sk-legacy", STT_OPENAI_API_KEY="sk-new",
        FIREWORKS_API_KEY="fw-legacy", LLM_API_KEY="fw-new",
        LLM_MODEL="accounts/x/models/y",
    )
    assert cfg.stt_openai_api_key == "sk-new"
    assert cfg.llm_api_key == "fw-new"
    assert cfg.llm_model == "accounts/x/models/y"


def test_validate_ok_for_cloud():
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    cfg.validate()  # must not raise


def test_validate_missing_openai_key():
    cfg = _cfg(STT_BACKEND="openai", FIREWORKS_API_KEY="y")
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validate_missing_llm_key():
    cfg = _cfg(OPENAI_API_KEY="x", LLM_BACKEND="openai_compat")
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validate_unknown_backend():
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y", STT_BACKEND="nonsense")
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validate_local_backends_combo():
    """Local backends need NO cloud keys to validate."""
    cfg = _cfg(
        STT_BACKEND="whisper_local",
        TTS_BACKEND="omnivoice_local", OMNIVOICE_MODE="instruct",
        LLM_BACKEND="openclaw", OPENCLAW_GATEWAY_TOKEN="tok",
    )
    cfg.validate()  # must not raise


def test_validate_omnivoice_clone_requires_ref_audio():
    cfg = _cfg(TTS_BACKEND="omnivoice_local", OMNIVOICE_MODE="clone",
               OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    with pytest.raises(ConfigError):
        cfg.validate()


def test_house_mode_subflags_follow_master():
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y", HOUSE_MODE="1")
    assert cfg.house_mode is True
    assert cfg.house_speaker_id is True
    assert cfg.house_wake_word is True


def test_resolved_voice_system_default_is_voice_hint_only():
    # Default: NO persona, only the terse voice hint.
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    sys_prompt = cfg.resolved_voice_system()
    assert "VOICE MODE" in sys_prompt          # English default (APP_LANGUAGE=en)
    assert "Du bist" not in sys_prompt          # no built-in identity


def test_resolved_voice_system_prepends_configured_persona():
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y",
               SYSTEM_PROMPT="Du bist Klaus, ein knurriger Butler.")
    sys_prompt = cfg.resolved_voice_system()
    assert sys_prompt.startswith("Du bist Klaus, ein knurriger Butler.")
    assert "VOICE MODE" in sys_prompt          # English default (APP_LANGUAGE=en)


def test_app_language_de_switches_voice_hint_and_stt():
    """APP_LANGUAGE=de uses the German voice hint and makes STT default to 'de'."""
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y", APP_LANGUAGE="de")
    assert cfg.app_language == "de"
    assert cfg.stt_language == "de"
    assert "VOICE-MODE" in cfg.resolved_voice_system()


def test_app_language_defaults_to_en():
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y")
    assert cfg.app_language == "en"
    assert "VOICE MODE" in cfg.resolved_voice_system()


def test_voice_mode_system_full_override():
    cfg = _cfg(OPENAI_API_KEY="x", FIREWORKS_API_KEY="y",
               VOICE_MODE_SYSTEM="NUR DAS HIER.")
    assert cfg.resolved_voice_system() == "NUR DAS HIER."


def test_strip_turn_hint():
    from plauder.config import _DEFAULT_TURN_HINTS, strip_turn_hint

    en = _DEFAULT_TURN_HINTS["en"]
    de = _DEFAULT_TURN_HINTS["de"]
    # Default hints in either language are stripped (gateway history may
    # contain both when APP_LANGUAGE changed between sessions).
    assert strip_turn_hint(f"Hallo Joy\n\n{en}") == "Hallo Joy"
    assert strip_turn_hint(f"Hallo Joy\n\n{de}") == "Hallo Joy"
    # Custom hint via extra + stacked hints.
    custom = "[Custom voice rule]"
    assert strip_turn_hint(f"Hi\n\n{custom}\n\n{en}", (custom,)) == "Hi"
    # Untouched cases.
    assert strip_turn_hint("Just text") == "Just text"
    assert strip_turn_hint("") == ""


def test_real_env_file_loads(tmp_path, monkeypatch):
    """``load_config`` reads a .env file and yields a valid cloud config.

    Hermetic: we point the loader at a temp .env (via VOICE_ENV_FILE) instead of
    the developer's real on-disk .env, so the test doesn't depend on the repo.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "STT_BACKEND=openai\n"
        "TTS_BACKEND=openai\n"
        "LLM_BACKEND=openai_compat\n"
        "OPENAI_API_KEY=sk-test-dummy\n"
        "FIREWORKS_API_KEY=fw-test-dummy\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VOICE_ENV_FILE", str(env_file))
    cfg = config.load_config()
    assert cfg.stt_backend in ("openai", "whisper_local")
    assert cfg.llm_backend in ("openai_compat", "openclaw")


def test_load_dotenv_strips_inline_comments(tmp_path, monkeypatch):
    """Regression: `cp .env.example .env` must work — the template uses inline
    `# comments`, which previously became part of the value (flags like
    `SPEAKER_DEBUG=0   # …` silently flipped to True)."""
    from pathlib import Path
    from plauder.config import load_dotenv
    envf = tmp_path / ".env"
    envf.write_text(
        "PLAUDER_T1=0              # a comment\n"
        "PLAUDER_T2=openai            # openai | whisper_local\n"
        'PLAUDER_T3="quoted # not a comment" # real comment\n'
        "PLAUDER_T4=http://x/#anchor\n",   # '#' not preceded by space → kept
        encoding="utf-8")
    for k in ("PLAUDER_T1", "PLAUDER_T2", "PLAUDER_T3", "PLAUDER_T4"):
        monkeypatch.delenv(k, raising=False)
    load_dotenv(Path(envf))
    import os
    assert os.environ["PLAUDER_T1"] == "0"
    assert os.environ["PLAUDER_T2"] == "openai"
    assert os.environ["PLAUDER_T3"] == "quoted # not a comment"
    assert os.environ["PLAUDER_T4"] == "http://x/#anchor"
    for k in ("PLAUDER_T1", "PLAUDER_T2", "PLAUDER_T3", "PLAUDER_T4"):
        monkeypatch.delenv(k, raising=False)
