"""Configuration: .env loading, Config dataclass and validation.

A single source of truth for all runtime options. The values are read from the
environment at startup (os.environ, filled from the .env) and packed into a
frozen ``Config`` object that is passed through the server.

Backend selection happens via ``STT_BACKEND`` / ``TTS_BACKEND`` / ``LLM_BACKEND``.
So that the *existing* .env (OpenAI + Fireworks, without new backend variables)
keeps working, there are fallback chains onto the old variable names
(``OPENAI_API_KEY``, ``FIREWORKS_*`` etc.).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# OpenAI TTS returns raw PCM at a fixed sample rate. Default 24 kHz (OpenAI Cloud,
# Kokoro); overridable via TTS_OPENAI_SAMPLE_RATE (e.g. 16000 for Piper-kerstin,
# 22050 for Piper-thorsten) — otherwise playback sounds too fast/slow.
OPENAI_TTS_SAMPLE_RATE = 24000
# Browser delivers/expects 16 kHz mono.
SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# .env loader (not a required dependency)
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path) -> None:
    """Minimal .env parser. Already-set real ENV variables win
    (override=False), so that tests and shell exports take precedence.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (len(val) >= 2) and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


# --------------------------------------------------------------------------- #
# ENV helpers
# --------------------------------------------------------------------------- #
def _first(*names_or_values: str | None, default: str = "") -> str:
    """First non-empty variant. Arguments are already-resolved values."""
    for v in names_or_values:
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


# Voice-mode hint appended to the system prompt, per language. Keeps the
# assistant terse and speech-friendly (no markdown/emoji, since everything is
# read aloud). The "<lang> as default" line nudges the spoken response language.
# The active language is chosen via APP_LANGUAGE (see Config.app_language).
_DEFAULT_VOICE_HINTS = {
    "en": (
        "\n\n---\n"
        "VOICE MODE (important for this speech session):\n"
        "• You are currently talking via voice. Your answers are read aloud by a "
        "text-to-speech system.\n"
        "• NO emojis – they would be read out.\n"
        "• NO markdown, no bullet points, no code blocks, do not read out URLs "
        "– describe them briefly in words.\n"
        "• Short, natural, spoken sentences. English as default.\n"
        "• Be as concise as possible."
    ),
    "de": (
        "\n\n---\n"
        "VOICE-MODE (wichtig für diese Sprach-Session):\n"
        "• Du sprichst gerade per Sprache. Deine Antworten werden von einem "
        "Text-to-Speech-System vorgelesen.\n"
        "• KEINE Emojis – sie würden vorgelesen.\n"
        "• KEIN Markdown, keine Aufzählungszeichen, keine Code-Blöcke, keine URLs "
        "vorlesen – beschreibe sie kurz in Worten.\n"
        "• Kurze, natürliche, gesprochene Sätze. Deutsch als Default.\n"
        "• Halte dich so knapp wie möglich."
    ),
}

# UI / app languages that ship with a full translation.
SUPPORTED_LANGUAGES = ("en", "de")


def _norm_lang(value: str) -> str:
    """Normalize an APP_LANGUAGE value to a supported code ('en'/'de')."""
    code = (value or "").strip().lower()[:2]
    return code if code in SUPPORTED_LANGUAGES else "en"


VALID_STT_BACKENDS = ("openai", "whisper_local")
VALID_TTS_BACKENDS = ("openai", "omnivoice_local")
VALID_LLM_BACKENDS = ("openai_compat", "openclaw")


class ConfigError(RuntimeError):
    """Configuration is invalid (missing keys, unknown backend …)."""


@dataclass(frozen=True)
class Config:
    # --- General ---
    host: str = "127.0.0.1"
    port: int = 8319
    agent_name: str = "Antonia"
    soul_path: str = ""
    log_level: str = "INFO"
    # App / UI language ('en'/'de'). Drives the UI i18n locale handed to the
    # browser and the assistant's default spoken language (see _DEFAULT_VOICE_HINTS).
    app_language: str = "en"

    # --- Backend selection ---
    stt_backend: str = "openai"
    tts_backend: str = "openai"
    llm_backend: str = "openai_compat"

    # --- STT: OpenAI ---
    stt_openai_api_key: str = ""
    stt_openai_model: str = "whisper-1"
    stt_openai_base_url: str | None = None
    stt_language: str | None = "de"

    # --- STT: faster_whisper (local) ---
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_local_files_only: bool = True

    # --- STT Partial / Streaming (B2) ---
    # While a segment is still being streamed in (B1), periodically transcribe
    # the audio buffer accumulated so far and send intermediate results
    # (transcript.partial). Useful mainly with local Whisper (cheap to repeat).
    stt_partial: bool = False
    stt_partial_min_interval_ms: int = 700   # minimum gap between partials
    stt_partial_min_new_ms: int = 500        # minimum new audio before the next partial

    # --- STT hallucination filter ---
    stt_hallucination_filter: bool = True
    stt_ghost_no_speech_prob: float = 0.6
    stt_ghost_use_duration: bool = False
    stt_ghost_max_dur_s: float = 1.5
    stt_ghost_extra_phrases: str = ""

    # --- TTS: OpenAI ---
    tts_openai_api_key: str = ""
    tts_openai_model: str = "tts-1"
    tts_openai_voice: str = "nova"
    tts_openai_format: str = "pcm"
    tts_openai_base_url: str | None = None
    tts_openai_sample_rate: int = OPENAI_TTS_SAMPLE_RATE
    # local_speed=True: apply TTS_SPEED / UI slider locally via time-stretch
    # (pitch-preserving) instead of sending `speed` to the server — needed for
    # servers that ignore the OpenAI `speed` parameter (e.g. local XTTS).
    tts_openai_local_speed: bool = False
    tts_speed: float = 1.0

    # --- TTS: OmniVoice (local) ---
    omnivoice_model: str = "k2-fsa/OmniVoice"
    omnivoice_device: str = "cuda"
    omnivoice_mode: str = "clone"
    omnivoice_ref_audio: str | None = None
    omnivoice_ref_text: str | None = None
    omnivoice_language: str | None = None

    # --- LLM: OpenAI-compatible (Fireworks/OpenAI/…) ---
    llm_api_key: str = ""
    llm_base_url: str = "https://api.fireworks.ai/inference/v1"
    llm_model: str = "accounts/fireworks/models/glm-5p2"
    llm_max_tokens: int = 4096
    llm_timeout: int = 300
    llm_history_turns: int = 20
    llm_retry_timeout_on_idle: bool = True

    # --- LLM: OpenClaw (legacy) ---
    openclaw_gateway_url: str = "http://localhost:8080"
    openclaw_gateway_token: str = ""
    openclaw_agent_id: str = "antonia"
    openclaw_user_id: str = "voice-user"

    # --- Turn-taking ---
    debounce_ms: int = 1200
    debounce_ms_min: int = 200
    debounce_ms_max: int = 5000
    tts_sentence_split: bool = False
    tts_sentence_gap_ms: int = 120
    tts_max_chars_per_chunk: int = 220

    # --- Wake word (STT prefix gate) ---
    # Wake word is an input mode (alongside VAD & push-to-talk) chosen in the UI.
    # In wake mode only segments whose transcript starts with the wake word
    # (filler words before it allowed) trigger a turn — the rest is discarded.
    # After a reply the gate stays open for wake_word_window_s (follow-up
    # questions without repeating the wake word).
    # `wake_word_enabled` is now only the START default: WAKE_WORD_ENABLED=1
    # makes the UI start in wake mode. Default OFF → start in VAD mode.
    wake_word_enabled: bool = False
    wake_word: str = "antonia"          # empty in the ENV → follows AGENT_NAME
    # Behavior AFTER a reply:
    #   "conversation" = conversation window stays open for wake_word_window_s,
    #                    follow-up questions without the wake word possible (default).
    #   "alexa"        = one-shot: after the reply the window closes immediately,
    #                    every new command needs the wake word again.
    wake_mode: str = "conversation"
    wake_word_window_s: float = 8.0     # conversation window after a reply
    wake_word_fuzzy: bool = True        # tolerate Whisper mishearings
    wake_word_anywhere: bool = False    # True = wake word anywhere in the sentence instead of at the start
    wake_word_strip: bool = True        # strip the wake word from the text before the LLM
    wake_word_ratio: float = 0.78       # fuzzy threshold (higher = stricter)

    # --- Streaming (latency) ---
    # streaming=True: LLM tokens are streamed, handed to TTS sentence by sentence
    # immediately and sent progressively to the client as PCM chunks (A1+A2). When
    # False the classic "generate fully, then one WAV" path applies (fallback).
    streaming: bool = True
    # Target chunk size for the progressive audio transfer (ms per VCT2 frame).
    tts_chunk_ms: int = 400

    # --- House Mode ---
    house_mode: bool = False
    house_data_dir: str = ""
    house_speaker_id: bool = False
    house_wake_word: bool = False
    house_auth: bool = False

    # --- Voice persona (optional) ---
    # Optional persona prepended to the terse voice hint. Empty = no persona
    # (only "be concise, no markdown/emoji"). Source: SOUL.md file (soul_path)
    # > system_prompt. Full prompt override: voice_mode_system.
    system_prompt: str = ""
    voice_mode_system: str = ""

    # --- Warmups ---
    stt_warmup: bool = False
    tts_warmup: bool = False

    # --- Pronunciations ---
    pronunciations_file: str = ""

    @property
    def soul_persona(self) -> str:
        """Optional persona: SOUL.md file > system_prompt (Config/.env) > empty.
        By default NO persona — then only the terse voice hint remains
        (see resolved_voice_system). Not cached — cheap."""
        if self.soul_path:
            try:
                text = Path(self.soul_path).read_text(encoding="utf-8").strip()
                if text:
                    return text
            except OSError as exc:
                print(f"⚠️  Persona file not loadable ({self.soul_path}): {exc} "
                      "— using system_prompt/no persona.", file=sys.stderr)
        return self.system_prompt.strip()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> "Config":
        """Builds the Config from os.environ incl. legacy fallback chains."""
        # Persona file optional; NO personal default path. Empty = persona
        # comes from SYSTEM_PROMPT or the generic default (see soul_persona).
        soul_path = _env("SOUL_PATH")

        # Voice-mode system: fully overridable, otherwise persona + hint.
        # (Persona is read lazily via .soul_persona; here only the hint part
        #  or a full override.)
        voice_mode_system = _env("VOICE_MODE_SYSTEM")  # empty = built later from persona+hint

        # App/UI language; also the default for STT (overridable via STT_LANGUAGE).
        app_language = _norm_lang(_first(_env("APP_LANGUAGE"), _env("APP_LANG"), default="en"))

        house_mode = env_flag("HOUSE_MODE", False)
        house_data_dir = _first(
            _env("HOUSE_DATA_DIR"),
            default=str(Path(__file__).resolve().parent.parent / "house_data"),
        )

        cfg = cls(
            host=_first(_env("HOST"), default="127.0.0.1"),
            port=_env_int("PORT", 8319),
            agent_name=_first(_env("AGENT_NAME"), default="Antonia"),
            soul_path=soul_path,
            system_prompt=_env("SYSTEM_PROMPT"),
            log_level=_first(_env("LOG_LEVEL"), default="INFO").upper(),
            app_language=app_language,

            stt_backend=_first(_env("STT_BACKEND"), default="openai").lower(),
            tts_backend=_first(_env("TTS_BACKEND"), default="openai").lower(),
            llm_backend=_first(_env("LLM_BACKEND"), default="openai_compat").lower(),

            # STT openai (fallback: OPENAI_API_KEY)
            stt_openai_api_key=_first(_env("STT_OPENAI_API_KEY"), _env("OPENAI_API_KEY")),
            stt_openai_model=_first(_env("STT_OPENAI_MODEL"), default="whisper-1"),
            stt_openai_base_url=(_first(_env("STT_OPENAI_BASE_URL"), _env("OPENAI_BASE_URL")) or None),
            stt_language=(_first(_env("STT_LANGUAGE"), _env("WHISPER_LANGUAGE"), default=app_language) or None),

            # STT local whisper
            whisper_model=_first(_env("WHISPER_MODEL"), default="large-v3-turbo"),
            whisper_device=_first(_env("WHISPER_DEVICE"), default="cuda"),
            whisper_compute_type=_first(_env("WHISPER_COMPUTE_TYPE"), default="int8"),
            whisper_beam_size=_env_int("WHISPER_BEAM_SIZE", 5),
            whisper_local_files_only=env_flag("WHISPER_LOCAL_FILES_ONLY", True),

            # Partials by default only with local Whisper (the API backend would
            # otherwise be called multiple times per second at a cost).
            stt_partial=env_flag("STT_PARTIAL",
                                 _first(_env("STT_BACKEND"), default="openai").lower() == "whisper_local"),
            stt_partial_min_interval_ms=_env_int("STT_PARTIAL_MIN_INTERVAL_MS", 700),
            stt_partial_min_new_ms=_env_int("STT_PARTIAL_MIN_NEW_MS", 500),

            stt_hallucination_filter=env_flag("STT_HALLUCINATION_FILTER", True),
            stt_ghost_no_speech_prob=_env_float("STT_GHOST_NO_SPEECH_PROB", 0.6),
            stt_ghost_use_duration=env_flag("STT_GHOST_USE_DURATION", False),
            stt_ghost_max_dur_s=_env_float("STT_GHOST_MAX_DUR_S", 1.5),
            stt_ghost_extra_phrases=_env("STT_GHOST_EXTRA_PHRASES", ""),

            # TTS openai (fallback: OPENAI_API_KEY, OPENAI_TTS_*)
            tts_openai_api_key=_first(_env("TTS_OPENAI_API_KEY"), _env("OPENAI_API_KEY")),
            tts_openai_model=_first(_env("TTS_OPENAI_MODEL"), _env("OPENAI_TTS_MODEL"), default="tts-1"),
            tts_openai_voice=_first(_env("TTS_OPENAI_VOICE"), _env("OPENAI_TTS_VOICE"), default="nova"),
            tts_openai_format=_first(_env("TTS_OPENAI_FORMAT"), default="pcm").lower(),
            tts_openai_base_url=(_first(_env("TTS_OPENAI_BASE_URL"), _env("OPENAI_BASE_URL")) or None),
            tts_openai_sample_rate=int(_first(_env("TTS_OPENAI_SAMPLE_RATE"), default=str(OPENAI_TTS_SAMPLE_RATE))),
            tts_openai_local_speed=env_flag("TTS_OPENAI_LOCAL_SPEED", False),
            tts_speed=_env_float("TTS_SPEED", 1.0),

            # TTS omnivoice
            omnivoice_model=_first(_env("OMNIVOICE_MODEL"), default="k2-fsa/OmniVoice"),
            omnivoice_device=_first(_env("OMNIVOICE_DEVICE"), default="cuda"),
            omnivoice_mode=_first(_env("OMNIVOICE_MODE"), default="clone"),
            omnivoice_ref_audio=(_env("OMNIVOICE_REF_AUDIO") or None),
            omnivoice_ref_text=(_env("OMNIVOICE_REF_TEXT") or None),
            omnivoice_language=(_first(_env("OMNIVOICE_LANGUAGE"), default=app_language) or None),

            # LLM openai_compat (fallback: FIREWORKS_*, OPENCLAW_GATEWAY_TOKEN)
            llm_api_key=_first(_env("LLM_API_KEY"), _env("FIREWORKS_API_KEY"),
                               _env("OPENCLAW_GATEWAY_TOKEN")),
            llm_base_url=_first(_env("LLM_BASE_URL"), _env("FIREWORKS_BASE_URL"),
                                default="https://api.fireworks.ai/inference/v1").rstrip("/"),
            llm_model=_first(_env("LLM_MODEL"), _env("FIREWORKS_MODEL"),
                             default="accounts/fireworks/models/glm-5p2"),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", _env_int("OPENCLAW_MAX_TOKENS", 4096)),
            llm_timeout=_env_int("LLM_TIMEOUT", _env_int("OPENCLAW_TIMEOUT", 300)),
            llm_history_turns=_env_int("LLM_HISTORY_TURNS", 20),
            llm_retry_timeout_on_idle=env_flag("OPENCLAW_RETRY_TIMEOUT", True),

            # LLM openclaw (legacy)
            openclaw_gateway_url=_first(_env("OPENCLAW_GATEWAY_URL"),
                                        default="http://localhost:8080").rstrip("/"),
            openclaw_gateway_token=_first(_env("OPENCLAW_GATEWAY_TOKEN"), _env("FIREWORKS_API_KEY")),
            openclaw_agent_id=_first(_env("OPENCLAW_AGENT_ID"), default="antonia"),
            openclaw_user_id=_first(_env("OPENCLAW_USER_ID"), default="voice-user"),

            debounce_ms=_env_int("DEBOUNCE_MS", 1200),
            tts_sentence_split=env_flag("TTS_SENTENCE_SPLIT", False),
            tts_sentence_gap_ms=_env_int("TTS_SENTENCE_GAP_MS", 120),
            tts_max_chars_per_chunk=_env_int("TTS_MAX_CHARS_PER_CHUNK", 220),

            streaming=env_flag("STREAMING", True),
            tts_chunk_ms=_env_int("TTS_CHUNK_MS", 400),

            wake_word_enabled=env_flag("WAKE_WORD_ENABLED", False),
            # Default = AGENT_NAME (lowercased), overridable via WAKE_WORD.
            wake_word=_first(_env("WAKE_WORD"),
                             default=_first(_env("AGENT_NAME"), default="Antonia").lower()),
            wake_mode=_first(_env("WAKE_MODE"), default="conversation").strip().lower(),
            wake_word_window_s=_env_float("WAKE_WORD_WINDOW_S", 8.0),
            wake_word_fuzzy=env_flag("WAKE_WORD_FUZZY", True),
            wake_word_anywhere=env_flag("WAKE_WORD_ANYWHERE", False),
            wake_word_strip=env_flag("WAKE_WORD_STRIP", True),
            wake_word_ratio=_env_float("WAKE_WORD_RATIO", 0.78),

            house_mode=house_mode,
            house_data_dir=house_data_dir,
            house_speaker_id=env_flag("HOUSE_SPEAKER_ID", house_mode),
            house_wake_word=env_flag("HOUSE_WAKE_WORD", house_mode),
            house_auth=env_flag("HOUSE_AUTH", house_mode),

            voice_mode_system=voice_mode_system,

            stt_warmup=env_flag("STT_WARMUP", False),
            tts_warmup=env_flag("TTS_WARMUP", False),

            pronunciations_file=_first(
                _env("VOICE_PRONUNCIATIONS"),
                default=str(Path(__file__).resolve().parent.parent / "pronunciations.json"),
            ),
        )
        return cfg

    # ------------------------------------------------------------------ #
    def resolved_voice_system(self) -> str:
        """System prompt for the voice session.

        Default: only a terse voice hint (be concise, no markdown/emoji — it is
        read aloud). An optional persona (system_prompt / SOUL.md) is prepended
        if set; otherwise no persona. Full override of the entire prompt via
        VOICE_MODE_SYSTEM.
        """
        if self.voice_mode_system:
            return self.voice_mode_system
        default_hint = _DEFAULT_VOICE_HINTS.get(self.app_language, _DEFAULT_VOICE_HINTS["en"])
        hint = os.environ.get("VOICE_MODE_HINT", default_hint)
        return (self.soul_persona + hint).strip()

    def validate(self) -> None:
        """Raises ConfigError on an obviously unusable configuration.
        Checks ONLY the respective active backend — inactive backends need no
        keys/deps."""
        if self.stt_backend not in VALID_STT_BACKENDS:
            raise ConfigError(
                f"STT_BACKEND={self.stt_backend!r} unknown "
                f"(allowed: {', '.join(VALID_STT_BACKENDS)})")
        if self.tts_backend not in VALID_TTS_BACKENDS:
            raise ConfigError(
                f"TTS_BACKEND={self.tts_backend!r} unknown "
                f"(allowed: {', '.join(VALID_TTS_BACKENDS)})")
        if self.llm_backend not in VALID_LLM_BACKENDS:
            raise ConfigError(
                f"LLM_BACKEND={self.llm_backend!r} unknown "
                f"(allowed: {', '.join(VALID_LLM_BACKENDS)})")

        if self.stt_backend == "openai" and not self.stt_openai_api_key:
            raise ConfigError(
                "STT_BACKEND=openai, but no API key "
                "(STT_OPENAI_API_KEY or OPENAI_API_KEY) set.")
        if self.tts_backend == "openai" and not self.tts_openai_api_key:
            raise ConfigError(
                "TTS_BACKEND=openai, but no API key "
                "(TTS_OPENAI_API_KEY or OPENAI_API_KEY) set.")
        if self.llm_backend == "openai_compat" and not self.llm_api_key:
            raise ConfigError(
                "LLM_BACKEND=openai_compat, but no API key "
                "(LLM_API_KEY or FIREWORKS_API_KEY) set.")
        if self.llm_backend == "openclaw" and not self.openclaw_gateway_token:
            raise ConfigError(
                "LLM_BACKEND=openclaw, but no OPENCLAW_GATEWAY_TOKEN set.")
        if self.tts_backend == "omnivoice_local" and self.omnivoice_mode == "clone" \
                and not self.omnivoice_ref_audio:
            raise ConfigError(
                "TTS_BACKEND=omnivoice_local with OMNIVOICE_MODE=clone needs "
                "OMNIVOICE_REF_AUDIO.")


def load_config(env_file: str | os.PathLike | None = None) -> Config:
    """Loads .env (if present) and builds the validated Config.
    ``env_file`` overrides the path; otherwise VOICE_ENV_FILE or ./.env next to
    the project root.
    """
    if env_file is None:
        env_file = os.environ.get(
            "VOICE_ENV_FILE",
            str(Path(__file__).resolve().parent.parent / ".env"),
        )
    load_dotenv(Path(env_file))
    cfg = Config.from_env()
    return cfg
