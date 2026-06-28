"""Konfiguration: .env-Loading, Config-Dataclass und Validierung.

Eine einzige Quelle der Wahrheit für alle Laufzeit-Optionen. Die Werte werden
beim Start aus der Umgebung (os.environ, gefüllt aus der .env) gelesen und in
ein eingefrorenes ``Config``-Objekt gepackt, das durch den Server gereicht wird.

Backend-Auswahl erfolgt über ``STT_BACKEND`` / ``TTS_BACKEND`` / ``LLM_BACKEND``.
Damit die *bestehende* .env (OpenAI + Fireworks, ohne neue Backend-Variablen)
weiter funktioniert, gibt es Fallback-Ketten auf die alten Variablennamen
(``OPENAI_API_KEY``, ``FIREWORKS_*`` usw.).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# OpenAI-TTS liefert raw-PCM bei fester Sample-Rate. Default 24 kHz (OpenAI Cloud,
# Kokoro); per TTS_OPENAI_SAMPLE_RATE überschreibbar (z.B. 16000 für Piper-kerstin,
# 22050 für Piper-thorsten) — sonst klingt die Wiedergabe zu schnell/langsam.
OPENAI_TTS_SAMPLE_RATE = 24000
# Browser liefert/erwartet 16 kHz Mono.
SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# .env-Loader (kein Pflicht-Dependency)
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path) -> None:
    """Minimaler .env-Parser. Bereits gesetzte echte ENV-Variablen gewinnen
    (override=False), damit Tests und Shell-Exports Vorrang behalten.
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
# ENV-Helfer
# --------------------------------------------------------------------------- #
def _first(*names_or_values: str | None, default: str = "") -> str:
    """Erste nicht-leere Variante. Argumente sind bereits aufgelöste Werte."""
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


_DEFAULT_VOICE_HINT = (
    "\n\n---\n"
    "VOICE-MODE (wichtig für diese Sprach-Session):\n"
    "• Du sprichst gerade per Sprache. Deine Antworten werden von einem "
    "Text-to-Speech-System vorgelesen.\n"
    "• KEINE Emojis – sie würden vorgelesen.\n"
    "• KEIN Markdown, keine Aufzählungszeichen, keine Code-Blöcke, keine URLs "
    "vorlesen – beschreibe sie kurz in Worten.\n"
    "• Kurze, natürliche, gesprochene Sätze. Deutsch als Default.\n"
    "• Halte dich so knapp wie möglich."
)

VALID_STT_BACKENDS = ("openai", "whisper_local")
VALID_TTS_BACKENDS = ("openai", "omnivoice_local")
VALID_LLM_BACKENDS = ("openai_compat", "openclaw")


class ConfigError(RuntimeError):
    """Konfiguration ist ungültig (fehlende Keys, unbekanntes Backend …)."""


@dataclass(frozen=True)
class Config:
    # --- General ---
    host: str = "127.0.0.1"
    port: int = 8319
    agent_name: str = "Antonia"
    soul_path: str = ""
    log_level: str = "INFO"

    # --- Backend-Auswahl ---
    stt_backend: str = "openai"
    tts_backend: str = "openai"
    llm_backend: str = "openai_compat"

    # --- STT: OpenAI ---
    stt_openai_api_key: str = ""
    stt_openai_model: str = "whisper-1"
    stt_openai_base_url: str | None = None
    stt_language: str | None = "de"

    # --- STT: faster_whisper (lokal) ---
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_local_files_only: bool = True

    # --- STT Partial / Streaming (B2) ---
    # Während ein Segment noch eingestreamt wird (B1), den bisher angesammelten
    # Audio-Puffer periodisch transkribieren und Zwischenergebnisse (transcript.
    # partial) schicken. Sinnvoll v.a. mit lokalem Whisper (billig wiederholbar).
    stt_partial: bool = False
    stt_partial_min_interval_ms: int = 700   # Mindestabstand zwischen Partials
    stt_partial_min_new_ms: int = 500        # Mindest-Neuaudio bis zum nächsten Partial

    # --- STT Halluzinations-Filter ---
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
    # local_speed=True: TTS_SPEED / UI-Slider lokal per Zeitdehnung (tonhöhen-
    # erhaltend) umsetzen, statt `speed` an den Server zu schicken — nötig für
    # Server, die den OpenAI-`speed`-Parameter ignorieren (z.B. lokales XTTS).
    tts_openai_local_speed: bool = False
    tts_speed: float = 1.0

    # --- TTS: OmniVoice (lokal) ---
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

    # --- Turn-Taking ---
    debounce_ms: int = 1200
    debounce_ms_min: int = 200
    debounce_ms_max: int = 5000
    tts_sentence_split: bool = False
    tts_sentence_gap_ms: int = 120
    tts_max_chars_per_chunk: int = 220

    # --- Wake-Word (STT-Prefix-Gate) ---
    # Wake-Word ist ein Eingabe-Modus (neben VAD & Push-to-Talk), der in der UI
    # gewählt wird. Im Wake-Modus lösen nur Segmente, deren Transkript mit dem
    # Wake-Word beginnt (Füllwörter davor erlaubt), einen Turn aus — der Rest
    # wird verworfen. Nach einer Antwort bleibt das Gate wake_word_window_s offen
    # (Folgefragen ohne erneutes Wake-Word).
    # `wake_word_enabled` ist nur noch der START-Default: WAKE_WORD_ENABLED=1
    # lässt die UI im Wake-Modus starten. Default AUS → Start im VAD-Modus.
    wake_word_enabled: bool = False
    wake_word: str = "antonia"          # leer in der ENV → folgt AGENT_NAME
    # Verhalten NACH einer Antwort:
    #   "conversation" = Konversationsfenster bleibt wake_word_window_s offen,
    #                    Folgefragen ohne Weckwort möglich (Standard).
    #   "alexa"        = One-Shot: nach der Antwort schließt das Fenster sofort,
    #                    jeder neue Befehl braucht wieder das Weckwort.
    wake_mode: str = "conversation"
    wake_word_window_s: float = 8.0     # Konversationsfenster nach einer Antwort
    wake_word_fuzzy: bool = True        # Whisper-Verhörer tolerieren
    wake_word_anywhere: bool = False    # True = Wake-Word irgendwo im Satz statt am Anfang
    wake_word_strip: bool = True        # Wake-Word vor dem LLM aus dem Text schneiden
    wake_word_ratio: float = 0.78       # Fuzzy-Schwelle (höher = strenger)

    # --- Streaming (Latenz) ---
    # streaming=True: LLM-Token werden gestreamt, satzweise sofort an TTS gegeben
    # und als PCM-Chunks progressiv an den Client geschickt (A1+A2). Bei False
    # gilt der klassische „erst komplett, dann ein WAV"-Pfad (Fallback).
    streaming: bool = True
    # Ziel-Chunkgröße für die progressive Audio-Übertragung (ms je VCT2-Frame).
    tts_chunk_ms: int = 400

    # --- House Mode ---
    house_mode: bool = False
    house_data_dir: str = ""
    house_speaker_id: bool = False
    house_wake_word: bool = False
    house_auth: bool = False

    # --- Voice-Persona (optional) ---
    # Optionale Persona, die dem knappen Voice-Hinweis vorangestellt wird. Leer
    # = keine Persona (nur "kurz fassen, kein Markdown/Emoji"). Quelle:
    # SOUL.md-Datei (soul_path) > system_prompt. Voller Prompt-Override: voice_mode_system.
    system_prompt: str = ""
    voice_mode_system: str = ""

    # --- Warmups ---
    stt_warmup: bool = False
    tts_warmup: bool = False

    # --- Pronunciations ---
    pronunciations_file: str = ""

    @property
    def soul_persona(self) -> str:
        """Optionale Persona: SOUL.md-Datei > system_prompt (Config/.env) > leer.
        Standardmäßig KEINE Persona — dann bleibt nur der knappe Voice-Hinweis
        (siehe resolved_voice_system). Nicht gecacht — billig."""
        if self.soul_path:
            try:
                text = Path(self.soul_path).read_text(encoding="utf-8").strip()
                if text:
                    return text
            except OSError as exc:
                print(f"⚠️  Persona-Datei nicht ladbar ({self.soul_path}): {exc} "
                      "— nutze system_prompt/keine Persona.", file=sys.stderr)
        return self.system_prompt.strip()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> "Config":
        """Baut die Config aus os.environ inkl. Legacy-Fallback-Ketten."""
        # Persona-Datei optional; KEIN persönlicher Default-Pfad. Leer = Persona
        # kommt aus SYSTEM_PROMPT bzw. dem generischen Default (siehe soul_persona).
        soul_path = _env("SOUL_PATH")

        # Voice-Mode-System: komplett überschreibbar, sonst Persona + Hint.
        # (Persona wird lazy via .soul_persona gelesen; hier nur der Hint-Teil
        #  bzw. ein vollständiges Override.)
        voice_mode_system = _env("VOICE_MODE_SYSTEM")  # leer = später aus persona+hint

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

            stt_backend=_first(_env("STT_BACKEND"), default="openai").lower(),
            tts_backend=_first(_env("TTS_BACKEND"), default="openai").lower(),
            llm_backend=_first(_env("LLM_BACKEND"), default="openai_compat").lower(),

            # STT openai (Fallback: OPENAI_API_KEY)
            stt_openai_api_key=_first(_env("STT_OPENAI_API_KEY"), _env("OPENAI_API_KEY")),
            stt_openai_model=_first(_env("STT_OPENAI_MODEL"), default="whisper-1"),
            stt_openai_base_url=(_first(_env("STT_OPENAI_BASE_URL"), _env("OPENAI_BASE_URL")) or None),
            stt_language=(_first(_env("STT_LANGUAGE"), _env("WHISPER_LANGUAGE"), default="de") or None),

            # STT local whisper
            whisper_model=_first(_env("WHISPER_MODEL"), default="large-v3-turbo"),
            whisper_device=_first(_env("WHISPER_DEVICE"), default="cuda"),
            whisper_compute_type=_first(_env("WHISPER_COMPUTE_TYPE"), default="int8"),
            whisper_beam_size=_env_int("WHISPER_BEAM_SIZE", 5),
            whisper_local_files_only=env_flag("WHISPER_LOCAL_FILES_ONLY", True),

            # Partials standardmäßig nur bei lokalem Whisper (API-Backend würde
            # sonst pro Sekunde mehrfach kostenpflichtig aufgerufen).
            stt_partial=env_flag("STT_PARTIAL",
                                 _first(_env("STT_BACKEND"), default="openai").lower() == "whisper_local"),
            stt_partial_min_interval_ms=_env_int("STT_PARTIAL_MIN_INTERVAL_MS", 700),
            stt_partial_min_new_ms=_env_int("STT_PARTIAL_MIN_NEW_MS", 500),

            stt_hallucination_filter=env_flag("STT_HALLUCINATION_FILTER", True),
            stt_ghost_no_speech_prob=_env_float("STT_GHOST_NO_SPEECH_PROB", 0.6),
            stt_ghost_use_duration=env_flag("STT_GHOST_USE_DURATION", False),
            stt_ghost_max_dur_s=_env_float("STT_GHOST_MAX_DUR_S", 1.5),
            stt_ghost_extra_phrases=_env("STT_GHOST_EXTRA_PHRASES", ""),

            # TTS openai (Fallback: OPENAI_API_KEY, OPENAI_TTS_*)
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
            omnivoice_language=(_env("OMNIVOICE_LANGUAGE") or None),

            # LLM openai_compat (Fallback: FIREWORKS_*, OPENCLAW_GATEWAY_TOKEN)
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
            # Default = AGENT_NAME (kleingeschrieben), per WAKE_WORD übersteuerbar.
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
        """System-Prompt für die Sprach-Session.

        Standard: nur ein knapper Voice-Hinweis (kurz fassen, kein Markdown/Emoji
        — wird vorgelesen). Eine optionale Persona (system_prompt / SOUL.md) wird
        davor gesetzt, wenn gesetzt; sonst ohne Persona. Voller Override des
        gesamten Prompts via VOICE_MODE_SYSTEM.
        """
        if self.voice_mode_system:
            return self.voice_mode_system
        hint = os.environ.get("VOICE_MODE_HINT", _DEFAULT_VOICE_HINT)
        return (self.soul_persona + hint).strip()

    def validate(self) -> None:
        """Wirft ConfigError bei offensichtlich unbrauchbarer Konfiguration.
        Prüft NUR das jeweils aktive Backend — inaktive Backends brauchen keine
        Keys/Deps."""
        if self.stt_backend not in VALID_STT_BACKENDS:
            raise ConfigError(
                f"STT_BACKEND={self.stt_backend!r} unbekannt "
                f"(erlaubt: {', '.join(VALID_STT_BACKENDS)})")
        if self.tts_backend not in VALID_TTS_BACKENDS:
            raise ConfigError(
                f"TTS_BACKEND={self.tts_backend!r} unbekannt "
                f"(erlaubt: {', '.join(VALID_TTS_BACKENDS)})")
        if self.llm_backend not in VALID_LLM_BACKENDS:
            raise ConfigError(
                f"LLM_BACKEND={self.llm_backend!r} unbekannt "
                f"(erlaubt: {', '.join(VALID_LLM_BACKENDS)})")

        if self.stt_backend == "openai" and not self.stt_openai_api_key:
            raise ConfigError(
                "STT_BACKEND=openai, aber kein API-Key "
                "(STT_OPENAI_API_KEY oder OPENAI_API_KEY) gesetzt.")
        if self.tts_backend == "openai" and not self.tts_openai_api_key:
            raise ConfigError(
                "TTS_BACKEND=openai, aber kein API-Key "
                "(TTS_OPENAI_API_KEY oder OPENAI_API_KEY) gesetzt.")
        if self.llm_backend == "openai_compat" and not self.llm_api_key:
            raise ConfigError(
                "LLM_BACKEND=openai_compat, aber kein API-Key "
                "(LLM_API_KEY oder FIREWORKS_API_KEY) gesetzt.")
        if self.llm_backend == "openclaw" and not self.openclaw_gateway_token:
            raise ConfigError(
                "LLM_BACKEND=openclaw, aber kein OPENCLAW_GATEWAY_TOKEN gesetzt.")
        if self.tts_backend == "omnivoice_local" and self.omnivoice_mode == "clone" \
                and not self.omnivoice_ref_audio:
            raise ConfigError(
                "TTS_BACKEND=omnivoice_local mit OMNIVOICE_MODE=clone braucht "
                "OMNIVOICE_REF_AUDIO.")


def load_config(env_file: str | os.PathLike | None = None) -> Config:
    """Lädt .env (falls vorhanden) und baut die validierte Config.
    ``env_file`` überschreibt den Pfad; sonst VOICE_ENV_FILE oder ./.env neben
    dem Projekt-Root.
    """
    if env_file is None:
        env_file = os.environ.get(
            "VOICE_ENV_FILE",
            str(Path(__file__).resolve().parent.parent / ".env"),
        )
    load_dotenv(Path(env_file))
    cfg = Config.from_env()
    return cfg
