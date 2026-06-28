"""Per-Connection-Turn-State + VAD-Parameter.

Debounce + Coalescing: Voice-Segmente und Text-Sends teilen sich ein
Debounce-Fenster. Neue Eingaben während des Fensters werden gesammelt und
nach Ablauf in EINEM LLM-Call abgeschickt.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

# VAD-Frames ≈ 32 ms pro Frame bei 16 kHz/512 samples (silero-vad-web).
VAD_FRAME_MS = 32
VAD_REDEMPTION_MIN = 8       # ~256 ms
VAD_REDEMPTION_MAX = 160     # ~5.1 s


def vad_params_for_debounce(debounce_ms: int) -> dict:
    """VAD-Parameter abhängig vom gewünschten Debounce. Für lange Denkpausen
    muss das VAD die Stille länger durchhalten, bevor es "speech end" feuert.
    """
    frames = int(round((debounce_ms * 0.8) / VAD_FRAME_MS))
    frames = max(VAD_REDEMPTION_MIN, min(VAD_REDEMPTION_MAX, frames))
    return {
        "redemptionFrames": frames,
        "minSpeechFrames": 3,
        "preSpeechPadFrames": 8,
        "frameMs": VAD_FRAME_MS,
    }


@dataclass
class TurnState:
    """Hält den aktuellen Turn einer Connection (Voice + Text + Bilder)."""
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    pending_texts: list = field(default_factory=list)
    pending_segment_ids: list = field(default_factory=list)
    pending_text_parts: list = field(default_factory=list)
    pending_image_urls: list = field(default_factory=list)
    debounce_task: object = None
    agent_task: object = None
    audio_ids: set = field(default_factory=set)
    speed: float = 1.0
    debounce_ms: int = 1200
    # Legacy: parallele Text-Tasks; jetzt unbenutzt, bleibt für _cancel_in_flight.
    text_tasks: list = field(default_factory=list)
    # User-Suffix, der pro Connection den LLM-Session-Key bestimmt.
    session_user: str | None = None
    # Wake-Word ist ein per-Connection-Eingabemodus (vom Client gesetzt). Ist es
    # aus, läuft die Connection wie reines VAD/PTT ohne Gate. Startwert =
    # CFG.wake_word_enabled (Start-Default), danach via 'settings' umschaltbar.
    wake_word_enabled: bool = False
    # Wake-Word: bis zu diesem Zeitpunkt (time.time()) ist das Konversations-
    # fenster offen → Segmente ohne Wake-Word werden durchgelassen (Folgefragen).
    wake_until: float = 0.0
    # Segment-ID, für die bereits ein `wake.detected` (akustisches Früh-Feedback)
    # gesendet wurde — verhindert Doppel-Pling aus Partial + finalem Segment.
    wake_detected_seg: str | None = None
    # Hat der User das Gesprächsfenster während einer laufenden Antwort manuell
    # geschlossen ('wake.close')? Dann das sonst folgende playback.done NICHT zum
    # Wieder-Öffnen des Fensters nutzen.
    wake_suppress_reopen: bool = False
    # Nach manuellem Schließen kurzer Guard (time.time()-Schwelle): bis dahin KEIN
    # automatisches Wieder-Öffnen und kein wake.detected — sonst macht ein
    # nachlaufendes Partial / Echo / Störgeräusch das Fenster sofort wieder auf.
    wake_closed_until: float = 0.0
    # End-to-End-Latenz-Anker: time.time(), an dem das letzte zum Turn
    # beitragende Segment beim Server ankam ("User ist fertig mit Sprechen").
    # Gegen diesen Zeitpunkt wird die Zeit bis zur ersten Wiedergabe gemessen.
    speech_end_ts: float = 0.0

    def has_pending(self) -> bool:
        return bool(self.pending_texts or self.pending_text_parts or self.pending_image_urls)

    def reset(self) -> None:
        """Beginne neuen Turn (nach erfolgreichem Agent-Call)."""
        self.turn_id = uuid.uuid4().hex[:8]
        self.pending_texts.clear()
        self.pending_segment_ids.clear()
        self.pending_text_parts.clear()
        self.pending_image_urls.clear()
        self.speech_end_ts = 0.0
