"""Wake-Word-Pipeline unter Last: was passiert mit dem Konversationsfenster,
wenn WÄHREND einer laufenden Verarbeitung weiterer Voice-Input reinkommt?

Diese Tests simulieren die echte Pipeline mit Mock-Backends:
  - ``ScriptedSTT``        liefert pro Segment einen vorgegebenen Text,
  - ``GatedStreamingLLM``  hält die Antwort kontrolliert „in flight" (erstes
                           Delta sofort, Rest erst nach Freigabe),
  - ``FakeTTS``            liefert sofort etwas PCM.

Damit lässt sich gezielt nachstellen: Fenster offen → Befehl → Antwort läuft →
Fenster MANUELL schließen → noch während die Antwort läuft kommt weiterer
Voice-Input (Echo der eigenen TTS, Folgegerede, Fuzzy-Fehltreffer). Das Fenster
darf dabei NICHT wieder aufspringen.
"""
import asyncio
import dataclasses

from aiohttp.test_utils import TestClient, TestServer

from plauder import server as srv
from plauder.config import Config
from plauder.sanitizer import HallucinationFilter
from plauder.session import ConversationManager


# --- Mock-Backends ----------------------------------------------------------
class ScriptedSTT:
    """Liefert pro transcribe()-Aufruf den nächsten Text aus der Liste."""
    last_no_speech_prob = None

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    async def transcribe(self, audio_pcm, sample_rate):
        i = self.calls
        self.calls += 1
        return self.texts[i] if i < len(self.texts) else ""

    def describe(self):
        return {"engine": "scripted-stt", "loaded": True}


class FakeTTS:
    sample_rate = 24000

    async def synth(self, text, *, speed=1.0):
        return b"\x00\x00\x01\x00", 24000

    def describe(self):
        return {"engine": "fake-tts", "sample_rate": 24000, "loaded": True}


class GatedStreamingLLM:
    """Hält die Antwort in-flight: erstes Delta sofort, Rest erst nach gate.set()."""
    loaded = True

    def __init__(self, deltas):
        self.deltas = list(deltas)
        self.gate = asyncio.Event()
        self.last_meta = {"finish_reason": "stop", "usage": {"total_tokens": 9}}

    async def chat(self, messages, system_hint=None):
        return "".join(self.deltas)

    async def chat_stream(self, messages, system_hint=None):
        if self.deltas:
            yield self.deltas[0]
        await self.gate.wait()
        for d in self.deltas[1:]:
            yield d
        self.last_meta = {"finish_reason": "stop", "usage": {"total_tokens": 9}}

    def describe(self):
        return {"engine": "gated-llm", "model": "fake", "ready": True}


def _configure_wake(stt_texts, deltas, *, window_s=8.0, guard_s=2.0, mode="conversation"):
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=20, streaming=True,
                              wake_word_window_s=window_s, wake_mode=mode)
    llm = GatedStreamingLLM(deltas)
    conv = ConversationManager(llm, system_prompt="sys")
    srv.configure(cfg, stt=ScriptedSTT(stt_texts), tts=FakeTTS(), conv=conv,
                  bridge=None, ghost=HallucinationFilter(enabled=False))
    srv.WAKE_CLOSE_GUARD_S = guard_s
    return cfg, llm


# --- WS-Helfer --------------------------------------------------------------
async def _drain_until(ws, want_type, *, timeout=3.0):
    seen = []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=max(0.02, deadline - loop.time()))
        except asyncio.TimeoutError:
            break
        if msg.type.name == "BINARY":
            seen.append("__binary__")
            continue
        if msg.type.name in ("CLOSE", "CLOSING", "CLOSED", "ERROR"):
            break
        data = msg.json()
        seen.append(data.get("type"))
        if data.get("type") == want_type:
            return data, seen
    return None, seen


async def _collect(ws, seconds):
    """Sammelt ALLE Frame-Typen für `seconds` — um die ABWESENHEIT von Events
    (z.B. wake.window) zu prüfen."""
    seen = []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=max(0.02, deadline - loop.time()))
        except asyncio.TimeoutError:
            break
        if msg.type.name == "BINARY":
            seen.append("__binary__")
            continue
        if msg.type.name in ("CLOSE", "CLOSING", "CLOSED", "ERROR"):
            break
        seen.append(msg.json().get("type"))
    return seen


async def _enable_wake(ws):
    await ws.send_json({"type": "settings", "wakeWordEnabled": True})
    await _drain_until(ws, "settings.ack")


async def _send_voice(ws, seg_id):
    """Ein 'committed' Segment: segment.start (Meta) + ein Binär-PCM-Frame."""
    await ws.send_json({"type": "segment.start", "segmentId": seg_id})
    await ws.send_bytes(b"\x00" * 800)


# --- Tests ------------------------------------------------------------------
def test_nonwake_input_during_processing_does_not_reopen_window():
    """Fenster geschlossen, Antwort läuft → Folgegerede OHNE Weckwort darf das
    Fenster nicht aufreißen und die laufende Antwort nicht abbrechen."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia erzähl eine geschichte", "nur so nebenbei ohne anrede"],
        deltas=["Es war einmal eine Lampe. ", "Sie leuchtete schwach. ", "Schluss."],
        guard_s=0.0,  # Guard AUS: hier soll allein das Gating schützen
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)

            await _send_voice(ws, "s1")                  # Weckwort + Befehl
            rs, seen1 = await _drain_until(ws, "reply.start")
            assert rs is not None, f"keine reply.start; {seen1}"

            await ws.send_json({"type": "wake.close", "playing": True})
            wc, _ = await _drain_until(ws, "wake.closed")
            assert wc is not None

            await _send_voice(ws, "s2")                  # KEIN Weckwort, mitten drin
            seen = await _collect(ws, 0.6)
            assert "wake.window" not in seen, f"Fenster wieder aufgegangen: {seen}"
            assert "wake.detected" not in seen, f"wake.detected trotz geschlossen: {seen}"
            assert "transcript.ignored" in seen, f"Segment nicht ignoriert: {seen}"

            # Laufende Antwort freigeben → sie muss UNVERSEHRT zu Ende laufen.
            llm.gate.set()
            reply, seen2 = await _drain_until(ws, "reply")
            assert reply is not None, f"Antwort abgebrochen? {seen2}"
            assert reply["text"] == "Es war einmal eine Lampe. Sie leuchtete schwach. Schluss."
            await ws.close()

    asyncio.run(run())


def test_wakeword_match_during_processing_is_ignored_while_closed():
    """Selbst wenn STT während der laufenden Antwort etwas mit Weckwort liefert
    (Echo der eigenen Stimme / Fuzzy-Fehltreffer), bleibt das Fenster zu —
    solange die Antwort noch läuft."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia erzähl eine geschichte", "antonia neue frage zwischendurch"],
        deltas=["Teil eins. ", "Teil zwei. ", "Teil drei."],
        guard_s=0.0,
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)

            await _send_voice(ws, "s1")
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            await ws.send_json({"type": "wake.close", "playing": True})
            assert (await _drain_until(ws, "wake.closed"))[0] is not None

            await _send_voice(ws, "s2")  # enthält "antonia" → würde sonst öffnen
            seen = await _collect(ws, 0.6)
            assert "wake.window" not in seen, f"Echo/Fehltreffer hat geöffnet: {seen}"
            assert "wake.detected" not in seen, f"wake.detected trotz geschlossen: {seen}"
            assert "turn.discarded" not in seen, f"laufende Antwort abgebrochen: {seen}"

            llm.gate.set()
            reply, seen2 = await _drain_until(ws, "reply")
            assert reply is not None and reply["text"] == "Teil eins. Teil zwei. Teil drei.", seen2
            await ws.close()

    asyncio.run(run())


def test_playback_done_after_manual_close_does_not_reopen():
    """Schließt der User mitten in der Antwort, darf das playback.done am Ende
    der Antwort das Fenster NICHT wieder öffnen."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia sag was"],
        deltas=["Erster Satz. ", "Zweiter Satz."],
        guard_s=0.0,
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)
            await _send_voice(ws, "s1")
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            await ws.send_json({"type": "wake.close", "playing": True})
            assert (await _drain_until(ws, "wake.closed"))[0] is not None

            llm.gate.set()
            end, _ = await _drain_until(ws, "audio.end")
            assert end is not None
            # Client meldet Wiedergabe-Ende → darf das Fenster nicht öffnen.
            await ws.send_json({"type": "playback.done", "turnId": end.get("turnId"),
                                "audioId": end.get("audioId")})
            seen = await _collect(ws, 0.4)
            assert "wake.window" not in seen, f"playback.done hat Fenster geöffnet: {seen}"
            await ws.close()

    asyncio.run(run())


def test_barge_in_does_not_abort_reply_while_wake_closed():
    """Kern des gemeldeten Bugs: Fenster geschlossen, Antonia denkt/spricht noch,
    der User redet weiter → das Client-VAD schickt ein barge_in. Bei
    geschlossenem Fenster darf das die laufende Antwort NICHT abbrechen."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia sag was"],
        deltas=["Erster Teil. ", "Zweiter Teil. ", "Dritter Teil."],
        guard_s=5.0,
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)
            await _send_voice(ws, "s1")
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            await ws.send_json({"type": "wake.close", "playing": True})
            assert (await _drain_until(ws, "wake.closed"))[0] is not None

            # User redet weiter → Client würde barge_in senden (VAD onSpeechStart).
            await ws.send_json({"type": "barge_in", "reason": "vad-speech", "playing": False})
            seen = await _collect(ws, 0.5)
            assert "turn.discarded" not in seen, f"Antwort durch barge_in abgebrochen: {seen}"

            llm.gate.set()
            reply, seen2 = await _drain_until(ws, "reply")
            assert reply is not None, f"Antwort abgebrochen? {seen2}"
            assert reply["text"] == "Erster Teil. Zweiter Teil. Dritter Teil."
            await ws.close()

    asyncio.run(run())


def test_conversation_mode_reopens_window_after_reply():
    """Standard-Modus: nach der Antwort öffnet playback.done das Konversations-
    fenster wieder (Folgefragen ohne Weckwort)."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia sag was"], deltas=["Fertig."], mode="conversation")
    llm.gate.set()  # nicht blockieren

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)
            await _send_voice(ws, "s1")
            end, _ = await _drain_until(ws, "audio.end")
            await ws.send_json({"type": "playback.done", "turnId": end.get("turnId"),
                                "audioId": end.get("audioId")})
            win, seen = await _drain_until(ws, "wake.window")
            assert win is not None and win.get("reason") == "done", seen
            await ws.close()

    asyncio.run(run())


def test_alexa_mode_closes_window_after_reply():
    """Alexa-Modus (WAKE_MODE=alexa): nach der Antwort schließt das Fenster
    (wake.closed reason=oneshot), KEIN Folgefragen-Fenster — und eine Folgefrage
    OHNE Weckwort wird danach ignoriert."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia sag was", "und was ist mit morgen"],
        deltas=["Fertig."], mode="alexa")
    llm.gate.set()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)
            await _send_voice(ws, "s1")
            end, _ = await _drain_until(ws, "audio.end")
            await ws.send_json({"type": "playback.done", "turnId": end.get("turnId"),
                                "audioId": end.get("audioId")})
            closed, seen = await _drain_until(ws, "wake.closed")
            assert closed is not None and closed.get("reason") == "oneshot", seen
            assert "wake.window" not in seen, f"One-Shot öffnete trotzdem: {seen}"

            # Folgefrage OHNE Weckwort → muss ignoriert werden (Fenster ist zu).
            await _send_voice(ws, "s2")
            ig, seen2 = await _drain_until(ws, "transcript.ignored")
            assert ig is not None, f"Folgefrage nicht ignoriert: {seen2}"
            assert "reply.start" not in seen2, f"Folgefrage löste Antwort aus: {seen2}"
            await ws.close()

    asyncio.run(run())


def test_hello_advertises_wake_mode():
    _configure_wake(stt_texts=[], deltas=["x"], mode="alexa")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello, _ = await _drain_until(ws, "hello")
            assert hello["wakeWord"]["mode"] == "alexa"
            await ws.close()

    asyncio.run(run())


def test_fresh_wakeword_after_processing_reopens():
    """Gegenprobe: NACH Abschluss der Antwort (und abgelaufenem Guard) muss ein
    frisch gesprochenes Weckwort das Fenster wieder öffnen — sonst hätten wir
    überdrückt."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia sag was", "antonia bist du da"],
        deltas=["Kurz. ", "Antwort."],
        guard_s=0.1,
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)
            await _send_voice(ws, "s1")
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            await ws.send_json({"type": "wake.close", "playing": True})
            assert (await _drain_until(ws, "wake.closed"))[0] is not None

            llm.gate.set()
            end, _ = await _drain_until(ws, "audio.end")
            await ws.send_json({"type": "playback.done", "turnId": end.get("turnId"),
                                "audioId": end.get("audioId")})
            await _collect(ws, 0.05)
            await asyncio.sleep(0.25)   # Guard (0.1s) sicher abgelaufen

            await _send_voice(ws, "s2")  # frisches Weckwort → muss öffnen
            win, seen = await _drain_until(ws, "wake.window", timeout=2.0)
            assert win is not None, f"frisches Weckwort öffnete NICHT: {seen}"
            await ws.close()

    asyncio.run(run())
