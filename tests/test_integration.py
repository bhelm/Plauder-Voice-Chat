"""Integration: ConversationManager-Verlauf + volle Audio-Pipeline (mock backends)."""
import asyncio
import dataclasses

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from plauder import server as srv
from plauder.config import SAMPLE_RATE, Config
from plauder.sanitizer import HallucinationFilter
from plauder.session import ConversationManager
from plauder.turn_state import TurnState
from tests.test_server import FakeSTT, FakeTTS, FakeLLM, _drain_until


class _CollectWS:
    """Minimaler WS-Doppel: sammelt nur send_json-Frames (für Unit-Tests)."""
    def __init__(self):
        self.sent = []

    async def send_json(self, msg):
        self.sent.append(msg)


# --- ConversationManager ----------------------------------------------------
class _RecordingLLM:
    last_meta = {}

    def __init__(self):
        self.received = []
        self.reply = "antwort"

    async def chat(self, messages, system_hint=None):
        self.received.append((list(messages), system_hint))
        return self.reply


def test_conversation_keeps_history_across_turns():
    llm = _RecordingLLM()
    conv = ConversationManager(llm, system_prompt="SYS", history_turns=20)
    asyncio.run(conv.chat("erste frage", user_key="u1"))
    llm.reply = "zweite antwort"
    asyncio.run(conv.chat("zweite frage", user_key="u1"))
    # zweiter Call enthält den Verlauf des ersten.
    msgs, hint = llm.received[1]
    contents = [m["content"] for m in msgs]
    assert hint == "SYS"
    assert "erste frage" in contents
    assert "antwort" in contents
    assert "zweite frage" in contents


def test_conversation_reset_clears_history():
    llm = _RecordingLLM()
    conv = ConversationManager(llm, system_prompt="SYS")
    asyncio.run(conv.chat("frage eins", user_key="u1"))
    conv.reset("u1")
    asyncio.run(conv.chat("frage zwei", user_key="u1"))
    msgs, _ = llm.received[1]
    assert [m["content"] for m in msgs] == ["frage zwei"]


def test_conversation_separate_keys_isolated():
    llm = _RecordingLLM()
    conv = ConversationManager(llm, system_prompt="SYS")
    asyncio.run(conv.chat("a-frage", user_key="A"))
    asyncio.run(conv.chat("b-frage", user_key="B"))
    msgs, _ = llm.received[1]
    assert [m["content"] for m in msgs] == ["b-frage"]  # B kennt A nicht


def test_conversation_image_builds_multimodal():
    llm = _RecordingLLM()
    conv = ConversationManager(llm, system_prompt="SYS")
    asyncio.run(conv.chat("Was ist das?", user_key="u1",
                          image_urls=["http://x/y.png"]))
    msgs, _ = llm.received[0]
    user_msg = msgs[-1]
    assert isinstance(user_msg["content"], list)
    types = [p["type"] for p in user_msg["content"]]
    assert "text" in types and "image_url" in types


# --- volle Audio-Pipeline über WebSocket ------------------------------------
def _configure_audio(reply="Audio-Antwort."):
    # Wake-Word hier aus: diese Tests prüfen die Audio-Pipeline, nicht das Gate
    # (FakeSTT liefert "hallo welt", was sonst ohne Wake-Word verworfen würde).
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=30, wake_word_enabled=False)
    conv = ConversationManager(FakeLLM(reply), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))


def test_audio_segment_pipeline_end_to_end():
    _configure_audio("Antwort auf Sprache.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            # segment.start (Metadaten) + Binary-Audio (1s 16kHz float32)
            await ws.send_json({"type": "segment.start", "segmentId": "seg1"})
            pcm = np.zeros(SAMPLE_RATE, dtype=np.float32).tobytes()
            await ws.send_bytes(pcm)
            # Erwartung: transcript → turn.commit → reply → audio
            transcript, seen, _ = await _drain_until(ws, "transcript")
            assert transcript["text"] == "hallo welt"
            reply, seen2, _ = await _drain_until(ws, "reply")
            assert reply["text"] == "Antwort auf Sprache."
            await ws.close()

    asyncio.run(run())


def test_audio_segment_reports_first_audio_latency():
    """audio.start trägt E2E-/First-Latenzen (bis zur ersten Wiedergabe),
    getrennt von den Gesamtzeiten in reply/audio.end."""
    _configure_audio("Antwort auf Sprache.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.start", "segmentId": "seg1"})
            await ws.send_bytes(np.zeros(SAMPLE_RATE, dtype=np.float32).tobytes())
            start, seen, _ = await _drain_until(ws, "audio.start")
            assert start is not None, f"kein audio.start; gesehen: {seen}"
            # E2E (fertig gesprochen → erste Wiedergabe) ist gesetzt, weil der
            # Anker beim Voice-Segment gesetzt wurde.
            assert isinstance(start.get("e2eMs"), int) and start["e2eMs"] >= 0
            assert isinstance(start.get("llmFirstMs"), int) and start["llmFirstMs"] >= 0
            assert isinstance(start.get("ttsFirstMs"), int) and start["ttsFirstMs"] >= 0
            # debounceMs begleitet e2eMs, damit der Client die Pause abspalten kann.
            assert start.get("debounceMs") == 30   # aus _configure_audio
            await ws.close()

    asyncio.run(run())


def test_streamed_input_segment_end_to_end():
    """B1: segment.stream.start + Frames + commit → assembliert → STT → reply."""
    _configure_audio("Antwort auf gestreamte Sprache.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.stream.start", "segmentId": "s1"})
            half = np.zeros(SAMPLE_RATE // 2, dtype=np.float32).tobytes()
            await ws.send_bytes(half)        # Frame 1 (während "gesprochen" wird)
            await ws.send_bytes(half)        # Frame 2
            await ws.send_json({"type": "segment.stream.commit", "segmentId": "s1"})
            transcript, seen, _ = await _drain_until(ws, "transcript")
            assert transcript is not None, f"kein transcript; gesehen: {seen}"
            assert transcript["text"] == "hallo welt"
            reply, _, _ = await _drain_until(ws, "reply")
            assert reply["text"] == "Antwort auf gestreamte Sprache."
            await ws.close()

    asyncio.run(run())


def test_streamed_input_emits_partial_transcripts():
    """B2: während ein Segment eingestreamt wird, kommen transcript.partial."""
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=30, streaming=True,
                              wake_word_enabled=False,
                              stt_partial=True, stt_partial_min_interval_ms=0,
                              stt_partial_min_new_ms=100)
    conv = ConversationManager(FakeLLM("Antwort."), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.stream.start", "segmentId": "p1"})
            # 0.5s Audio → genug "neues" Audio für ein Partial.
            await ws.send_bytes(np.zeros(SAMPLE_RATE // 2, dtype=np.float32).tobytes())
            partial, seen, _ = await _drain_until(ws, "transcript.partial")
            assert partial is not None, f"kein transcript.partial; gesehen: {seen}"
            assert partial["text"] == "hallo welt"
            await ws.send_json({"type": "segment.stream.commit", "segmentId": "p1"})
            final, _, _ = await _drain_until(ws, "transcript")
            assert final["text"] == "hallo welt"
            await ws.close()

    asyncio.run(run())


def test_streamed_input_abort_discards_buffer():
    """segment.stream.abort verwirft das Segment — kein transcript/reply."""
    _configure_audio()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.stream.start", "segmentId": "s9"})
            await ws.send_bytes(np.zeros(SAMPLE_RATE // 2, dtype=np.float32).tobytes())
            await ws.send_json({"type": "segment.stream.abort"})
            # Danach ein ping → ack muss kommen, aber KEIN transcript davor.
            await ws.send_json({"type": "ping", "ts": 1})
            ack, seen, _ = await _drain_until(ws, "ack")
            assert ack is not None
            assert "transcript" not in seen
            await ws.close()

    asyncio.run(run())


class _ScriptedSTT:
    """Liefert vordefinierte Transkripte in Reihenfolge (für Wake-Word-Tests)."""
    last_no_speech_prob = None

    def __init__(self, texts):
        self.texts = list(texts)
        self.i = 0

    async def transcribe(self, audio_pcm, sample_rate):
        t = self.texts[min(self.i, len(self.texts) - 1)]
        self.i += 1
        return t

    def describe(self):
        return {"engine": "scripted-stt", "loaded": True}


def _configure_wake(texts, reply="Klar.", *, enabled=True):
    # `enabled` = Start-Default des Wake-Modus (entspricht WAKE_WORD_ENABLED).
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=30, streaming=False,
                              wake_word_enabled=enabled, wake_word="antonia",
                              wake_word_window_s=30.0)
    conv = ConversationManager(FakeLLM(reply), system_prompt="sys")
    srv.configure(cfg, stt=_ScriptedSTT(texts), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))


def _send_voice(ws, seg_id):
    return ws.send_bytes(np.zeros(SAMPLE_RATE // 2, dtype=np.float32).tobytes())


def test_wake_gate_ignores_without_wake_word():
    _configure_wake(["Wie spät ist es?"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            ev, seen, _ = await _drain_until(ws, "transcript.ignored")
            assert ev is not None, f"kein transcript.ignored; gesehen: {seen}"
            assert "reply" not in seen and "turn.commit" not in seen
            await ws.close()

    asyncio.run(run())


def test_wake_gate_accepts_and_strips_wake_word():
    _configure_wake(["Antonia, wie spät ist es?"], reply="Es ist drei.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            commit, seen, _ = await _drain_until(ws, "turn.commit")
            assert commit is not None, f"kein turn.commit; gesehen: {seen}"
            assert commit["text"] == "wie spät ist es?"   # Wake-Word abgeschnitten
            reply, _, _ = await _drain_until(ws, "reply")
            assert reply["text"] == "Es ist drei."
            await ws.close()

    asyncio.run(run())


def test_wake_followup_within_window_bypasses_gate():
    # 1. Segment mit Wake-Word → öffnet Fenster; 2. ohne Wake-Word → akzeptiert.
    _configure_wake(["Antonia, hallo.", "Und wie geht es dir?"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            r1, _, _ = await _drain_until(ws, "reply")
            assert r1 is not None
            # Folgefrage ohne Wake-Word — im offenen Fenster.
            await ws.send_json({"type": "segment.start", "segmentId": "g2"})
            await _send_voice(ws, "g2")
            commit, seen, _ = await _drain_until(ws, "turn.commit")
            assert commit is not None, f"Folgefrage verworfen; gesehen: {seen}"
            assert commit["text"] == "Und wie geht es dir?"
            assert "transcript.ignored" not in seen
            await ws.close()

    asyncio.run(run())


def test_wake_default_off_no_gate():
    # Start-Default AUS → Segment ohne Wake-Word geht trotzdem durch (kein Gate).
    _configure_wake(["Wie spät ist es?"], reply="Es ist drei.", enabled=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello = await ws.receive_json()
            assert hello["wakeWord"]["available"] is True
            assert hello["wakeWord"]["enabled"] is False     # Start-Default
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            reply, seen, _ = await _drain_until(ws, "reply")
            assert reply is not None, f"kein reply; gesehen: {seen}"
            assert "transcript.ignored" not in seen
            await ws.close()

    asyncio.run(run())


def test_wake_mode_toggled_on_via_settings_gates():
    # Default AUS; Client schaltet Wake-Modus per 'settings' an → jetzt wird
    # ein Segment ohne Wake-Word verworfen.
    _configure_wake(["Wie spät ist es?"], enabled=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            ack, _, _ = await _drain_until(ws, "settings.ack")
            assert ack["wakeWordEnabled"] is True
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            ev, seen, _ = await _drain_until(ws, "transcript.ignored")
            assert ev is not None, f"kein transcript.ignored; gesehen: {seen}"
            assert "reply" not in seen
            await ws.close()

    asyncio.run(run())


def test_wake_disable_closes_conversation_window():
    # Wake an → Wake-Wort öffnet Fenster; Wake aus schließt es (wake_until=0).
    # Nach erneutem Einschalten ist das Fenster zu → Folgesegment ohne Wake-Wort
    # wird wieder verworfen (Fenster wurde nicht über das Aus hinweg gehalten).
    _configure_wake(["Antonia, hallo.", "Wie geht es?"], enabled=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            await _drain_until(ws, "reply")            # öffnet das Fenster
            # Aus und wieder an → Fenster muss geschlossen bleiben.
            await ws.send_json({"type": "settings", "wakeWordEnabled": False})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "segment.start", "segmentId": "g2"})
            await _send_voice(ws, "g2")
            ev, seen, _ = await _drain_until(ws, "transcript.ignored")
            assert ev is not None, f"Fenster blieb offen; gesehen: {seen}"
            await ws.close()

    asyncio.run(run())


def test_partial_emits_wake_detected_once():
    # Früh-Pling: das Weckwort im (gestreamten) Partial löst genau EIN
    # wake.detected pro Segment aus — auch bei mehreren Partials.
    _configure_wake(["Antonia hallo"], enabled=False)

    async def run():
        state = TurnState()
        state.wake_word_enabled = True        # Wake-Modus aktiv, Fenster zu
        ws = _CollectWS()
        seg = {"id": "s1", "done": False}
        await srv._do_partial(ws, state, seg, b"")
        await srv._do_partial(ws, state, seg, b"")   # zweites Partial, gleiches Segment
        types = [m["type"] for m in ws.sent]
        assert types.count("wake.detected") == 1, types

    asyncio.run(run())


def test_partial_no_wake_detected_when_mode_off():
    _configure_wake(["Antonia hallo"], enabled=False)

    async def run():
        state = TurnState()
        state.wake_word_enabled = False       # kein Wake-Modus → kein Pling
        ws = _CollectWS()
        await srv._do_partial(ws, state, {"id": "s1", "done": False}, b"")
        assert "wake.detected" not in [m["type"] for m in ws.sent]

    asyncio.run(run())


def test_wake_accept_emits_detected_and_window():
    # Akzeptiertes Weckwort-Segment → wake.detected (Pling) + wake.window (Timer).
    _configure_wake(["Antonia, wie spät ist es?"], reply="Es ist drei.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            _, seen, _ = await _drain_until(ws, "reply")
            assert "wake.detected" in seen, seen
            assert "wake.window" in seen, seen
            await ws.close()

    asyncio.run(run())


def test_wake_followup_refreshes_window_without_redetect():
    # 1. Weckwort-Turn (detected+window). 2. Folgefrage im Fenster: window wird
    # aufgefrischt, aber KEIN erneutes wake.detected (kein Weckwort nötig).
    _configure_wake(["Antonia, hallo.", "Wie geht es?"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            await _drain_until(ws, "reply")
            await ws.send_json({"type": "segment.start", "segmentId": "g2"})
            await _send_voice(ws, "g2")
            commit, seen2, _ = await _drain_until(ws, "turn.commit")
            assert commit is not None, f"Folgefrage verworfen; gesehen: {seen2}"
            assert "wake.window" in seen2, seen2        # Fenster aufgefrischt
            assert "wake.detected" not in seen2, seen2  # kein zweites Pling
            await ws.close()

    asyncio.run(run())


def test_wake_command_window_reason_is_command():
    # Eingehender Befehl → wake.window mit reason=command (Antwort folgt, der
    # Client lässt den Idle-Timer NICHT laufen).
    _configure_wake(["Antonia, wie spät ist es?"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            win, seen, _ = await _drain_until(ws, "wake.window")
            assert win is not None, f"kein wake.window; gesehen: {seen}"
            assert win["reason"] == "command"
            await ws.close()

    asyncio.run(run())


def test_wake_armed_window_reason_is_armed():
    # Nur das Weckwort → wake.window mit reason=armed (Antonia wartet → Timer an).
    _configure_wake(["Antonia"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            win, seen, _ = await _drain_until(ws, "wake.window")
            assert win is not None, f"kein wake.window; gesehen: {seen}"
            assert win["reason"] == "armed"
            armed, _, _ = await _drain_until(ws, "wake.armed")  # folgt direkt danach
            assert armed is not None
            await ws.close()

    asyncio.run(run())


def test_stop_command_closes_window():
    # Im offenen Fenster beendet „stop" das Fenster (wake.closed, kein Turn).
    _configure_wake(["Antonia, erzähl was.", "stop"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            await _drain_until(ws, "reply")          # Fenster ist nun offen
            await ws.send_json({"type": "segment.start", "segmentId": "g2"})
            await _send_voice(ws, "g2")              # „stop"
            ev, seen, _ = await _drain_until(ws, "wake.closed")
            assert ev is not None, f"kein wake.closed; gesehen: {seen}"
            assert ev["reason"] == "stop_command"
            assert "reply" not in seen and "turn.commit" not in seen
            await ws.close()

    asyncio.run(run())


def test_wake_followup_accepted_if_speech_started_in_window():
    # Race: Fenster läuft zwischen Sprech-Beginn und Commit/STT ab. Wer im noch
    # offenen Fenster zu sprechen begann, muss trotzdem durchkommen.
    import time as _t
    _configure_wake(["Wie geht es?"], enabled=False)

    async def run():
        state = TurnState(); state.wake_word_enabled = True
        ws = _CollectWS()
        state.wake_until = _t.time()                    # zum Commit bereits abgelaufen
        await srv._handle_audio_segment(
            ws, state, b"\x00\x00\x00\x00", "s1", "peer",
            speech_start_ts=_t.time() - 1.0)            # Sprech-Beginn war IM Fenster
        if state.debounce_task:
            state.debounce_task.cancel()
        types = [m.get("type") for m in ws.sent]
        assert "transcript.ignored" not in types, types
        assert "turn.pending" in types, types

    asyncio.run(run())


def test_wake_followup_rejected_if_speech_started_after_window():
    import time as _t
    _configure_wake(["Wie geht es?"], enabled=False)

    async def run():
        state = TurnState(); state.wake_word_enabled = True
        ws = _CollectWS()
        state.wake_until = _t.time()                    # abgelaufen
        await srv._handle_audio_segment(
            ws, state, b"\x00\x00\x00\x00", "s1", "peer",
            speech_start_ts=None)                       # → Commit-Zeit, nach Ablauf
        if state.debounce_task:
            state.debounce_task.cancel()
        types = [m.get("type") for m in ws.sent]
        assert "transcript.ignored" in types, types
        assert "turn.pending" not in types

    asyncio.run(run())


def test_barge_in_while_playing_keeps_window_open():
    # Unterbricht man eine laufende Wiedergabe (barge_in playing=true), bleibt das
    # Wake-Fenster offen → die unterbrechende Eingabe wird NICHT als „kein
    # Weckwort" verworfen.
    _configure_wake(["Wie geht es?"], enabled=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "barge_in", "reason": "vad-speech", "playing": True})
            win, _, _ = await _drain_until(ws, "wake.window")
            assert win is not None and win["reason"] == "command"
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")          # ohne Weckwort
            commit, seen, _ = await _drain_until(ws, "turn.commit")
            assert commit is not None, f"Unterbrechung verworfen; gesehen: {seen}"
            assert "transcript.ignored" not in seen
            await ws.close()

    asyncio.run(run())


def test_barge_in_without_playing_does_not_open_window():
    # Bloßes Geräusch ohne laufende Antwort darf das Fenster NICHT öffnen
    # (sonst wäre das Gate dauerhaft offen).
    _configure_wake(["Wie geht es?"], enabled=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "barge_in", "reason": "vad-speech", "playing": False})
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")          # ohne Weckwort
            ev, seen, _ = await _drain_until(ws, "transcript.ignored")
            assert ev is not None, f"sollte verworfen werden; gesehen: {seen}"
            await ws.close()

    asyncio.run(run())


def test_audio_pipeline_uses_active_backends_via_healthz():
    _configure_audio()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            resp = await client.get("/healthz")
            body = await resp.json()
            assert body["backends"] == {
                "stt": "openai", "tts": "openai", "llm": "openai_compat"}

    asyncio.run(run())
