"""Integration: ConversationManager history + full audio pipeline (mock backends)."""
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
    """Minimal WS double: collects only send_json frames (for unit tests)."""
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
    # second call contains the history of the first.
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
    assert [m["content"] for m in msgs] == ["b-frage"]  # B doesn't know A


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


# --- full audio pipeline over WebSocket -------------------------------------
def _configure_audio(reply="Audio-Antwort."):
    # Wake word off here: these tests check the audio pipeline, not the gate
    # (FakeSTT returns "hallo welt", which would otherwise be discarded without
    # a wake word).
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
            # segment.start (metadata) + binary audio (1s 16kHz float32)
            await ws.send_json({"type": "segment.start", "segmentId": "seg1"})
            pcm = np.zeros(SAMPLE_RATE, dtype=np.float32).tobytes()
            await ws.send_bytes(pcm)
            # Expectation: transcript → turn.commit → reply → audio
            transcript, seen, _ = await _drain_until(ws, "transcript")
            assert transcript["text"] == "hallo welt"
            reply, seen2, _ = await _drain_until(ws, "reply")
            assert reply["text"] == "Antwort auf Sprache."
            await ws.close()

    asyncio.run(run())


def test_audio_segment_reports_first_audio_latency():
    """audio.start carries E2E/first latencies (up to first playback),
    separate from the total times in reply/audio.end."""
    _configure_audio("Antwort auf Sprache.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.start", "segmentId": "seg1"})
            await ws.send_bytes(np.zeros(SAMPLE_RATE, dtype=np.float32).tobytes())
            start, seen, _ = await _drain_until(ws, "audio.start")
            assert start is not None, f"kein audio.start; gesehen: {seen}"
            # E2E (done speaking → first playback) is set, because the anchor
            # was set at the voice segment.
            assert isinstance(start.get("e2eMs"), int) and start["e2eMs"] >= 0
            assert isinstance(start.get("llmFirstMs"), int) and start["llmFirstMs"] >= 0
            assert isinstance(start.get("ttsFirstMs"), int) and start["ttsFirstMs"] >= 0
            # debounceMs accompanies e2eMs so the client can split off the pause.
            assert start.get("debounceMs") == 30   # from _configure_audio
            await ws.close()

    asyncio.run(run())


def test_streamed_input_segment_end_to_end():
    """B1: segment.stream.start + frames + commit → assembled → STT → reply."""
    _configure_audio("Antwort auf gestreamte Sprache.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.stream.start", "segmentId": "s1"})
            half = np.zeros(SAMPLE_RATE // 2, dtype=np.float32).tobytes()
            await ws.send_bytes(half)        # Frame 1 (while "speaking")
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
    """B2: while a segment is streamed in, transcript.partial events arrive."""
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
            # 0.5s audio → enough "new" audio for a partial.
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
    """segment.stream.abort discards the segment — no transcript/reply."""
    _configure_audio()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.stream.start", "segmentId": "s9"})
            await ws.send_bytes(np.zeros(SAMPLE_RATE // 2, dtype=np.float32).tobytes())
            await ws.send_json({"type": "segment.stream.abort"})
            # Then a ping → ack must arrive, but NO transcript before it.
            await ws.send_json({"type": "ping", "ts": 1})
            ack, seen, _ = await _drain_until(ws, "ack")
            assert ack is not None
            assert "transcript" not in seen
            await ws.close()

    asyncio.run(run())


class _ScriptedSTT:
    """Returns predefined transcripts in order (for wake-word tests)."""
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
    # `enabled` = start default of the wake mode (corresponds to WAKE_WORD_ENABLED).
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
            assert commit["text"] == "wie spät ist es?"   # wake word stripped
            reply, _, _ = await _drain_until(ws, "reply")
            assert reply["text"] == "Es ist drei."
            await ws.close()

    asyncio.run(run())


def test_wake_followup_within_window_bypasses_gate():
    # 1. segment with wake word → opens window; 2. without wake word → accepted.
    _configure_wake(["Antonia, hallo.", "Und wie geht es dir?"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            r1, _, _ = await _drain_until(ws, "reply")
            assert r1 is not None
            # Follow-up question without wake word — within the open window.
            await ws.send_json({"type": "segment.start", "segmentId": "g2"})
            await _send_voice(ws, "g2")
            commit, seen, _ = await _drain_until(ws, "turn.commit")
            assert commit is not None, f"Folgefrage verworfen; gesehen: {seen}"
            assert commit["text"] == "Und wie geht es dir?"
            assert "transcript.ignored" not in seen
            await ws.close()

    asyncio.run(run())


def test_wake_default_off_no_gate():
    # Start default OFF → segment without wake word still passes (no gate).
    _configure_wake(["Wie spät ist es?"], reply="Es ist drei.", enabled=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello = await ws.receive_json()
            assert hello["wakeWord"]["available"] is True
            assert hello["wakeWord"]["enabled"] is False     # start default
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            reply, seen, _ = await _drain_until(ws, "reply")
            assert reply is not None, f"kein reply; gesehen: {seen}"
            assert "transcript.ignored" not in seen
            await ws.close()

    asyncio.run(run())


def test_wake_mode_toggled_on_via_settings_gates():
    # Default OFF; client turns on wake mode via 'settings' → now a segment
    # without a wake word is discarded.
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
    # Wake on → wake word opens window; wake off closes it (wake_until=0).
    # After re-enabling, the window is closed → follow-up segment without a wake
    # word is discarded again (the window was not kept across the off).
    _configure_wake(["Antonia, hallo.", "Wie geht es?"], enabled=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            await _drain_until(ws, "reply")            # opens the window
            # Off and on again → window must stay closed.
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
    # Early cue: the wake word in the (streamed) partial triggers exactly ONE
    # wake.detected per segment — even with multiple partials.
    _configure_wake(["Antonia hallo"], enabled=False)

    async def run():
        state = TurnState()
        state.wake_word_enabled = True        # wake mode active, window closed
        ws = _CollectWS()
        seg = {"id": "s1", "done": False}
        await srv._do_partial(ws, state, seg, b"")
        await srv._do_partial(ws, state, seg, b"")   # second partial, same segment
        types = [m["type"] for m in ws.sent]
        assert types.count("wake.detected") == 1, types

    asyncio.run(run())


def test_partial_no_wake_detected_when_mode_off():
    _configure_wake(["Antonia hallo"], enabled=False)

    async def run():
        state = TurnState()
        state.wake_word_enabled = False       # no wake mode → no cue
        ws = _CollectWS()
        await srv._do_partial(ws, state, {"id": "s1", "done": False}, b"")
        assert "wake.detected" not in [m["type"] for m in ws.sent]

    asyncio.run(run())


def test_wake_accept_emits_detected_and_window():
    # Accepted wake-word segment → wake.detected (cue) + wake.window (timer).
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
    # 1. wake-word turn (detected+window). 2. follow-up question in the window:
    # window is refreshed, but NO new wake.detected (no wake word needed).
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
            assert "wake.window" in seen2, seen2        # window refreshed
            assert "wake.detected" not in seen2, seen2  # no second cue
            await ws.close()

    asyncio.run(run())


def test_wake_command_window_reason_is_command():
    # Incoming command → wake.window with reason=command (a reply is coming, the
    # client does NOT run the idle timer).
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
    # Only the wake word → wake.window with reason=armed (Antonia waits → timer on).
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
            armed, _, _ = await _drain_until(ws, "wake.armed")  # follows right after
            assert armed is not None
            await ws.close()

    asyncio.run(run())


def test_stop_command_closes_window():
    # In the open window, "stop" ends the window (wake.closed, no turn).
    _configure_wake(["Antonia, erzähl was.", "stop"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")
            await _drain_until(ws, "reply")          # window is now open
            await ws.send_json({"type": "segment.start", "segmentId": "g2"})
            await _send_voice(ws, "g2")              # "stop"
            ev, seen, _ = await _drain_until(ws, "wake.closed")
            assert ev is not None, f"kein wake.closed; gesehen: {seen}"
            assert ev["reason"] == "stop_command"
            assert "reply" not in seen and "turn.commit" not in seen
            await ws.close()

    asyncio.run(run())


def test_wake_followup_accepted_if_speech_started_in_window():
    # Race: window lapses between speech start and commit/STT. Whoever began
    # speaking while the window was still open must still pass through.
    import time as _t
    _configure_wake(["Wie geht es?"], enabled=False)

    async def run():
        state = TurnState(); state.wake_word_enabled = True
        ws = _CollectWS()
        state.wake_until = _t.time()                    # already lapsed at commit
        await srv._handle_audio_segment(
            ws, state, b"\x00\x00\x00\x00", "s1", "peer",
            speech_start_ts=_t.time() - 1.0)            # speech start was IN the window
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
        state.wake_until = _t.time()                    # lapsed
        await srv._handle_audio_segment(
            ws, state, b"\x00\x00\x00\x00", "s1", "peer",
            speech_start_ts=None)                       # → commit time, after lapse
        if state.debounce_task:
            state.debounce_task.cancel()
        types = [m.get("type") for m in ws.sent]
        assert "transcript.ignored" in types, types
        assert "turn.pending" not in types

    asyncio.run(run())


def test_barge_in_while_playing_keeps_window_open():
    # If you interrupt an ongoing playback (barge_in playing=true), the wake
    # window stays open → the interrupting input is NOT discarded as "no wake
    # word".
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
            await _send_voice(ws, "g1")          # without wake word
            commit, seen, _ = await _drain_until(ws, "turn.commit")
            assert commit is not None, f"Unterbrechung verworfen; gesehen: {seen}"
            assert "transcript.ignored" not in seen
            await ws.close()

    asyncio.run(run())


def test_barge_in_without_playing_does_not_open_window():
    # Mere noise without an ongoing reply must NOT open the window
    # (otherwise the gate would be permanently open).
    _configure_wake(["Wie geht es?"], enabled=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack")
            await ws.send_json({"type": "barge_in", "reason": "vad-speech", "playing": False})
            await ws.send_json({"type": "segment.start", "segmentId": "g1"})
            await _send_voice(ws, "g1")          # without wake word
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
