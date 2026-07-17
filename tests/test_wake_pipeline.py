"""Wake-word pipeline under load: what happens to the conversation window
when further voice input arrives WHILE processing is ongoing?

These tests simulate the real pipeline with mock backends:
  - ``ScriptedSTT``        returns a predefined text per segment,
  - ``GatedStreamingLLM``  keeps the reply controllably "in flight" (first
                           delta immediately, the rest only after release),
  - ``FakeTTS``            returns some PCM immediately.

This lets us reproduce: window open → command → reply running →
window closed MANUALLY → while the reply is still running, more
voice input arrives (echo of one's own TTS, follow-up chatter, fuzzy
mismatches). The window must NOT pop open again.
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
    """Returns the next text from the list per transcribe() call."""
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
    """Keeps the reply in-flight: first delta immediately, rest only after gate.set()."""
    loaded = True

    def __init__(self, deltas):
        self.deltas = list(deltas)
        self.gate = asyncio.Event()
        self.last_meta = {"finish_reason": "stop", "usage": {"total_tokens": 9}}
        self.calls = []   # message lists received, one entry per chat_stream call

    async def chat(self, messages, system_hint=None):
        return "".join(self.deltas)

    async def chat_stream(self, messages, system_hint=None):
        self.calls.append(list(messages))
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


# --- WS helpers -------------------------------------------------------------
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
    """Collects ALL frame types for `seconds` — to check the ABSENCE of events
    (e.g. wake.window)."""
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
    """A 'committed' segment: segment.start (meta) + one binary PCM frame."""
    await ws.send_json({"type": "segment.start", "segmentId": seg_id})
    await ws.send_bytes(b"\x00" * 800)


# --- Tests ------------------------------------------------------------------
def test_nonwake_input_during_processing_does_not_reopen_window():
    """Window closed, reply running → follow-up chatter WITHOUT a wake word must
    not tear the window open nor abort the running reply."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia erzähl eine geschichte", "nur so nebenbei ohne anrede"],
        deltas=["Es war einmal eine Lampe. ", "Sie leuchtete schwach. ", "Schluss."],
        guard_s=0.0,  # Guard OFF: here the gating alone should protect
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)

            await _send_voice(ws, "s1")                  # wake word + command
            rs, seen1 = await _drain_until(ws, "reply.start")
            assert rs is not None, f"keine reply.start; {seen1}"

            await ws.send_json({"type": "wake.close", "playing": True})
            wc, _ = await _drain_until(ws, "wake.closed")
            assert wc is not None

            await _send_voice(ws, "s2")                  # NO wake word, in the middle
            seen = await _collect(ws, 0.6)
            assert "wake.window" not in seen, f"Fenster wieder aufgegangen: {seen}"
            assert "wake.detected" not in seen, f"wake.detected trotz geschlossen: {seen}"
            assert "transcript.ignored" in seen, f"Segment nicht ignoriert: {seen}"

            # Release the running reply → it must finish UNHARMED.
            llm.gate.set()
            reply, seen2 = await _drain_until(ws, "reply")
            assert reply is not None, f"Antwort abgebrochen? {seen2}"
            assert reply["text"] == "Es war einmal eine Lampe. Sie leuchtete schwach. Schluss."
            await ws.close()

    asyncio.run(run())


def test_wakeword_match_during_processing_is_ignored_while_closed():
    """Even if STT returns something with a wake word during the running reply
    (echo of one's own voice / fuzzy mismatch), the window stays closed —
    as long as the reply is still running."""
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

            await _send_voice(ws, "s2")  # contains "antonia" → would otherwise open
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
    """If the user closes in the middle of the reply, the playback.done at the
    end of the reply must NOT reopen the window."""
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
            # Client reports end of playback → must not open the window.
            await ws.send_json({"type": "playback.done", "turnId": end.get("turnId"),
                                "audioId": end.get("audioId")})
            seen = await _collect(ws, 0.4)
            assert "wake.window" not in seen, f"playback.done hat Fenster geöffnet: {seen}"
            await ws.close()

    asyncio.run(run())


def test_barge_in_does_not_abort_reply_while_wake_closed():
    """Core of the reported bug: window closed, Antonia still thinking/speaking,
    the user keeps talking → the client VAD sends a barge_in. With the window
    closed, that must NOT abort the running reply."""
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

            # User keeps talking → client would send barge_in (VAD onSpeechStart).
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
    """Default mode: after the reply, playback.done reopens the conversation
    window (follow-up questions without a wake word)."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia sag was"], deltas=["Fertig."], mode="conversation")
    llm.gate.set()  # don't block

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
    """Alexa mode (WAKE_MODE=alexa): after the reply the window closes
    (wake.closed reason=oneshot), NO follow-up-question window — and a follow-up
    question WITHOUT a wake word is ignored afterwards."""
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

            # Follow-up question WITHOUT a wake word → must be ignored (window is closed).
            await _send_voice(ws, "s2")
            ig, seen2 = await _drain_until(ws, "transcript.ignored")
            assert ig is not None, f"Folgefrage nicht ignoriert: {seen2}"
            assert "reply.start" not in seen2, f"Folgefrage löste Antwort aus: {seen2}"
            await ws.close()

    asyncio.run(run())


def test_end_marker_answers_command_and_closes_window():
    """Trailing "Ende": the command in front of the marker is still answered,
    but the window closes immediately (wake.closed reason=end_command) — a
    follow-up without a wake word is ignored afterwards."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia sag was ende", "und was ist mit morgen"],
        deltas=["Fertig."], mode="conversation")
    llm.gate.set()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)
            await _send_voice(ws, "s1")
            closed, seen = await _drain_until(ws, "wake.closed")
            assert closed is not None and closed.get("reason") == "end_command", seen
            # The marker itself must not reach the LLM; the reply still comes.
            end, seen2 = await _drain_until(ws, "audio.end")
            assert end is not None, seen2
            assert llm.calls and llm.calls[0][-1]["content"] == "sag was"
            await ws.send_json({"type": "playback.done", "turnId": end.get("turnId"),
                                "audioId": end.get("audioId")})
            post = await _collect(ws, 0.15)
            assert "wake.window" not in post, f"end marker reopened: {post}"

            # Follow-up WITHOUT a wake word → ignored (window is closed).
            await _send_voice(ws, "s2")
            ig, seen3 = await _drain_until(ws, "transcript.ignored")
            assert ig is not None, f"Folgefrage nicht ignoriert: {seen3}"
            assert "reply.start" not in seen3, f"Folgefrage löste Antwort aus: {seen3}"
            await ws.close()

    asyncio.run(run())


def test_bare_end_marker_acts_as_stop_command():
    """"Antonia, Ende" (nothing in front of the marker) behaves like a stop
    command: window closed, no reply started."""
    _cfg, llm = _configure_wake(
        stt_texts=["antonia ende"], deltas=["Nie."], mode="conversation")
    llm.gate.set()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _enable_wake(ws)
            await _send_voice(ws, "s1")
            closed, seen = await _drain_until(ws, "wake.closed")
            assert closed is not None and closed.get("reason") == "stop_command", seen
            assert "reply.start" not in seen and not llm.calls
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
    """Counter-check: AFTER the reply finishes (and the guard has lapsed), a
    freshly spoken wake word must reopen the window — otherwise we would have
    over-suppressed."""
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
            await asyncio.sleep(0.25)   # guard (0.1s) safely lapsed

            await _send_voice(ws, "s2")  # fresh wake word → must open
            win, seen = await _drain_until(ws, "wake.window", timeout=2.0)
            assert win is not None, f"frisches Weckwort öffnete NICHT: {seen}"
            await ws.close()

    asyncio.run(run())
