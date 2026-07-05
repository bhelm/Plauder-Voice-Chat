"""Ghost filter vs. barge-in on the plain-VAD path (no wake word, no speaker
lock): the in-flight cancel is deferred past STT + the hallucination filter,
so a Whisper ghost ("Untertitelung des ZDF, 2020" from mic noise) must NOT
cancel a turn that is still thinking. Real speech still barges in.
"""
import asyncio
import dataclasses

from aiohttp.test_utils import TestClient, TestServer

from plauder import server as srv
from plauder.config import Config
from plauder.sanitizer import HallucinationFilter
from plauder.session import ConversationManager

from test_wake_pipeline import (GatedStreamingLLM, ScriptedSTT, FakeTTS,
                                _drain_until, _collect, _send_voice)


def _configure_plain(stt_texts, deltas):
    """Plain VAD mode: ghost filter ON, no wake gating, no speaker lock."""
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=20, streaming=True)
    llm = GatedStreamingLLM(deltas)
    conv = ConversationManager(llm, system_prompt="sys")
    srv.configure(cfg, stt=ScriptedSTT(stt_texts), tts=FakeTTS(), conv=conv,
                  bridge=None, ghost=HallucinationFilter(enabled=True))
    return cfg, llm


def test_ghost_segment_does_not_cancel_thinking_turn():
    """A hallucinated segment arriving while the reply is in flight is filtered
    AND leaves the turn untouched (no turn.discarded); the reply finishes."""
    _cfg, llm = _configure_plain(
        stt_texts=["erzähl eine geschichte", "Untertitelung des ZDF, 2020"],
        deltas=["Erster Teil. ", "Zweiter Teil."])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")                    # real question → turn
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            # Mic noise becomes a credit-roll ghost while Antonia is thinking.
            await _send_voice(ws, "s2")
            tr, seen = await _drain_until(ws, "transcript")
            assert tr is not None and tr.get("filtered") == "hallucination", seen
            assert "turn.discarded" not in seen, f"Ghost hat unterbrochen: {seen}"
            seen2 = await _collect(ws, 0.3)
            assert "turn.discarded" not in seen2, f"Ghost hat unterbrochen: {seen2}"

            llm.gate.set()                                  # release the reply
            reply, seen3 = await _drain_until(ws, "reply")
            assert reply is not None, f"Antwort abgebrochen? {seen3}"
            assert reply["text"] == "Erster Teil. Zweiter Teil."
            await ws.close()

    asyncio.run(run())


def test_real_speech_still_cancels_thinking_turn():
    """The deferral must not break the coalescing barge-in: genuine speech
    during a thinking turn still discards it and starts a fresh turn."""
    _cfg, llm = _configure_plain(
        stt_texts=["erzähl eine geschichte", "warte, noch eine frage"],
        deltas=["Erster Teil. ", "Zweiter Teil."])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            await _send_voice(ws, "s2")                    # real follow-up
            disc, seen = await _drain_until(ws, "turn.discarded")
            assert disc is not None, f"echte Sprache hat NICHT unterbrochen: {seen}"

            llm.gate.set()                                  # release turn 2
            reply, seen2 = await _drain_until(ws, "reply")
            assert reply is not None, f"keine neue Antwort: {seen2}"
            await ws.close()

    asyncio.run(run())
