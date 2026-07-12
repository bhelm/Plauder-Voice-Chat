"""Gateway push delivery into the browser: handle_gateway_push speaks
unsolicited agent messages (background task results, cron) like a normal
reply — and queues them while no browser is connected.
"""
import asyncio
import dataclasses

from aiohttp.test_utils import TestClient, TestServer

from plauder import server as srv
from plauder.config import Config
from plauder.sanitizer import HallucinationFilter
from plauder.session import ConversationManager

from tests.conftest import FakeSTT, FakeTTS, FakeLLM, _drain_until


def _configure(streaming=True):
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=30,
                              streaming=streaming)
    conv = ConversationManager(FakeLLM(), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))
    srv._PENDING_PUSHES.clear()
    return cfg


def test_push_is_spoken_on_connected_client():
    _configure()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await srv.handle_gateway_push("Der Task ist fertig.")
            reply, seen, b1 = await _drain_until(ws, "reply")
            assert reply is not None, f"kein reply; gesehen: {seen}"
            assert reply["text"] == "Der Task ist fertig."
            assert reply.get("push") is True
            assert not reply.get("echo")
            assert "reply.start" in seen
            end, seen2, b2 = await _drain_until(ws, "audio.end")
            assert end is not None, f"kein audio.end; gesehen: {seen + seen2}"
            assert (b1 or b2)[:4] == b"VCT2"
            await ws.close()

    asyncio.run(run())


def test_push_without_client_is_queued_and_spoken_on_connect():
    _configure()

    async def run():
        await srv.handle_gateway_push("Nachricht von unterwegs.")
        assert len(srv._PENDING_PUSHES) == 1

        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            reply, seen, _ = await _drain_until(ws, "reply", timeout=5.0)
            assert reply is not None, f"kein reply; gesehen: {seen}"
            assert reply["text"] == "Nachricht von unterwegs."
            assert reply.get("push") is True
            assert len(srv._PENDING_PUSHES) == 0
            await ws.close()

    asyncio.run(run())


def test_empty_push_is_ignored():
    _configure()

    async def run():
        await srv.handle_gateway_push("   ")
        assert len(srv._PENDING_PUSHES) == 0

    asyncio.run(run())
