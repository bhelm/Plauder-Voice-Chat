"""Bridge protocol tests: the gateway-side VoiceBridgeServer (from
hermes_plugin/, gateway-import-free) against the plauder-side
hermes_gateway LLM backend — the real wire compatibility check.
"""
import asyncio
import dataclasses
import sys
from pathlib import Path

import pytest
from aiohttp import ClientSession, WSMsgType

from plauder.backends.base import UpstreamTimeoutError
from plauder.backends.llm.hermes_gateway import HermesGatewayLLMBackend
from plauder.config import Config, ConfigError

# The gateway plugin lives in the repo but outside the plauder package;
# only its gateway-free bridge module is imported here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hermes_plugin"))
from voice_chat.bridge import CLOSE_UNAUTHORIZED, VoiceBridgeServer  # noqa: E402

TOKEN = "test-bridge-token"


def _mk_server(on_user_message=None):
    async def _ignore(chat_id, frame):
        return None

    return VoiceBridgeServer("127.0.0.1", 0, TOKEN,
                             on_user_message=on_user_message or _ignore)


def _mk_backend(port, **kw):
    return HermesGatewayLLMBackend(
        url=f"ws://127.0.0.1:{port}/ws", token=TOKEN, chat_id="default", **kw)


# --------------------------------------------------------------------------- #
# Handshake
# --------------------------------------------------------------------------- #
def test_bad_token_is_rejected():
    async def run():
        server = _mk_server()
        await server.start()
        try:
            async with ClientSession() as session:
                ws = await session.ws_connect(f"ws://127.0.0.1:{server.port}/ws")
                await ws.send_json({"type": "hello", "token": "WRONG",
                                    "chat_id": "default", "proto": 1})
                msg = await ws.receive(timeout=3)
                assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING,
                                    WSMsgType.CLOSED)
                assert ws.close_code == CLOSE_UNAUTHORIZED
                assert not server.connected("default")
        finally:
            await server.stop()

    asyncio.run(run())


def test_non_hello_first_frame_is_rejected():
    async def run():
        server = _mk_server()
        await server.start()
        try:
            async with ClientSession() as session:
                ws = await session.ws_connect(f"ws://127.0.0.1:{server.port}/ws")
                await ws.send_json({"type": "user.message", "text": "hi"})
                msg = await ws.receive(timeout=3)
                assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING,
                                    WSMsgType.CLOSED)
                assert ws.close_code == CLOSE_UNAUTHORIZED
        finally:
            await server.stop()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Turn roundtrip (backend <-> bridge)
# --------------------------------------------------------------------------- #
def test_turn_roundtrip_single_reply():
    async def run():
        received = []
        server_box = {}

        async def on_user_message(chat_id, frame):
            received.append((chat_id, frame))
            srv = server_box["server"]
            await srv.send_frame(chat_id, {
                "type": "agent.message", "text": "Hallo zurück.",
                "turn_id": frame["turn_id"], "push": False})
            await srv.send_frame(chat_id, {
                "type": "turn.done", "turn_id": frame["turn_id"],
                "status": "ok"})

        server = _mk_server(on_user_message)
        server_box["server"] = server
        await server.start()
        backend = _mk_backend(server.port)
        try:
            await backend.load()
            reply = await backend.chat(
                [{"role": "user", "content": "Guten Tag"}])
            assert reply == "Hallo zurück."
            assert backend.last_meta.get("finish_reason") == "stop"
            chat_id, frame = received[0]
            assert chat_id == "default"
            assert frame["text"] == "Guten Tag"
            assert frame["modality"] == "voice"
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_turn_multiple_messages_become_paragraphs():
    async def run():
        server_box = {}

        async def on_user_message(chat_id, frame):
            srv = server_box["server"]
            for part in ("Erster Teil.", "Zweiter Teil."):
                await srv.send_frame(chat_id, {
                    "type": "agent.message", "text": part,
                    "turn_id": frame["turn_id"], "push": False})
            await srv.send_frame(chat_id, {
                "type": "turn.done", "turn_id": frame["turn_id"],
                "status": "ok"})

        server = _mk_server(on_user_message)
        server_box["server"] = server
        await server.start()
        backend = _mk_backend(server.port)
        try:
            await backend.load()
            reply = await backend.chat([{"role": "user", "content": "hi"}])
            assert reply == "Erster Teil.\n\nZweiter Teil."
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_multimodal_last_user_message_sends_text_and_image_urls():
    async def run():
        frames = []
        server_box = {}

        async def on_user_message(chat_id, frame):
            frames.append(frame)
            await server_box["server"].send_frame(chat_id, {
                "type": "turn.done", "turn_id": frame["turn_id"],
                "status": "ok"})

        server = _mk_server(on_user_message)
        server_box["server"] = server
        await server.start()
        backend = _mk_backend(server.port)
        try:
            await backend.load()
            reply = await backend.chat([
                {"role": "user", "content": "alter Kontext"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [
                    {"type": "text", "text": "Schau mal"},
                    {"type": "image_url",
                     "image_url": {"url": "http://x/uploads/a.jpg"}},
                ]},
            ])
            assert reply == ""             # silent turn -> empty reply
            assert len(frames) == 1        # only the LAST user message went out
            assert frames[0]["text"] == "Schau mal"
            assert frames[0]["image_urls"] == ["http://x/uploads/a.jpg"]
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_turn_timeout_raises_upstream_timeout():
    async def run():
        server = _mk_server()      # never answers
        await server.start()
        backend = _mk_backend(server.port, timeout=1)
        try:
            await backend.load()
            with pytest.raises(UpstreamTimeoutError):
                await backend.chat([{"role": "user", "content": "hi"}])
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_chat_fails_fast_when_no_bridge_running():
    async def run():
        import plauder.backends.llm.hermes_gateway as hg
        backend = HermesGatewayLLMBackend(
            url="ws://127.0.0.1:1/ws", token=TOKEN)   # nothing listens there
        orig = hg.CONNECT_WAIT_S
        hg.CONNECT_WAIT_S = 0.3
        try:
            await backend.load()
            with pytest.raises(RuntimeError, match="not connected"):
                await backend.chat([{"role": "user", "content": "hi"}])
        finally:
            hg.CONNECT_WAIT_S = orig
            await backend.close()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Push path
# --------------------------------------------------------------------------- #
def test_push_frame_reaches_handler():
    async def run():
        server = _mk_server()
        await server.start()
        backend = _mk_backend(server.port)
        got = asyncio.Queue()

        async def on_push(text):
            await got.put(text)

        backend.set_push_handler(on_push)
        try:
            await backend.load()
            # Wait until the client is registered on the server side.
            for _ in range(100):
                if server.connected("default"):
                    break
                await asyncio.sleep(0.05)
            assert server.connected("default")
            ok = await server.send_frame("default", {
                "type": "agent.message", "text": "Task fertig!",
                "turn_id": None, "push": True})
            assert ok
            text = await asyncio.wait_for(got.get(), timeout=3)
            assert text == "Task fertig!"
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_push_buffered_until_handler_registered():
    async def run():
        server = _mk_server()
        await server.start()
        backend = _mk_backend(server.port)
        try:
            await backend.load()
            for _ in range(100):
                if server.connected("default"):
                    break
                await asyncio.sleep(0.05)
            await server.send_frame("default", {
                "type": "agent.message", "text": "früher Push",
                "turn_id": None, "push": True})
            await asyncio.sleep(0.2)       # frame lands in _pending_pushes
            got = asyncio.Queue()

            async def on_push(text):
                await got.put(text)

            backend.set_push_handler(on_push)      # flushes the buffer
            text = await asyncio.wait_for(got.get(), timeout=3)
            assert text == "früher Push"
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_queued_frames_flush_on_connect():
    async def run():
        server = _mk_server()
        await server.start()
        # Queue BEFORE any client exists (voice chat down during delivery).
        server.queue_frame("default", {
            "type": "agent.message", "text": "verspätete Lieferung",
            "turn_id": None, "push": True})
        backend = _mk_backend(server.port)
        got = asyncio.Queue()

        async def on_push(text):
            await got.put(text)

        backend.set_push_handler(on_push)
        try:
            await backend.load()
            text = await asyncio.wait_for(got.get(), timeout=5)
            assert text == "verspätete Lieferung"
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_config_hermes_gateway_requires_token():
    cfg = dataclasses.replace(Config.from_env(), llm_backend="hermes_gateway",
                              hermes_gateway_token="")
    with pytest.raises(ConfigError, match="HERMES_GATEWAY_TOKEN"):
        cfg.validate()


def test_config_hermes_gateway_valid_with_token():
    cfg = dataclasses.replace(Config.from_env(), llm_backend="hermes_gateway",
                              hermes_gateway_token="secret")
    cfg.validate()


def test_factory_builds_hermes_gateway_backend():
    from plauder.backends import LLMBackend
    cfg = dataclasses.replace(Config.from_env(), llm_backend="hermes_gateway",
                              hermes_gateway_token="secret")
    backend = LLMBackend.from_config(cfg)
    assert isinstance(backend, HermesGatewayLLMBackend)
    assert backend.describe()["engine"] == "hermes_gateway"
