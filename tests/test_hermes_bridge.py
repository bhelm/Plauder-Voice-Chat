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
from voice_chat.bridge import (  # noqa: E402
    CLOSE_UNAUTHORIZED,
    UNDELIVERED_MAX,
    VoiceBridgeServer,
    format_undelivered_note,
)

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
# Streaming (agent.partial) path
# --------------------------------------------------------------------------- #
def test_streaming_partials_yield_suffix_deltas():
    async def run():
        server_box = {}

        async def on_user_message(chat_id, frame):
            srv = server_box["server"]
            tid = frame["turn_id"]
            # Gateway stream consumer: initial send(), then growing edits,
            # then a finalize edit, then turn.done.
            await srv.send_frame(chat_id, {
                "type": "agent.message", "text": "Hallo",
                "turn_id": tid, "message_id": "m1", "push": False})
            await srv.send_frame(chat_id, {
                "type": "agent.partial", "turn_id": tid, "message_id": "m1",
                "text": "Hallo Bernd.", "finalize": False})
            await srv.send_frame(chat_id, {
                "type": "agent.partial", "turn_id": tid, "message_id": "m1",
                "text": "Hallo Bernd. Wie geht's?", "finalize": True})
            await srv.send_frame(chat_id, {
                "type": "turn.done", "turn_id": tid, "status": "ok"})

        server = _mk_server(on_user_message)
        server_box["server"] = server
        await server.start()
        backend = _mk_backend(server.port)
        try:
            await backend.load()
            deltas = [d async for d in backend.chat_stream(
                [{"role": "user", "content": "hi"}])]
            assert deltas == ["Hallo", " Bernd.", " Wie geht's?"]
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_streaming_reformatted_finalize_is_not_respoken():
    async def run():
        server_box = {}

        async def on_user_message(chat_id, frame):
            srv = server_box["server"]
            tid = frame["turn_id"]
            await srv.send_frame(chat_id, {
                "type": "agent.partial", "turn_id": tid, "message_id": "m1",
                "text": "Der **Plan** steht.", "finalize": False})
            # finalize reformats (markdown cleanup) -> prefix mismatch
            await srv.send_frame(chat_id, {
                "type": "agent.partial", "turn_id": tid, "message_id": "m1",
                "text": "Der Plan steht.", "finalize": True})
            await srv.send_frame(chat_id, {
                "type": "turn.done", "turn_id": tid, "status": "ok"})

        server = _mk_server(on_user_message)
        server_box["server"] = server
        await server.start()
        backend = _mk_backend(server.port)
        try:
            await backend.load()
            deltas = [d async for d in backend.chat_stream(
                [{"role": "user", "content": "hi"}])]
            assert deltas == ["Der **Plan** steht."]
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_streaming_segment_break_second_message_is_new_paragraph():
    async def run():
        server_box = {}

        async def on_user_message(chat_id, frame):
            srv = server_box["server"]
            tid = frame["turn_id"]
            await srv.send_frame(chat_id, {
                "type": "agent.message", "text": "Ich schaue nach.",
                "turn_id": tid, "message_id": "m1", "push": False})
            await srv.send_frame(chat_id, {
                "type": "agent.message", "text": "Fertig: alles gut.",
                "turn_id": tid, "message_id": "m2", "push": False})
            await srv.send_frame(chat_id, {
                "type": "turn.done", "turn_id": tid, "status": "ok"})

        server = _mk_server(on_user_message)
        server_box["server"] = server
        await server.start()
        backend = _mk_backend(server.port)
        try:
            await backend.load()
            reply = await backend.chat([{"role": "user", "content": "hi"}])
            assert reply == "Ich schaue nach.\n\nFertig: alles gut."
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_orphan_finalize_partial_becomes_push():
    async def run():
        server = _mk_server()
        await server.start()
        backend = _mk_backend(server.port)
        got = asyncio.Queue()

        async def on_push(text, speak=True):
            await got.put(text)

        backend.set_push_handler(on_push)
        try:
            await backend.load()
            for _ in range(100):
                if server.connected("default"):
                    break
                await asyncio.sleep(0.05)
            # finalize partial for a turn the client never had -> spoken late
            await server.send_frame("default", {
                "type": "agent.partial", "turn_id": "gone", "message_id": "m9",
                "text": "Späte Antwort.", "finalize": True})
            # mid-stream orphan partial -> silently dropped
            await server.send_frame("default", {
                "type": "agent.partial", "turn_id": "gone", "message_id": "m9",
                "text": "Späte Antwort. Mehr.", "finalize": False})
            text = await asyncio.wait_for(got.get(), timeout=3)
            assert text == "Späte Antwort."
            assert got.empty()
        finally:
            await backend.close()
            await server.stop()

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

        async def on_push(text, speak=True):
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

            async def on_push(text, speak=True):
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

        async def on_push(text, speak=True):
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


def test_streamed_orphan_push_spoken_once():
    """Regression (12.07. OOM incident): a streamed reply whose turn died
    arrives as initial chunk + partial growth + finalize — it must be
    dispatched ONCE with the final text, not chunk-by-chunk ("Session" …
    then the full sentence again)."""
    async def run():
        server = _mk_server()
        await server.start()
        backend = _mk_backend(server.port)
        got = asyncio.Queue()

        async def on_push(text, speak=True):
            await got.put((text, speak))

        backend.set_push_handler(on_push)
        try:
            await backend.load()
            for _ in range(100):
                if server.connected("default"):
                    break
                await asyncio.sleep(0.05)
            await server.send_frame("default", {
                "type": "agent.message", "text": "Session",
                "turn_id": None, "push": True, "message_id": "m1"})
            await asyncio.sleep(0.3)   # inside the debounce window
            await server.send_frame("default", {
                "type": "agent.partial", "turn_id": None, "message_id": "m1",
                "text": "Session wiederhergestellt.", "finalize": False})
            await server.send_frame("default", {
                "type": "agent.partial", "turn_id": None, "message_id": "m1",
                "text": "Session wiederhergestellt, alles online.",
                "finalize": True})
            text, speak = await asyncio.wait_for(got.get(), timeout=3)
            assert text == "Session wiederhergestellt, alles online."
            assert speak is True
            await asyncio.sleep(1.5)   # a full debounce window later …
            assert got.empty()         # … still exactly ONE dispatch
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_speak_false_push_flag_reaches_handler():
    async def run():
        server = _mk_server()
        await server.start()
        backend = _mk_backend(server.port)
        got = asyncio.Queue()

        async def on_push(text, speak=True):
            await got.put((text, speak))

        backend.set_push_handler(on_push)
        try:
            await backend.load()
            for _ in range(100):
                if server.connected("default"):
                    break
                await asyncio.sleep(0.05)
            await server.send_frame("default", {
                "type": "agent.message", "text": "♻️ Gateway online",
                "turn_id": None, "push": True, "speak": False,
                "system": True, "message_id": "sys1"})
            text, speak = await asyncio.wait_for(got.get(), timeout=3)
            assert text == "♻️ Gateway online"
            assert speak is False
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Session reset ("New Session" button -> gateway /new)
# --------------------------------------------------------------------------- #
def test_reset_session_frame_reaches_server_callback():
    async def run():
        resets = asyncio.Queue()

        async def on_session_reset(chat_id):
            await resets.put(chat_id)

        async def on_user_message(chat_id, frame):
            return None

        server = VoiceBridgeServer("127.0.0.1", 0, TOKEN,
                                   on_user_message=on_user_message,
                                   on_session_reset=on_session_reset)
        await server.start()
        backend = _mk_backend(server.port)
        try:
            await backend.load()
            for _ in range(100):
                if server.connected("default"):
                    break
                await asyncio.sleep(0.05)
            await backend.reset_session()
            chat_id = await asyncio.wait_for(resets.get(), timeout=3)
            assert chat_id == "default"
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_reset_session_without_connection_is_noop():
    async def run():
        backend = HermesGatewayLLMBackend(url="ws://127.0.0.1:1/ws",
                                          token=TOKEN)
        await backend.reset_session()   # must not raise

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# HTTP /push endpoint (out-of-process senders: hermes send, standalone cron)
# --------------------------------------------------------------------------- #
def test_http_push_delivers_or_queues():
    async def run():
        server = _mk_server()
        await server.start()
        base = f"http://127.0.0.1:{server.port}"
        backend = _mk_backend(server.port)
        got = asyncio.Queue()

        async def on_push(text, speak=True):
            await got.put(text)

        backend.set_push_handler(on_push)
        try:
            async with ClientSession() as http:
                # Bad token -> 401
                async with http.post(f"{base}/push",
                                     json={"text": "x"},
                                     headers={"X-Bridge-Token": "WRONG"}) as r:
                    assert r.status == 401
                # No client yet -> queued
                async with http.post(f"{base}/push",
                                     json={"text": "aus der Ferne"},
                                     headers={"X-Bridge-Token": TOKEN}) as r:
                    assert r.status == 200
                    body = await r.json()
                    assert body["success"] and body["queued"]
                # Client connects -> queued push is flushed and spoken
                await backend.load()
                text = await asyncio.wait_for(got.get(), timeout=5)
                assert text == "aus der Ferne"
                # Live client -> delivered directly (queued: false)
                async with http.post(f"{base}/push",
                                     json={"text": "direkt"},
                                     headers={"X-Bridge-Token": TOKEN}) as r:
                    body = await r.json()
                    assert body["success"] and not body["queued"]
                text = await asyncio.wait_for(got.get(), timeout=5)
                assert text == "direkt"
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Undelivered-push notes (barge-in cancelled a push before it was heard)
# --------------------------------------------------------------------------- #
def test_notify_push_undelivered_reaches_bridge_store():
    """The plauder backend's notify_push_undelivered frame lands in the
    bridge's per-chat pending store and is consumed one-shot."""
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
            assert server.connected("default")
            await backend.notify_push_undelivered("Der Subagent ist fertig.", 1.2)
            # Poll until the frame has been routed into the store.
            pending = []
            for _ in range(100):
                pending = server.consume_undelivered("default")
                if pending:
                    break
                await asyncio.sleep(0.02)
            assert len(pending) == 1
            assert pending[0]["text"] == "Der Subagent ist fertig."
            assert pending[0]["played_s"] == 1.2
            # One-shot: a second consume returns nothing.
            assert server.consume_undelivered("default") == []
        finally:
            await backend.close()
            await server.stop()

    asyncio.run(run())


def test_notify_push_undelivered_without_connection_is_noop():
    async def run():
        backend = HermesGatewayLLMBackend(url="ws://127.0.0.1:1/ws",
                                          token=TOKEN)
        await backend.notify_push_undelivered("egal", 0.0)   # must not raise

    asyncio.run(run())


def test_undelivered_store_is_bounded_fifo():
    """More than UNDELIVERED_MAX notes: the oldest are dropped, newest kept."""
    server = _mk_server()
    for i in range(UNDELIVERED_MAX + 3):
        server._record_undelivered("default", {"text": f"n{i}", "played_s": 0.0})
    pending = server.consume_undelivered("default")
    assert len(pending) == UNDELIVERED_MAX
    texts = [p["text"] for p in pending]
    assert texts == [f"n{i}" for i in range(3, UNDELIVERED_MAX + 3)]
    assert server.consume_undelivered("default") == []


def test_record_undelivered_ignores_empty_text():
    server = _mk_server()
    server._record_undelivered("default", {"text": "   ", "played_s": 5.0})
    server._record_undelivered("default", {"played_s": 1.0})
    assert server.consume_undelivered("default") == []


def test_format_undelivered_note_is_pure_and_full_text():
    long = "Der Subagent hat 12 Dateien geändert und alle Tests laufen grün. " * 3
    note = format_undelivered_note([
        {"text": "Kurze Info.", "played_s": 0.0},
        {"text": long, "played_s": 2.6},
    ])
    assert note.startswith("HINWEIS:")
    assert "»Kurze Info.«" in note
    assert "(nicht vorgelesen)" in note      # 0 s
    assert "(nur 3s vorgelesen)" in note      # 2.6 -> 3
    assert long.strip() in note               # full text, no truncation
    # Pure: empty in -> empty out.
    assert format_undelivered_note([]) == ""
    assert format_undelivered_note([{"text": "  ", "played_s": 1.0}]) == ""


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
