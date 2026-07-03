"""HTTP routes, /healthz and WebSocket handler — with mock backends."""
import asyncio
import dataclasses
from unittest.mock import AsyncMock, patch

from aiohttp.test_utils import TestClient, TestServer

from plauder import server as srv
from plauder.config import Config
from plauder.sanitizer import HallucinationFilter
from plauder.session import ConversationManager

# Shared duck-typed backends + WS drain helper live in conftest.
from tests.conftest import FakeSTT, FakeTTS, FakeLLM, _drain_until


# --- Mock-Backends ----------------------------------------------------------
class StreamingFakeLLM:
    """Fake with a real chat_stream (multiple deltas)."""
    loaded = True

    def __init__(self, deltas):
        self.deltas = list(deltas)
        self.last_meta = {"finish_reason": "stop", "usage": {"total_tokens": 9}}

    async def chat(self, messages, system_hint=None):
        return "".join(self.deltas)

    async def chat_stream(self, messages, system_hint=None):
        for d in self.deltas:
            yield d
        self.last_meta = {"finish_reason": "stop", "usage": {"total_tokens": 9}}

    def describe(self):
        return {"engine": "fake-stream-llm", "model": "fake", "ready": True}


def _configure_streaming_llm(deltas):
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=30, streaming=True)
    conv = ConversationManager(StreamingFakeLLM(deltas), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))
    return cfg


def test_ws_streaming_multi_sentence_pipelines_audio():
    _configure_streaming_llm(["Hallo. ", "Wie geht ", "es dir? ", "Alles gut."])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "text.message", "text": "Hi"})
            reply, seen, b1 = await _drain_until(ws, "reply")
            assert reply["text"] == "Hallo. Wie geht es dir? Alles gut."
            assert seen.count("reply.delta") >= 2   # live Token-Deltas
            end, seen2, b2 = await _drain_until(ws, "audio.end")
            assert end is not None
            assert end["chunks"] >= 1
            assert (b1 or b2)[:4] == b"VCT2"
            await ws.close()

    asyncio.run(run())


def test_ws_streaming_no_reply_across_deltas_is_silent():
    # NO_REPLY spread across multiple deltas → no audio, reply.silent.
    _configure_streaming_llm(["NO_", "REPLY"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "text.message", "text": "egal"})
            silent, seen, _ = await _drain_until(ws, "reply.silent")
            assert silent is not None, f"kein reply.silent; gesehen: {seen}"
            assert silent["reason"] == "no_reply"
            assert "audio.start" not in seen   # no speech output for NO_REPLY
            await ws.close()

    asyncio.run(run())


def _configure(reply="Hallo, ich bin Antonia."):
    cfg = Config.from_env()
    cfg = dataclasses.replace(cfg, debounce_ms=30)  # fast tests
    conv = ConversationManager(FakeLLM(reply), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))
    return cfg


def test_healthz_reports_active_backends():
    _configure()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            resp = await client.get("/healthz")
            assert resp.status == 200
            body = await resp.json()
            assert body["ok"] is True
            assert body["backends"]["stt"] == "openai"
            assert body["backends"]["llm"] == "openai_compat"
            assert body["stt"]["engine"] == "fake-stt"
            assert body["agent"]["engine"] == "fake-llm"
            assert body["tts"]["engine"] == "fake-tts"

    asyncio.run(run())


def test_index_served():
    _configure()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            resp = await client.get("/")
            assert resp.status == 200

    asyncio.run(run())


def test_ws_hello_frame():
    _configure()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello = await asyncio.wait_for(ws.receive_json(), timeout=3)
            assert hello["type"] == "hello"
            assert hello["agent_name"] == "Antonia"
            assert hello["stt"]["engine"] == "fake-stt"
            assert "vad" in hello["turn"]
            await ws.close()

    asyncio.run(run())


def test_ws_text_message_full_pipeline():
    _configure(reply="Das ist die Antwort.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "text.message", "text": "Hallo Antonia"})
            reply, seen, binary1 = await _drain_until(ws, "reply")
            assert reply is not None, f"kein reply-Frame; gesehen: {seen}"
            assert reply["text"] == "Das ist die Antwort."
            assert reply.get("streamed") is True
            assert "turn.commit" in seen
            assert "reply.start" in seen
            assert "reply.delta" in seen
            # Streaming TTS: audio.start + ≥1 PCM chunk (VCT2 binary) + audio.end
            end, seen2, binary2 = await _drain_until(ws, "audio.end")
            assert end is not None, f"kein audio.end; gesehen: {seen2}"
            assert "audio.start" in seen + seen2
            binary = binary1 or binary2
            assert binary is not None, "kein PCM-Chunk empfangen"
            assert binary[:4] == b"VCT2"
            assert end["chunks"] >= 1
            await ws.close()

    asyncio.run(run())


def test_ws_text_message_non_streaming_fallback():
    """STREAMING=0: classic path — audio.meta + one VCT1 WAV frame."""
    cfg = Config.from_env()
    cfg = dataclasses.replace(cfg, debounce_ms=30, streaming=False)
    conv = ConversationManager(FakeLLM("Hallo da."), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "text.message", "text": "Hallo Antonia"})
            meta, seen, binary = await _drain_until(ws, "audio.meta")
            assert meta is not None, f"kein audio.meta; gesehen: {seen}"
            assert "reply" in seen
            assert meta.get("framed") is True
            # WAV frame (VCT1) follows after audio.meta
            _, _, binary2 = await _drain_until(ws, "__never__", timeout=1.0)
            wav = binary or binary2
            assert wav is not None and wav[:4] == b"VCT1"
            await ws.close()

    asyncio.run(run())


def test_ws_no_reply_yields_silent():
    _configure(reply="NO_REPLY")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "text.message", "text": "egal"})
            silent, seen, _ = await _drain_until(ws, "reply.silent")
            assert silent is not None, f"kein reply.silent; gesehen: {seen}"
            assert silent["reason"] == "no_reply"
            await ws.close()

    asyncio.run(run())


def test_ws_settings_ack():
    _configure()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "speed": 1.2, "debounceMs": 800})
            ack, _, _ = await _drain_until(ws, "settings.ack")
            assert ack["speed"] == 1.2
            assert ack["debounceMs"] == 800
            await ws.close()

    asyncio.run(run())


def test_ws_session_reset_ack():
    _configure()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "session.reset"})
            ack, _, _ = await _drain_until(ws, "session.reset.ack")
            assert ack["sessionUser"].startswith("voice-user-")
            await ws.close()

    asyncio.run(run())


def test_upload_rejects_non_image():
    _configure()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            resp = await client.post("/upload", data={"file": ("x.txt", b"hi", "text/plain")})
            assert resp.status == 400

    asyncio.run(run())


# --- APP_LANGUAGE / locale surfaced to the client ---------------------------
def _configure_lang(lang):
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=30, app_language=lang)
    conv = ConversationManager(FakeLLM(), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))
    return cfg


def test_ws_hello_advertises_lang_en():
    _configure_lang("en")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello, _, _ = await _drain_until(ws, "hello")
            assert hello["lang"] == "en"
            await ws.close()

    asyncio.run(run())


def test_ws_hello_advertises_lang_de():
    _configure_lang("de")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello, _, _ = await _drain_until(ws, "hello")
            assert hello["lang"] == "de"
            await ws.close()

    asyncio.run(run())


def test_index_injects_lang_en():
    _configure_lang("en")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            resp = await client.get("/")
            assert resp.status == 200
            html = await resp.text()
            assert '<html lang="en">' in html
            assert "__APP_LANG__" not in html

    asyncio.run(run())


def test_index_injects_lang_de():
    _configure_lang("de")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            resp = await client.get("/")
            assert resp.status == 200
            html = await resp.text()
            assert '<html lang="de">' in html
            assert "__APP_LANG__" not in html

    asyncio.run(run())


# --- Sub-path (BASE_PATH) support -------------------------------------------
def _configure_base(base):
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=30, base_path=base)
    conv = ConversationManager(FakeLLM(), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))
    return cfg


def test_base_path_default_root_unprefixed():
    """Default (no BASE_PATH): everything at root, asset URLs unprefixed."""
    _configure_base("")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            resp = await client.get("/")
            assert resp.status == 200
            html = await resp.text()
            assert "__BASE_PATH__" not in html
            assert 'src="/static/vendor/ort.js"' in html

    asyncio.run(run())


def test_base_path_serves_under_prefix():
    """BASE_PATH=/voice: routes live under /voice, root 404s, assets prefixed."""
    _configure_base("/voice")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            resp = await client.get("/voice/")
            assert resp.status == 200
            html = await resp.text()
            assert "__BASE_PATH__" not in html
            assert 'src="/voice/static/vendor/ort.js"' in html
            assert (await client.get("/")).status == 404          # not at root
            assert (await client.get("/voice")).status == 200     # no trailing slash
            assert (await client.get("/voice/healthz")).status == 200

    asyncio.run(run())


def test_ws_hello_advertises_base_path():
    _configure_base("/voice")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/voice/ws")
            hello, _, _ = await _drain_until(ws, "hello")
            assert hello["basePath"] == "/voice"
            await ws.close()

    asyncio.run(run())


# --- Connection task lifecycle (B1: tracked detached handlers) --------------
def test_spawn_tracked_tracks_then_autodiscards():
    """_spawn_tracked registers the task and the done-callback removes it."""
    async def run():
        state = srv.TurnState()

        async def quick():
            return 1

        task = srv._spawn_tracked(state, quick())
        assert task in state.inflight_tasks
        await task
        await asyncio.sleep(0)          # let the done-callback run
        assert task not in state.inflight_tasks

    asyncio.run(run())


def test_cancel_connection_tasks_cancels_everything():
    """The WS-close cleanup cancels debounce, agent, legacy text AND the
    tracked segment/partial handlers (otherwise they could send on a closed ws)."""
    async def run():
        state = srv.TurnState()

        async def forever():
            await asyncio.sleep(3600)

        tracked = srv._spawn_tracked(state, forever())
        state.debounce_task = asyncio.create_task(forever())
        state.agent_task = asyncio.create_task(forever())
        text_task = asyncio.create_task(forever())
        state.text_tasks.append(text_task)
        await asyncio.sleep(0)          # let them start

        srv._cancel_connection_tasks(state)

        for task in (tracked, state.debounce_task, state.agent_task, text_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert task.cancelled()

    asyncio.run(run())


# --- Hermes backend history loading ----------------------------------------
_MOCK_HISTORY = [
    {"role": "user", "content": "Hallo"},
    {"role": "assistant", "content": "Hi, wie geht es dir?"},
]


def test_ws_history_frame_sent_on_connect():
    """When fetch_history returns messages, a 'history' frame is sent after hello
    and the ConversationManager is seeded. The fetch only runs when a Hermes
    session key is configured (without one it would probe a non-Hermes LLM
    endpoint with a real HTTP call on every connect)."""
    cfg = _configure()
    srv.configure(dataclasses.replace(cfg, hermes_session_key_separate="agent:test:voice"),
                  stt=srv.STT, tts=srv.TTS, conv=srv.CONV, bridge=None, ghost=srv.GHOST)

    async def run():
        mock = AsyncMock(return_value=list(_MOCK_HISTORY))
        with patch("plauder.server.fetch_history", mock):
            async with TestClient(TestServer(srv.build_app())) as client:
                ws = await client.ws_connect("/ws")
                hello = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert hello["type"] == "hello"
                hist = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert hist["type"] == "history"
                assert len(hist["messages"]) == 2
                assert hist["messages"][0]["role"] == "user"
                assert hist["messages"][1]["content"] == "Hi, wie geht es dir?"
                # ConversationManager should have been seeded.
                user_key = hello["agent"]["user_id"]
                local = srv.CONV.history_for(user_key)
                assert len(local) == 2
                assert local[0]["content"] == "Hallo"
                await ws.close()

    asyncio.run(run())


def test_ws_no_history_frame_when_empty():
    """When fetch_history returns [], no history frame is sent."""
    _configure()

    async def run():
        mock = AsyncMock(return_value=[])
        with patch("plauder.server.fetch_history", mock):
            async with TestClient(TestServer(srv.build_app())) as client:
                ws = await client.ws_connect("/ws")
                hello = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert hello["type"] == "hello"
                # Send a ping and check we get ack, not history.
                await ws.send_json({"type": "ping"})
                ack, seen, _ = await _drain_until(ws, "ack")
                assert ack is not None
                assert "history" not in seen
                await ws.close()

    asyncio.run(run())


def test_ws_history_fetch_failure_does_not_break_connection():
    """If fetch_history raises, the WS connection still works."""
    _configure()

    async def run():
        mock = AsyncMock(side_effect=RuntimeError("network down"))
        with patch("plauder.server.fetch_history", mock):
            async with TestClient(TestServer(srv.build_app())) as client:
                ws = await client.ws_connect("/ws")
                hello = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert hello["type"] == "hello"
                # Connection should still be functional.
                await ws.send_json({"type": "ping"})
                ack, seen, _ = await _drain_until(ws, "ack")
                assert ack is not None
                await ws.close()

    asyncio.run(run())


def test_no_history_fetch_without_hermes_key():
    """Regression: without a configured Hermes session key, connecting must NOT
    call fetch_history at all — it would fire a real, authenticated HTTP request
    against a non-Hermes LLM endpoint on every connect."""
    _configure()   # Config.from_env() with no HERMES_SESSION_KEY_SEPARATE

    async def run():
        mock = AsyncMock(return_value=list(_MOCK_HISTORY))
        with patch("plauder.server.fetch_history", mock):
            async with TestClient(TestServer(srv.build_app())) as client:
                ws = await client.ws_connect("/ws")
                hello = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert hello["type"] == "hello"
                await ws.send_json({"type": "ping"})
                nxt = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert nxt["type"] == "ack"
                await ws.close()
        assert mock.await_count == 0

    asyncio.run(run())
