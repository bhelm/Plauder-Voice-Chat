"""HTTP-Routen, /healthz und WebSocket-Handler — mit Mock-Backends."""
import asyncio
import dataclasses

from aiohttp.test_utils import TestClient, TestServer

from plauder import server as srv
from plauder.config import Config
from plauder.sanitizer import HallucinationFilter
from plauder.session import ConversationManager


# --- Mock-Backends ----------------------------------------------------------
class FakeSTT:
    last_no_speech_prob = None

    async def transcribe(self, audio_pcm, sample_rate):
        return "hallo welt"

    def describe(self):
        return {"engine": "fake-stt", "loaded": True}


class FakeTTS:
    sample_rate = 24000

    async def synth(self, text, *, speed=1.0):
        # 4 int16-Samples
        return b"\x00\x00\x01\x00\x02\x00\x03\x00", 24000

    def describe(self):
        return {"engine": "fake-tts", "sample_rate": 24000, "loaded": True}


class FakeLLM:
    loaded = True
    last_meta = {"finish_reason": "stop", "usage": {"total_tokens": 5}}

    def __init__(self, reply="Hallo, ich bin Antonia."):
        self.reply = reply

    async def chat(self, messages, system_hint=None):
        return self.reply

    def describe(self):
        return {"engine": "fake-llm", "model": "fake", "ready": True}


class StreamingFakeLLM:
    """Fake mit echtem chat_stream (mehrere Deltas)."""
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
    # NO_REPLY über mehrere Deltas verteilt → kein Audio, reply.silent.
    _configure_streaming_llm(["NO_", "REPLY"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "text.message", "text": "egal"})
            silent, seen, _ = await _drain_until(ws, "reply.silent")
            assert silent is not None, f"kein reply.silent; gesehen: {seen}"
            assert silent["reason"] == "no_reply"
            assert "audio.start" not in seen   # keine Sprachausgabe für NO_REPLY
            await ws.close()

    asyncio.run(run())


def _configure(reply="Hallo, ich bin Antonia."):
    cfg = Config.from_env()
    cfg = dataclasses.replace(cfg, debounce_ms=30)  # schnelle Tests
    conv = ConversationManager(FakeLLM(reply), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))
    return cfg


async def _drain_until(ws, want_type, *, timeout=3.0):
    """Liest WS-Frames, bis ein JSON-Frame mit type==want_type kommt.
    Sammelt alle gesehenen Typen + ggf. erstes Binary."""
    seen = []
    binary = None
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=max(0.05, deadline - loop.time()))
        except asyncio.TimeoutError:
            break
        if msg.type.name == "BINARY":
            binary = msg.data
            seen.append("__binary__")
            continue
        if msg.type.name in ("CLOSE", "CLOSING", "CLOSED", "ERROR"):
            break
        data = msg.json()
        seen.append(data.get("type"))
        if data.get("type") == want_type:
            return data, seen, binary
    return None, seen, binary


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
            # Streaming-TTS: audio.start + ≥1 PCM-Chunk (VCT2-Binary) + audio.end
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
    """STREAMING=0: klassischer Pfad — audio.meta + ein VCT1-WAV-Frame."""
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
            # WAV-Frame (VCT1) folgt nach audio.meta
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
