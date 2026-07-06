"""Opus audio compression: framing, negotiation, uplink decode, downlink encode.

The framing/negotiation tests run everywhere (availability is injected).
Tests marked with ``needs_opus`` exercise the real codec (opuslib + system
libopus) and skip cleanly when it is not installed.
"""
import asyncio
import dataclasses

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from plauder import audio as audio_utils
from plauder import opus_codec
from plauder import server as srv
from plauder.config import Config
from plauder.sanitizer import HallucinationFilter
from plauder.session import ConversationManager

from tests.conftest import FakeSTT, FakeTTS, FakeLLM, _drain_until

needs_opus = pytest.mark.skipif(
    not opus_codec.is_available(), reason="opuslib/libopus not installed")


def _configure(reply="Hallo, ich bin Antonia.", **cfg_overrides):
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=30, **cfg_overrides)
    conv = ConversationManager(FakeLLM(reply), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False))
    return cfg


async def _drain_collect(ws, want_type, *, timeout=3.0):
    """Like conftest._drain_until, but collects EVERY binary frame."""
    seen, binaries = [], []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            msg = await asyncio.wait_for(
                ws.receive(), timeout=max(0.05, deadline - loop.time()))
        except asyncio.TimeoutError:
            break
        if msg.type.name == "BINARY":
            binaries.append(msg.data)
            seen.append("__binary__")
            continue
        if msg.type.name in ("CLOSE", "CLOSING", "CLOSED", "ERROR"):
            break
        data = msg.json()
        seen.append(data)
        if data.get("type") == want_type:
            return data, seen, binaries
    return None, seen, binaries


# --------------------------------------------------------------------------- #
# Framing (pure helpers, no codec needed)
# --------------------------------------------------------------------------- #
def test_wrap_parse_opus_chunk_roundtrip():
    packets = [b"\x01\x02\x03", b"", b"x" * 500, b"\xff"]
    frame = audio_utils.wrap_opus_chunk("turn-abc", 7, packets)
    assert frame[:4] == b"VCT3"
    parsed = audio_utils.parse_opus_chunk(frame)
    assert parsed is not None
    turn_id, seq, out = parsed
    assert turn_id == "turn-abc"
    assert seq == 7
    # Empty packets are skipped by the writer.
    assert out == [b"\x01\x02\x03", b"x" * 500, b"\xff"]


def test_parse_opus_chunk_rejects_other_magic():
    vct2 = audio_utils.wrap_pcm_chunk("t1", 1, b"\x00\x00")
    assert audio_utils.parse_opus_chunk(vct2) is None
    assert audio_utils.parse_opus_chunk(b"") is None
    assert audio_utils.parse_opus_chunk(b"VCT3") is None          # truncated header


def test_parse_opus_chunk_drops_truncated_trailing_record():
    frame = audio_utils.wrap_opus_chunk("t", 1, [b"abc", b"defg"])
    _, _, packets = audio_utils.parse_opus_chunk(frame[:-2])      # cut into record 2
    assert packets == [b"abc"]


def test_wrap_opus_chunk_rejects_oversized_packet():
    with pytest.raises(ValueError):
        audio_utils.wrap_opus_chunk("t", 1, [b"x" * 70000])


def test_parse_opus_uplink_packets():
    def rec(p):
        return bytes([audio_utils.OPUS_UPLINK_MARKER,
                      (len(p) >> 8) & 0xFF, len(p) & 0xFF]) + p

    a, b = b"\x11" * 40, b"\x22" * 3
    assert audio_utils.parse_opus_uplink_packets(rec(a) + rec(b)) == [a, b]
    assert audio_utils.parse_opus_uplink_packets(b"") == []
    with pytest.raises(ValueError):
        audio_utils.parse_opus_uplink_packets(b"\x00\x00\x01a")   # wrong marker
    with pytest.raises(ValueError):
        audio_utils.parse_opus_uplink_packets(rec(a)[:-1])        # truncated packet
    with pytest.raises(ValueError):
        audio_utils.parse_opus_uplink_packets(bytes([audio_utils.OPUS_UPLINK_MARKER]))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_audio_opus_config_flag(monkeypatch):
    assert Config.from_env().audio_opus is True                   # default on
    monkeypatch.setenv("AUDIO_OPUS", "0")
    assert Config.from_env().audio_opus is False


# --------------------------------------------------------------------------- #
# Negotiation (availability injected — no codec needed)
# --------------------------------------------------------------------------- #
def _hello_audio(monkeypatch, *, available, audio_opus=True):
    _configure(audio_opus=audio_opus)
    monkeypatch.setattr(opus_codec, "_AVAILABLE", available)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello = await asyncio.wait_for(ws.receive_json(), timeout=3)
            await ws.close()
            return hello["audio"]

    return asyncio.run(run())


def test_hello_advertises_opus_when_available(monkeypatch):
    assert _hello_audio(monkeypatch, available=True) == {"opusIn": True, "opusOut": True}


def test_hello_no_opus_when_lib_missing(monkeypatch):
    assert _hello_audio(monkeypatch, available=False) == {"opusIn": False, "opusOut": False}


def test_hello_no_opus_when_config_off(monkeypatch):
    assert _hello_audio(monkeypatch, available=True, audio_opus=False) \
        == {"opusIn": False, "opusOut": False}


def _settings_codec_ack(monkeypatch, *, available, want="opus"):
    _configure()
    monkeypatch.setattr(opus_codec, "_AVAILABLE", available)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "audioCodec": want})
            ack, _, _ = await _drain_until(ws, "settings.ack")
            await ws.close()
            return ack["audioCodec"]

    return asyncio.run(run())


def test_settings_ack_opus_when_available(monkeypatch):
    assert _settings_codec_ack(monkeypatch, available=True) == "opus"


def test_settings_ack_falls_back_to_pcm_when_lib_missing(monkeypatch):
    assert _settings_codec_ack(monkeypatch, available=False) == "pcm"


def test_settings_ack_pcm_when_client_wants_pcm(monkeypatch):
    assert _settings_codec_ack(monkeypatch, available=True, want="pcm") == "pcm"


def test_uplink_opus_requested_but_unavailable_drops_segment(monkeypatch):
    """Defense in depth: opus stream start without a usable codec → the client
    is told via transcript.error, frames are discarded, commit is a no-op."""
    _configure()
    monkeypatch.setattr(opus_codec, "_AVAILABLE", False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.stream.start",
                                "segmentId": "s1", "codec": "opus"})
            err, _, _ = await _drain_until(ws, "transcript.error")
            assert err is not None and err["segmentId"] == "s1"
            # Frames + commit must not produce a transcript/turn.
            await ws.send_bytes(b"\x4f\x00\x03abc")
            await ws.send_json({"type": "segment.stream.commit", "segmentId": "s1"})
            got, seen, _ = await _drain_until(ws, "transcript", timeout=0.6)
            assert got is None, f"segment was not dropped: {seen}"
            await ws.close()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Real codec: loopback + WS paths (skip when opuslib/libopus is missing)
# --------------------------------------------------------------------------- #
def _sine_pcm16(sample_rate, seconds=1.0, freq=440.0):
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return (0.4 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)


@needs_opus
def test_opus_encode_decode_loopback():
    sr = 16000
    sig = _sine_pcm16(sr)
    enc = opus_codec.OpusEncoder(sr, bitrate=24_000, application="voip")
    packets = enc.encode_pcm16(sig.tobytes())
    packets += enc.flush()
    assert packets                                        # 1 s / 20 ms = 50 packets
    assert len(packets) == 50
    # ~24 kbit/s → ~60 bytes per 20 ms packet; raw would be 640 bytes PCM16.
    assert sum(len(p) for p in packets) < len(sig.tobytes()) / 4

    dec = opus_codec.OpusDecoder(sr)
    pcm = b"".join(dec.decode_packet(p) for p in packets)
    out = np.frombuffer(pcm, dtype=np.float32)
    assert out.shape[0] == sig.shape[0]                   # 16 kHz in = 16 kHz out
    assert 0.2 < np.abs(out).max() < 0.6                  # signal survived


@needs_opus
def test_opus_encoder_flush_pads_partial_frame():
    enc = opus_codec.OpusEncoder(24000)
    assert enc.encode_pcm16(b"\x00\x00" * 100) == []      # < 480 samples buffered
    tail = enc.flush()
    assert len(tail) == 1
    assert enc.flush() == []                              # idempotent

    dec = opus_codec.OpusDecoder(24000)
    out = np.frombuffer(dec.decode_packet(tail[0]), dtype=np.float32)
    assert out.shape[0] == 480                            # one full 20 ms frame


@needs_opus
def test_ws_uplink_opus_decoded_on_arrival():
    """segment.stream.start codec=opus + framed packets + commit → the buffer
    every consumer (incl. commit STT) sees is plain 16 kHz f32 PCM."""
    _configure(reply="Verstanden.")

    class CapturingSTT(FakeSTT):
        def __init__(self):
            self.samples = None

        async def transcribe(self, audio_pcm, sample_rate):
            self.samples = len(audio_pcm) // 4            # f32 bytes → samples
            return "hallo welt"

    stt = CapturingSTT()
    srv.STT = stt

    sig = _sine_pcm16(16000)                              # 1 s = 50 opus packets
    enc = opus_codec.OpusEncoder(16000, bitrate=24_000, application="voip")
    packets = enc.encode_pcm16(sig.tobytes()) + enc.flush()

    def rec(p):
        return bytes([audio_utils.OPUS_UPLINK_MARKER,
                      (len(p) >> 8) & 0xFF, len(p) & 0xFF]) + p

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.stream.start",
                                "segmentId": "s1", "codec": "opus"})
            # Mix single- and multi-packet messages (both are legal).
            await ws.send_bytes(rec(packets[0]))
            await ws.send_bytes(b"".join(rec(p) for p in packets[1:]))
            await ws.send_json({"type": "segment.stream.commit", "segmentId": "s1"})
            tr, seen, _ = await _drain_until(ws, "transcript")
            assert tr is not None, f"kein transcript; gesehen: {seen}"
            assert tr["text"] == "hallo welt"
            await ws.close()

    asyncio.run(run())
    assert stt.samples == 16000                           # 50 × 320 samples @16 kHz


@needs_opus
def test_ws_uplink_corrupt_packet_dropped_not_fatal():
    """A corrupt opus packet is dropped; the stream and connection survive."""
    _configure(reply="Ok.")

    sig = _sine_pcm16(16000, seconds=0.5)
    enc = opus_codec.OpusEncoder(16000, application="voip")
    packets = enc.encode_pcm16(sig.tobytes()) + enc.flush()

    def rec(p):
        return bytes([audio_utils.OPUS_UPLINK_MARKER,
                      (len(p) >> 8) & 0xFF, len(p) & 0xFF]) + p

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "segment.stream.start",
                                "segmentId": "s1", "codec": "opus"})
            await ws.send_bytes(rec(b"\xde\xad\xbe\xef" * 20))    # corrupt packet
            await ws.send_bytes(b"not-a-framed-message")          # malformed framing
            await ws.send_bytes(b"".join(rec(p) for p in packets))
            await ws.send_json({"type": "segment.stream.commit", "segmentId": "s1"})
            tr, seen, _ = await _drain_until(ws, "transcript")
            assert tr is not None, f"kein transcript; gesehen: {seen}"
            await ws.close()

    asyncio.run(run())


@needs_opus
def test_ws_downlink_opus_vct3_end_to_end():
    """settings audioCodec=opus → text.message → audio.start carries
    codec=opus and the audio arrives as decodable VCT3 opus frames."""
    _configure(reply="Hallo zusammen.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "audioCodec": "opus"})
            ack, _, _ = await _drain_until(ws, "settings.ack")
            assert ack["audioCodec"] == "opus"
            await ws.send_json({"type": "text.message", "text": "Hi"})
            end, seen, binaries = await _drain_collect(ws, "audio.end")
            assert end is not None, f"kein audio.end; gesehen: {seen}"
            start = next(m for m in seen if isinstance(m, dict)
                         and m.get("type") == "audio.start")
            assert start["codec"] == "opus"
            assert start["sampleRate"] == 24000
            assert binaries, "keine Binary-Frames empfangen"
            dec = opus_codec.OpusDecoder(24000)
            total = 0
            for frame in binaries:
                assert frame[:4] == b"VCT3", f"unexpected framing: {frame[:4]!r}"
                turn_id, seq, packets = audio_utils.parse_opus_chunk(frame)
                assert turn_id == start["turnId"]
                assert seq >= 1 and packets
                for p in packets:
                    total += len(dec.decode_packet(p)) // 4
            # FakeTTS ships 4 samples → flushed into one padded 20 ms frame.
            assert total >= 480 and total % 480 == 0
            assert end["chunks"] == len(binaries)
            await ws.close()

    asyncio.run(run())


@needs_opus
def test_ws_downlink_stays_pcm_without_opt_in():
    """No audioCodec setting → the downlink stays raw VCT2 even though the
    server could do opus (the client decides)."""
    _configure(reply="Hallo welt.")

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "text.message", "text": "Hi"})
            end, seen, binaries = await _drain_collect(ws, "audio.end")
            assert end is not None
            start = next(m for m in seen if isinstance(m, dict)
                         and m.get("type") == "audio.start")
            assert start["codec"] == "pcm"
            assert binaries and all(b[:4] == b"VCT2" for b in binaries)
            await ws.close()

    asyncio.run(run())


def test_ws_downlink_falls_back_to_vct2_when_lib_missing(monkeypatch):
    """Client asks for opus but the codec is unusable → ack says pcm and the
    audio still arrives as raw VCT2 (nothing breaks)."""
    _configure(reply="Hallo welt.")
    monkeypatch.setattr(opus_codec, "_AVAILABLE", False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "audioCodec": "opus"})
            ack, _, _ = await _drain_until(ws, "settings.ack")
            assert ack["audioCodec"] == "pcm"
            await ws.send_json({"type": "text.message", "text": "Hi"})
            end, seen, binaries = await _drain_collect(ws, "audio.end")
            assert end is not None
            assert binaries and all(b[:4] == b"VCT2" for b in binaries)
            await ws.close()

    asyncio.run(run())
