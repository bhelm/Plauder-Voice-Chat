"""House-Mode speaker-ID surface: hello.houseSpeaker capability block, live
settings (thresholds + on/off) and the speaker info on transcript events.

The identifier is duck-typed and injected via ``srv._SPEAKER_IDENTIFIER`` (the
external ``speaker_id`` module is not part of the repo); conftest restores the
module globals after every test.
"""
import asyncio
import dataclasses
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer

from plauder import server as srv
from plauder.config import Config
from plauder.sanitizer import HallucinationFilter
from plauder.session import ConversationManager

from test_wake_pipeline import (GatedStreamingLLM, ScriptedSTT, FakeTTS,
                                _drain_until, _send_voice)


class FakeStore:
    """Duck-typed speaker_id.SpeakerStore incl. the management API."""

    def __init__(self, names):
        self._speakers = {}
        for n in names:
            self._speakers[n] = SimpleNamespace(
                name=n, role="guest", relation="",
                embeddings=[[0.0]], registers=["take-1"], n_samples=1)

    def all(self):
        return list(self._speakers.values())

    def get(self, name):
        return self._speakers.get(name)

    def add_register(self, name, emb, role=None, label=None, relation=None):
        sp = self._speakers.get(name)
        if sp is None:
            sp = SimpleNamespace(name=name, role=role or "guest",
                                 relation=relation or "",
                                 embeddings=[], registers=[], n_samples=0)
            self._speakers[name] = sp
        sp.embeddings.append(emb)
        sp.registers.append(label or "")
        sp.n_samples = len(sp.embeddings)

    def remove(self, name):
        return self._speakers.pop(name, None) is not None

    def rename(self, old, new):
        sp = self._speakers.get(old)
        if sp is None or new in self._speakers:
            return False
        del self._speakers[old]
        sp.name = new
        self._speakers[new] = sp
        return True

    def remove_register(self, name, index):
        sp = self._speakers.get(name)
        if sp is None or not (0 <= index < len(sp.embeddings)):
            return False
        del sp.embeddings[index]
        if index < len(sp.registers):
            del sp.registers[index]
        sp.n_samples = len(sp.embeddings)
        return True

    def rename_register(self, name, index, label):
        sp = self._speakers.get(name)
        if sp is None or not (0 <= index < len(sp.embeddings)):
            return False
        while len(sp.registers) <= index:
            sp.registers.append("")
        sp.registers[index] = label
        return True


class FakeIdentifier:
    """Duck-typed speaker_id.SpeakerIdentifier (do NOT import the real one)."""

    def __init__(self, names=("robert", "xena")):
        self.store = FakeStore(names)
        self.embedder = SimpleNamespace(embed=lambda audio, sr=16000: [0.5, 0.5])
        self.sim_threshold = 0.5
        self.switch_threshold = 0.6
        self.min_dur_s = 5.0
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def identify(self, samples, sample_rate=16000, now=None,
                 speech_start_ts=None, force_hold=False):
        self.calls += 1
        return SimpleNamespace(name="robert", role="admin", relation="Vater",
                               score=0.71, known=True, held=False)


def _configure_house(stt_texts, deltas, *, house=True):
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=20, streaming=True,
                              house_speaker_id=house)
    llm = GatedStreamingLLM(deltas)
    llm.gate.set()
    conv = ConversationManager(llm, system_prompt="sys")
    srv.configure(cfg, stt=ScriptedSTT(stt_texts), tts=FakeTTS(), conv=conv,
                  bridge=None, ghost=HallucinationFilter(enabled=False))
    ident = FakeIdentifier() if house else None
    srv._SPEAKER_IDENTIFIER = ident
    srv._SPEAKER_INIT_FAILED = False
    srv.HOUSE_SPEAKER_ENABLED = True
    return cfg, ident


def test_hello_advertises_house_speaker():
    _configure_house(stt_texts=[], deltas=["x"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello, _ = await _drain_until(ws, "hello")
            hs = hello.get("houseSpeaker")
            assert hs and hs["available"] is True
            assert hs["enabled"] is True
            assert hs["count"] == 2 and hs["names"] == ["robert", "xena"]
            assert hs["simThreshold"] == 0.5
            assert hs["switchThreshold"] == 0.6
            assert hs["minDurS"] == 5.0
            await ws.close()

    asyncio.run(run())


def test_hello_house_speaker_unavailable_without_feature():
    _configure_house(stt_texts=[], deltas=["x"], house=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello, _ = await _drain_until(ws, "hello")
            assert hello.get("houseSpeaker") == {"available": False}
            await ws.close()

    asyncio.run(run())


def test_house_settings_update_thresholds_and_toggle():
    _cfg, ident = _configure_house(stt_texts=[], deltas=["x"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "settings", "houseSimThreshold": 0.42,
                                "houseSwitchThreshold": 0.8,
                                "houseSpeakerEnabled": False})
            ack, _ = await _drain_until(ws, "settings.ack")
            assert ack["houseSimThreshold"] == 0.42
            assert ack["houseSwitchThreshold"] == 0.8
            assert ack["houseSpeakerEnabled"] is False
            assert ident.sim_threshold == 0.42
            assert ident.switch_threshold == 0.8
            assert srv.HOUSE_SPEAKER_ENABLED is False

            # Out-of-range values are clamped, junk is ignored.
            await ws.send_json({"type": "settings", "houseSimThreshold": 5.0,
                                "houseSwitchThreshold": "junk"})
            ack, _ = await _drain_until(ws, "settings.ack")
            assert ack["houseSimThreshold"] == 0.95
            assert ident.switch_threshold == 0.8   # unchanged
            await ws.close()

    asyncio.run(run())


def test_settings_ack_house_none_when_unavailable():
    _configure_house(stt_texts=[], deltas=["x"], house=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "settings", "houseSpeakerEnabled": False})
            ack, _ = await _drain_until(ws, "settings.ack")
            assert ack["houseSpeakerEnabled"] is None
            assert ack["houseSimThreshold"] is None
            # Without the identifier the toggle must not flip the global.
            assert srv.HOUSE_SPEAKER_ENABLED is True
            await ws.close()

    asyncio.run(run())


def test_house_enroll_creates_speaker_and_adds_samples():
    _cfg, ident = _configure_house(stt_texts=[], deltas=["x"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")

            # New speaker: first committed take creates the store entry.
            await ws.send_json({"type": "house.enroll.start", "name": "kind"})
            await ws.send_bytes(b"\x00" * (16000 * 4 * 3))   # 3 s take
            await ws.send_json({"type": "house.enroll.commit"})
            ack, seen = await _drain_until(ws, "house.enroll.ack")
            assert ack is not None and ack["ok"] is True, seen
            assert ack["name"] == "kind" and ack["count"] == 1
            spk, _ = await _drain_until(ws, "house.speakers")
            assert [s["name"] for s in spk["speakers"]] == ["robert", "xena", "kind"]

            # Existing speaker: another take becomes a second sample.
            await ws.send_json({"type": "house.enroll.start", "name": "kind"})
            await ws.send_bytes(b"\x00" * (16000 * 4 * 3))
            await ws.send_json({"type": "house.enroll.commit"})
            ack, _ = await _drain_until(ws, "house.enroll.ack")
            assert ack["ok"] is True and ack["count"] == 2
            spk, _ = await _drain_until(ws, "house.speakers")
            assert len(spk["speakers"][2]["samples"]) == 2
            assert ident.resets >= 2
            await ws.close()

    asyncio.run(run())


def test_house_enroll_too_short_is_rejected():
    _configure_house(stt_texts=[], deltas=["x"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "house.enroll.start", "name": "kind"})
            await ws.send_bytes(b"\x00" * 16000)   # 0.25 s — far below 2 s
            await ws.send_json({"type": "house.enroll.commit"})
            ack, _ = await _drain_until(ws, "house.enroll.ack")
            assert ack["ok"] is False and ack["error"] == "too_short"
            await ws.close()

    asyncio.run(run())


def test_house_rename_delete_and_sample_delete():
    _cfg, ident = _configure_house(stt_texts=[], deltas=["x"])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")

            await ws.send_json({"type": "house.rename",
                                "name": "robert", "newName": "bob"})
            spk, _ = await _drain_until(ws, "house.speakers")
            names = [s["name"] for s in spk["speakers"]]
            assert "bob" in names and "robert" not in names

            await ws.send_json({"type": "house.delete", "name": "xena"})
            spk, _ = await _drain_until(ws, "house.speakers")
            assert [s["name"] for s in spk["speakers"]] == ["bob"]

            await ws.send_json({"type": "house.sample.rename",
                                "name": "bob", "index": 0, "label": "leise"})
            spk, _ = await _drain_until(ws, "house.speakers")
            assert spk["speakers"][0]["samples"] == ["leise"]

            # Rename with a bad index / empty label → error, store untouched.
            await ws.send_json({"type": "house.sample.rename",
                                "name": "bob", "index": 5, "label": "x"})
            err, _ = await _drain_until(ws, "house.error")
            assert err["op"] == "sample.rename" and err["error"] == "not_found"
            await ws.send_json({"type": "house.sample.rename",
                                "name": "bob", "index": 0, "label": "  "})
            err, _ = await _drain_until(ws, "house.error")
            assert err["op"] == "sample.rename" and err["error"] == "bad_request"

            await ws.send_json({"type": "house.sample.delete",
                                "name": "bob", "index": 0})
            spk, _ = await _drain_until(ws, "house.speakers")
            assert spk["speakers"][0]["samples"] == []

            # Unknown speaker → error to the requester only.
            await ws.send_json({"type": "house.delete", "name": "nope"})
            err, _ = await _drain_until(ws, "house.error")
            assert err["op"] == "delete" and err["error"] == "not_found"
            assert ident.resets >= 3
            await ws.close()

    asyncio.run(run())


def test_house_mgmt_unavailable_without_feature():
    _configure_house(stt_texts=[], deltas=["x"], house=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "house.list"})
            err, _ = await _drain_until(ws, "house.error")
            assert err["error"] == "unavailable"
            await ws.close()

    asyncio.run(run())


def test_transcript_carries_speaker_until_toggled_off():
    _cfg, ident = _configure_house(
        stt_texts=["hallo haus", "noch ein satz"], deltas=["Ok. ", "Gerne."])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")

            await _send_voice(ws, "s1")
            tr, seen = await _drain_until(ws, "transcript")
            assert tr is not None, seen
            assert tr["speaker"]["name"] == "robert"
            assert tr["speaker"]["role"] == "admin"
            assert tr["speaker"]["score"] == 0.71
            await _drain_until(ws, "reply")

            # Toggle recognition off → the next transcript is untagged.
            await ws.send_json({"type": "settings", "houseSpeakerEnabled": False})
            await _drain_until(ws, "settings.ack")
            await _send_voice(ws, "s2")
            tr2, seen2 = await _drain_until(ws, "transcript")
            assert tr2 is not None, seen2
            assert "speaker" not in tr2
            assert ident.calls == 1   # identify() not called while off
            await ws.close()

    asyncio.run(run())
