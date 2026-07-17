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


# --------------------------------------------------------------------------- #
# A deliberately SLOW streaming TTS so a reply/push is still in flight (its
# TTS worker draining) when a barge-in arrives. Duck-typed like FakeTTS; it
# exposes ``synth_stream`` (so _tts_synth_stream takes the streaming path and
# emits several VCT2 chunks) plus a ``synth`` fallback. Each chunk is gated by
# a tiny sleep — that is the whole point of this fake, hence the sleep.
# --------------------------------------------------------------------------- #
class SlowTTS:
    sample_rate = 24000
    # Each chunk = 480 int16 samples (20 ms @ 24 kHz) at amplitude 3000. That
    # is loud enough (RMS ≫ the StreamLeadGate's −45 dBFS onset threshold) and
    # long enough (> the gate's 10 ms window) that the gate opens on the FIRST
    # chunk — so audio.start fires early and there is a real streaming window
    # to barge into, instead of the gate holding tiny silent chunks to the end.
    _CHUNK = b"\xb8\x0b" * 480   # 0x0bb8 = 3000, little-endian

    def __init__(self, chunk_delay=0.08, chunks=6):
        self.chunk_delay = chunk_delay
        self.chunks = chunks

    async def synth_stream(self, text, *, speed=1.0, voice=None):
        for _ in range(self.chunks):
            await asyncio.sleep(self.chunk_delay)
            yield self._CHUNK, self.sample_rate

    async def synth(self, text, *, speed=1.0):
        return self._CHUNK, self.sample_rate

    def describe(self):
        return {"engine": "slow-tts", "sample_rate": self.sample_rate,
                "loaded": True}


async def _collect(ws, *, until=None, timeout=3.0):
    """Read WS frames until ``until(frame)`` returns True or the timeout
    elapses. Returns (frames, binaries): frames is the list of JSON dicts seen
    (including the matching one), binaries the raw binary payloads seen. Used
    where a test must reason about the ORDER of several frames, not just the
    first occurrence of one type (which is all `_drain_until` gives)."""
    frames = []
    binaries = []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            msg = await asyncio.wait_for(
                ws.receive(), timeout=max(0.02, deadline - loop.time()))
        except asyncio.TimeoutError:
            break
        if msg.type.name == "BINARY":
            binaries.append(msg.data)
            continue
        if msg.type.name in ("CLOSE", "CLOSING", "CLOSED", "ERROR"):
            break
        data = msg.json()
        frames.append(data)
        if until is not None and until(data):
            break
    return frames, binaries


def _types(frames):
    return [f.get("type") for f in frames]


class RecordingLLM(FakeLLM):
    """FakeLLM that also exposes the duck-typed notify_push_undelivered hook
    (the gateway backend's real method) so tests can assert an unheard,
    barged-into push notified the gateway."""

    def __init__(self, reply="Antwort auf die Unterbrechung."):
        super().__init__(reply=reply)
        self.undelivered = []   # list of (text, played_s)

    async def notify_push_undelivered(self, text, played_s):
        self.undelivered.append((text, played_s))


def _configure(streaming=True, tts=None, llm=None, push_heard_threshold_s=3.0):
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=30,
                              streaming=streaming,
                              push_heard_threshold_s=push_heard_threshold_s)
    conv = ConversationManager(llm or FakeLLM(), system_prompt="sys")
    srv.configure(cfg, stt=FakeSTT(), tts=tts or FakeTTS(), conv=conv,
                  bridge=None, ghost=HallucinationFilter(enabled=False))
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


def test_session_reset_calls_gateway_reset_hook():
    cfg = _configure()

    class ResettableFakeLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.resets = 0

        async def reset_session(self):
            self.resets += 1

    llm = ResettableFakeLLM()
    srv.configure(cfg, stt=FakeSTT(), tts=FakeTTS(),
                  conv=ConversationManager(llm, system_prompt="sys"),
                  bridge=None,
                  ghost=srv.sanitizer.HallucinationFilter(enabled=False))

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "session.reset"})
            ack, seen, _ = await _drain_until(ws, "session.reset.ack")
            assert ack is not None, f"kein ack; gesehen: {seen}"
            assert llm.resets == 1
            await ws.close()

    asyncio.run(run())


def test_system_push_is_shown_but_not_spoken():
    _configure()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await srv.handle_gateway_push("♻️ Gateway online", speak=False)
            evt, seen, binary = await _drain_until(ws, "external.message")
            assert evt is not None, f"kein external.message; gesehen: {seen}"
            assert evt["text"] == "♻️ Gateway online"
            assert evt["source"] == "system"
            assert "reply.start" not in seen
            assert "audio.start" not in seen
            assert binary is None
            await ws.close()

    asyncio.run(run())


def test_empty_push_is_ignored():
    _configure()

    async def run():
        await srv.handle_gateway_push("   ")
        assert len(srv._PENDING_PUSHES) == 0

    asyncio.run(run())


# ============================================================================ #
# Push barge-in contract (refined design).
#
# A push claims state.agent_task with its OWN turn id "push-<id>" while
# state.turn_id keeps belonging to the NEXT user turn. When a user barge-in
# cancels a still-speaking push, turn.discarded names the PUSH (agent_turn_id),
# and the cancelled content is never lost — but it is NOT re-spoken verbatim.
# What happens depends on how much the browser had actually played (reported via
# `playback.stopped`, threshold CFG.push_heard_threshold_s):
#   heard   (played ≥ threshold): the user knowingly stopped THIS message →
#           surface the text as a chat bubble, nothing else.
#   unheard (< threshold, or no report): text bubble AND notify the gateway
#           (notify_push_undelivered) so the agent weaves it into its answer.
# Other cancel causes: stop-word → text bubble; reset → drop; close → re-queue.
# ============================================================================ #
def _push_texts_of(frames):
    """external.message bubbles carrying a cancelled push's surfaced text."""
    return [f for f in frames
            if f.get("type") == "external.message" and f.get("source") == "push"]


async def _push_until_audio(ws, push_text):
    """Fire a push and drain until its audio is actually streaming (audio.start
    seen). Returns (push_turn_id, frames_seen). At this point the push's TTS
    worker is mid-stream (SlowTTS) so the push task is still in flight and a
    barge-in will cancel it via _cancel_in_flight."""
    await srv.handle_gateway_push(push_text)
    start, _ = await _collect(ws, until=lambda f: f.get("type") == "reply.start")
    rs = start[-1]
    assert rs.get("type") == "reply.start" and rs.get("push") is True, \
        f"expected push reply.start, saw: {_types(start)}"
    push_turn_id = rs["turnId"]
    aud, _ = await _collect(ws, until=lambda f: f.get("type") == "audio.start")
    assert aud and aud[-1].get("type") == "audio.start", \
        f"push audio never started; saw: {_types(aud)}"
    return push_turn_id, start + aud


def test_push_bargein_turn_discarded_carries_push_id():
    """turn.discarded during a push barge-in must name the PUSH's turn, not the
    incoming user turn (whose reply then gets suppressed client-side while its
    audio still plays)."""
    _configure(tts=SlowTTS())

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            push_turn_id, _ = await _push_until_audio(ws, "Und jetzt die zwei "
                                                      "Haken der Nacht.")
            # Barge in: a typed message cancels the in-flight push exactly like
            # a committed voice segment (both go through _coalesce_cancel ->
            # _cancel_in_flight); text is the simplest reliable trigger here.
            await ws.send_json({"type": "text.message", "text": "warte kurz"})
            disc, _ = await _collect(
                ws, until=lambda f: f.get("type") == "turn.discarded", timeout=4.0)
            assert disc and disc[-1].get("type") == "turn.discarded", \
                f"no turn.discarded; saw: {_types(disc)}"
            discarded_id = disc[-1]["turnId"]
            # The interrupting user turn's id shows up on its turn.commit.
            comm, _ = await _collect(
                ws, until=lambda f: f.get("type") == "turn.commit", timeout=4.0)
            assert comm and comm[-1].get("type") == "turn.commit", \
                f"no turn.commit for the user turn; saw: {_types(comm)}"
            new_turn_id = comm[-1]["turnId"]

            # Contract: the discarded id is the push's, and is NOT the id the
            # following user turn commits under. (The old bug had discarded_id
            # == new_turn_id, both == state.turn_id, differing from the push id.)
            assert discarded_id == push_turn_id, (
                f"turn.discarded turnId {discarded_id!r} should be the push's "
                f"{push_turn_id!r}")
            assert discarded_id != new_turn_id, (
                f"turn.discarded turnId {discarded_id!r} must not be the new "
                f"user turn's id {new_turn_id!r}")
            await ws.close()

    asyncio.run(run())


def test_push_bargein_unheard_notifies_gateway_and_persists_text():
    """No playback.stopped report (or a sub-threshold one) → the push counts as
    UNHEARD: the gateway is told via notify_push_undelivered (so the agent can
    weave the content into its next answer) AND the text is surfaced as a
    persisted chat bubble. It is NOT re-spoken as a fresh push reply."""
    llm = RecordingLLM()
    _configure(tts=SlowTTS(), llm=llm)
    push_text = "Und jetzt die zwei Haken der Nacht."

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            push_turn_id, _ = await _push_until_audio(ws, push_text)
            # No playback.stopped sent → server treats played as 0.0 (unheard).
            await ws.send_json({"type": "text.message", "text": "warte kurz"})

            # Drain past the interrupting turn's own (non-push) reply, then a bit.
            frames, _ = await _collect(
                ws,
                until=lambda f: (f.get("type") == "reply" and not f.get("push")),
                timeout=5.0)
            more, _ = await _collect(ws, timeout=1.5)
            all_frames = frames + more

            bubbles = _push_texts_of(all_frames)
            assert any(b.get("text") == push_text and b.get("persist")
                       for b in bubbles), (
                "unheard push must be surfaced as a persisted text bubble; "
                f"saw: {_types(all_frames)}")
            # NOT re-spoken: no second push reply.start after the original.
            push_starts = [f for f in all_frames
                           if f.get("type") == "reply.start" and f.get("push")]
            assert not push_starts, (
                f"unheard push must NOT be re-spoken; got {len(push_starts)} "
                "new push reply.start(s)")
            # Gateway was notified with the full text and played≈0.
            assert llm.undelivered, "gateway hook was not called"
            text, played = llm.undelivered[-1]
            assert text == push_text
            assert played == 0.0
            await ws.close()

    asyncio.run(run())


def test_push_bargein_heard_persists_text_without_gateway_notify():
    """A playback.stopped report ≥ threshold before the barge-in → the push
    counts as HEARD: the text is surfaced as a bubble but the gateway is NOT
    notified (the user knowingly stopped that message)."""
    llm = RecordingLLM()
    _configure(tts=SlowTTS(), llm=llm)
    push_text = "Und jetzt die zwei Haken der Nacht."

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            push_turn_id, _ = await _push_until_audio(ws, push_text)
            # Report 5 s played (≥ the 3 s threshold) for THIS push, THEN
            # barge in — same WS, so the report is recorded before the cancel.
            await ws.send_json({"type": "playback.stopped",
                                "turnId": push_turn_id, "playedS": 5.0})
            await ws.send_json({"type": "text.message", "text": "warte kurz"})

            frames, _ = await _collect(
                ws,
                until=lambda f: (f.get("type") == "reply" and not f.get("push")),
                timeout=5.0)
            more, _ = await _collect(ws, timeout=1.5)
            all_frames = frames + more

            bubbles = _push_texts_of(all_frames)
            assert any(b.get("text") == push_text and b.get("persist")
                       for b in bubbles), (
                "heard push must still be surfaced as a persisted text bubble; "
                f"saw: {_types(all_frames)}")
            assert not llm.undelivered, (
                "heard push must NOT notify the gateway; got "
                f"{llm.undelivered}")
            await ws.close()

    asyncio.run(run())


def test_queued_push_and_bargein_lose_no_content():
    """Two pushes: first speaking, second waiting for the slot; a barge-in
    interrupts the speaking one. Governing rule: nothing unspoken is lost. The
    first (unheard, cancelled) push surfaces as a text bubble + gateway notify;
    the still-waiting second push is spoken normally after the user turn; and
    the interrupting user turn gets its own (non-push) reply."""
    llm = RecordingLLM()
    _configure(tts=SlowTTS(chunk_delay=0.05, chunks=4), llm=llm)
    t1, t2 = "Erste wichtige Nachricht.", "Zweite wichtige Nachricht."

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await srv.handle_gateway_push(t1)
            await _collect(ws, until=lambda f: f.get("type") == "reply.start",
                           timeout=4.0)
            await srv.handle_gateway_push(t2)  # queued, waiting for the slot
            await _collect(ws, until=lambda f: f.get("type") == "audio.start",
                           timeout=4.0)
            await ws.send_json({"type": "text.message", "text": "warte"})

            frames, _ = await _collect(ws, timeout=8.0)
            spoken_pushes = {f.get("text") for f in frames
                             if f.get("type") == "reply" and f.get("push")}
            bubble_texts = {b.get("text") for b in _push_texts_of(frames)}
            user_replies = [f for f in frames
                            if f.get("type") == "reply" and not f.get("push")]
            # The still-waiting second push is spoken; the cancelled first push
            # is surfaced as a text bubble (not re-spoken).
            assert t2 in spoken_pushes, (
                f"the waiting second push must still be spoken; got {spoken_pushes}")
            assert t1 in bubble_texts, (
                f"the cancelled first push must surface as a bubble; got {bubble_texts}")
            assert t1 not in spoken_pushes, (
                "the cancelled first push must NOT be re-spoken verbatim")
            assert user_replies, "the interrupting user turn must still reply"
            assert (t1, 0.0) in llm.undelivered, (
                f"unheard first push must notify the gateway; got {llm.undelivered}")
            await ws.close()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Cancel matrix — the non-barge-in causes, exercised directly on
# _handle_cancelled_push (deterministic, no timing): the mode set by the
# cancelling caller decides, independent of any playback report.
# --------------------------------------------------------------------------- #
class _FakeWs:
    closed = False

    def __init__(self):
        self.sent = []

    async def send_json(self, obj):
        self.sent.append(obj)


def test_cancelled_push_stop_command_text_bubble_no_notify():
    """push_cancel_mode='text' (wake stop-command): persisted text bubble, no
    gateway notify — regardless of how much played."""
    llm = RecordingLLM()
    _configure(llm=llm)

    async def run():
        state = srv.TurnState()
        state.push_cancel_mode = "text"
        ws = _FakeWs()
        srv._handle_cancelled_push(ws, state, "Wichtige Info.", True, "push-x")
        await asyncio.gather(*list(state.inflight_tasks))
        bubbles = [f for f in ws.sent if f.get("type") == "external.message"]
        assert any(b["text"] == "Wichtige Info." and b.get("source") == "push"
                   and b.get("persist") for b in bubbles), ws.sent
        assert not llm.undelivered

    asyncio.run(run())


def test_cancelled_push_session_reset_drops():
    """push_cancel_mode='drop' (session reset): nothing sent, nothing queued."""
    _configure()

    async def run():
        state = srv.TurnState()
        state.push_cancel_mode = "drop"
        ws = _FakeWs()
        srv._handle_cancelled_push(ws, state, "Alt.", True, "push-x")
        await asyncio.gather(*list(state.inflight_tasks))
        assert ws.sent == []
        assert len(srv._PENDING_PUSHES) == 0

    asyncio.run(run())


def test_cancelled_push_closing_requeues():
    """Connection closing: the push goes back onto _PENDING_PUSHES (spoken on
    the next connect), never a text bubble."""
    _configure()

    async def run():
        state = srv.TurnState()
        state.closing = True
        ws = _FakeWs()
        srv._handle_cancelled_push(ws, state, "Später.", True, "push-x")
        await asyncio.gather(*list(state.inflight_tasks))
        assert ("Später.", True) in srv._PENDING_PUSHES
        assert ws.sent == []

    asyncio.run(run())


def test_disconnect_requeues_all_unspoken_pushes():
    """On disconnect a push still speaking AND a push still waiting in its
    slot-wait loop both go back onto _PENDING_PUSHES (spoken on next connect) —
    a waiting push must not vanish just because it never claimed the slot."""
    _configure(tts=SlowTTS(chunk_delay=0.05, chunks=6))
    t1, t2 = "Erste.", "Zweite unausgesprochene."

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await srv.handle_gateway_push(t1)
            await _collect(ws, until=lambda f: f.get("type") == "reply.start",
                           timeout=4.0)
            await srv.handle_gateway_push(t2)  # waiting for the slot
            await _collect(ws, until=lambda f: f.get("type") == "audio.start",
                           timeout=4.0)
            await ws.close()
            await asyncio.sleep(0.3)  # let cancellation handlers run

        pending = {t for (t, _s) in srv._PENDING_PUSHES}
        assert {t1, t2} <= pending, (
            f"both unspoken pushes must be re-queued on disconnect; got {pending}")

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Regression guards — these describe behaviour that is ALREADY correct and must
# stay green.
# --------------------------------------------------------------------------- #
def test_normal_bargein_turn_discarded_carries_running_turn_id():
    """Barge-in during a REGULAR (non-push) streaming reply: turn.discarded
    names the turn whose reply was in flight, and the interrupting turn then
    runs under a fresh id. (This is the correct counterpart to Bug 1 — the
    normal turn rotates state.turn_id on cancel, the push path does not.)"""
    _configure(tts=SlowTTS())

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "text.message", "text": "erste frage"})
            rs, _ = await _collect(
                ws, until=lambda f: f.get("type") == "reply.start", timeout=4.0)
            first_turn_id = rs[-1]["turnId"]
            # Wait until this reply is audibly streaming, so the barge-in is a
            # genuine interruption (no coalescing) — then interrupt.
            await _collect(
                ws, until=lambda f: f.get("type") == "audio.start", timeout=4.0)
            await ws.send_json({"type": "text.message", "text": "zweite frage"})

            disc, _ = await _collect(
                ws, until=lambda f: f.get("type") == "turn.discarded", timeout=4.0)
            assert disc and disc[-1].get("type") == "turn.discarded", \
                f"no turn.discarded; saw: {_types(disc)}"
            assert disc[-1]["turnId"] == first_turn_id, (
                f"turn.discarded turnId {disc[-1]['turnId']!r} should name the "
                f"running turn {first_turn_id!r}")

            comm, _ = await _collect(
                ws, until=lambda f: f.get("type") == "turn.commit", timeout=4.0)
            assert comm and comm[-1]["turnId"] != first_turn_id, (
                "the interrupting turn must commit under a fresh id, got "
                f"{comm[-1].get('turnId')!r}")
            await ws.close()

    asyncio.run(run())


def test_two_pushes_are_serialized_not_interleaved():
    """A second push arriving while the first is still speaking waits for the
    agent_task slot: both are spoken, and their audio streams never interleave
    (each push's audio.start..audio.end nests cleanly)."""
    _configure(tts=SlowTTS(chunk_delay=0.05, chunks=4))
    t1, t2 = "Erste Nachricht.", "Zweite Nachricht."

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await srv.handle_gateway_push(t1)
            # First push is now in flight; queue the second while it speaks.
            await _collect(ws, until=lambda f: f.get("type") == "reply.start",
                           timeout=4.0)
            await srv.handle_gateway_push(t2)

            # Drain until we have seen two audio.end frames (both pushes done).
            ends = 0

            def _two_ends(f):
                nonlocal ends
                if f.get("type") == "audio.end":
                    ends += 1
                return ends >= 2

            frames, _ = await _collect(ws, until=_two_ends, timeout=8.0)

            push_replies = [f for f in frames
                            if f.get("type") == "reply" and f.get("push")]
            texts = {f.get("text") for f in push_replies}
            assert texts == {t1, t2}, \
                f"both pushes must be spoken; got replies {texts}"

            # audio.start/audio.end must nest — no second start before the
            # first end (no interleaving of the two audio streams).
            open_audio = None
            for f in frames:
                if f.get("type") == "audio.start":
                    assert open_audio is None, (
                        f"audio {f.get('audioId')!r} started while "
                        f"{open_audio!r} was still open — interleaved")
                    open_audio = f.get("audioId")
                elif f.get("type") == "audio.end":
                    assert f.get("audioId") == open_audio, (
                        f"audio.end {f.get('audioId')!r} does not match the "
                        f"open stream {open_audio!r}")
                    open_audio = None
            await ws.close()

    asyncio.run(run())


def test_normal_turn_after_push_is_clean():
    """After a push finishes normally, a following user turn commits/replies
    under its own id with no stray turn.discarded."""
    _configure()  # fast FakeTTS: the push completes quickly

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await srv.handle_gateway_push("Kurze Info.")
            push_end, _ = await _collect(
                ws, until=lambda f: f.get("type") == "audio.end", timeout=4.0)
            push_turn_id = push_end[-1]["turnId"]
            # Quiescence drain: also lets the push's finally clear agent_task
            # (no frame signals that), so the next text turn is not mistaken
            # for a barge-in on the just-finished push.
            await _collect(ws, timeout=0.2)

            await ws.send_json({"type": "text.message", "text": "hallo"})
            frames, _ = await _collect(
                ws, until=lambda f: f.get("type") == "audio.end", timeout=4.0)
            assert "turn.discarded" not in _types(frames), (
                "a clean user turn after a push must not emit turn.discarded; "
                f"saw: {_types(frames)}")
            commit = next(f for f in frames if f.get("type") == "turn.commit")
            reply = next(f for f in frames if f.get("type") == "reply")
            end = frames[-1]
            assert commit["turnId"] == reply["turnId"] == end["turnId"], (
                "commit/reply/audio.end must share the user turn id, got "
                f"{commit['turnId']!r}/{reply['turnId']!r}/{end['turnId']!r}")
            assert reply["turnId"] != push_turn_id, \
                "user turn must not reuse the push's turn id"
            assert not reply.get("push")
            await ws.close()

    asyncio.run(run())


# ============================================================================ #
# A. Confirmed bugs — the waiting-push slot-wait loop.
# ============================================================================ #
def test_session_reset_drops_waiting_push():
    """A push still in its slot-wait loop (only in inflight_tasks, has NOT
    claimed agent_task) must be DROPPED by a session reset — reset is the
    sanctioned drop (the content belongs to the old session). It must not
    speak into the fresh session, and (unlike a close) must not re-queue."""
    _configure(tts=SlowTTS(chunk_delay=0.05, chunks=8))
    t1, t2 = "Erste spricht schon.", "Zweite wartet noch."

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await srv.handle_gateway_push(t1)   # speaks, claims the slot
            await _collect(ws, until=lambda f: f.get("type") == "audio.start",
                           timeout=4.0)
            await srv.handle_gateway_push(t2)   # waits for the slot
            # Session reset: t1 (speaking) is cancelled via agent_task → dropped;
            # t2 (still waiting) must ALSO drop.
            await ws.send_json({"type": "session.reset"})
            ack, seen, _ = await _drain_until(ws, "session.reset.ack", timeout=4.0)
            assert ack is not None, f"no reset ack; saw {seen}"
            # Let the (now-unblocked) waiting push task run if it wrongly would.
            frames, _ = await _collect(ws, timeout=1.5)
            spoken = {f.get("text") for f in frames
                      if f.get("type") == "reply" and f.get("push")}
            assert t2 not in spoken, (
                f"a waiting push must NOT speak into the fresh session; got {spoken}")
            assert not srv._PENDING_PUSHES, (
                "a reset DROPS the waiting push (not re-queue); got "
                f"{list(srv._PENDING_PUSHES)}")
            await ws.close()

    asyncio.run(run())


def test_peer_session_reset_drops_waiting_push_on_other_client():
    """A reset from ANOTHER connected browser is global (one shared session) →
    it must drop the peer's still-waiting pushes too, not only the resetter's."""
    _configure(tts=SlowTTS(chunk_delay=0.05, chunks=6))
    push_text = "Hintergrund-Info."

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            wsA = await client.ws_connect("/ws")
            await wsA.receive_json()  # hello
            wsB = await client.ws_connect("/ws")
            await wsB.receive_json()  # hello
            # Occupy A's reply slot with a user turn so a push to A must WAIT.
            await wsA.send_json({"type": "text.message", "text": "lange frage"})
            await _collect(wsA, until=lambda f: f.get("type") == "audio.start",
                           timeout=4.0)
            # Push reaches BOTH: A queues it (A busy), B speaks it.
            await srv.handle_gateway_push(push_text)
            await _collect(wsB, until=lambda f: f.get("type") == "reply.start",
                           timeout=4.0)
            # B resets the shared session → A's waiting push must drop as well.
            await wsB.send_json({"type": "session.reset"})
            await _drain_until(wsB, "session.reset.ack", timeout=4.0)
            # A's user turn finishes and frees A's slot; without the peer epoch
            # bump A's waiting push would then wrongly speak the stale content.
            frames, _ = await _collect(wsA, timeout=3.0)
            spoken = {f.get("text") for f in frames
                      if f.get("type") == "reply" and f.get("push")}
            assert push_text not in spoken, (
                f"a peer reset must drop A's waiting push; got {spoken}")
            assert not srv._PENDING_PUSHES, list(srv._PENDING_PUSHES)
            await wsA.close()
            await wsB.close()

    asyncio.run(run())


def test_push_slot_claim_failure_degrades_to_text_not_concurrent():
    """When the reply slot stays occupied for the whole (bounded) wait window,
    the push must NOT speak concurrently with whatever holds the slot — it
    degrades to a persisted text bubble on deadline expiry (content never
    silently lost). Pins agent_task to a live dummy for the wait window."""
    _configure(tts=SlowTTS())
    push_text = "Wichtige Info."

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            # Grab the server-side state; pin the reply slot to a live task that
            # never finishes → the slot is taken for the push's whole window.
            state = list(srv.WS_CLIENTS.values())[0]

            async def _forever():
                await asyncio.sleep(30)

            dummy = asyncio.create_task(_forever())
            state.agent_task = dummy
            orig_wait = srv.PUSH_WAIT_TURN_S
            srv.PUSH_WAIT_TURN_S = 0.5   # shrink the bounded wait for the test
            try:
                await srv.handle_gateway_push(push_text)
                frames, _ = await _collect(ws, timeout=2.5)
            finally:
                srv.PUSH_WAIT_TURN_S = orig_wait
                dummy.cancel()

            push_starts = [f for f in frames
                           if f.get("type") == "reply.start" and f.get("push")]
            push_replies = [f for f in frames
                            if f.get("type") == "reply" and f.get("push")]
            assert not push_starts and not push_replies, (
                "a push must not speak while the reply slot is taken; saw "
                f"{_types(frames)}")
            bubbles = _push_texts_of(frames)
            assert any(b.get("text") == push_text and b.get("persist")
                       for b in bubbles), (
                "a deadline-expired push must degrade to a persisted text "
                f"bubble; saw {_types(frames)}")
            await ws.close()

    asyncio.run(run())


# ============================================================================ #
# B. Fragile/untested — pin CURRENT behavior (do NOT change it).
# ============================================================================ #
def test_push_reopens_wake_window_on_playback_done_PINNED():
    """DELIBERATE-OPEN DESIGN QUESTION (user decision pending): after an
    UNINVITED background push, the client's playback.done currently reopens the
    wake conversation window (reason='done') exactly like a real reply, leaving
    the mic wake-free. This PINS the current behavior — do NOT change it."""
    _configure()

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await ws.send_json({"type": "settings", "wakeWordEnabled": True})
            await _drain_until(ws, "settings.ack", timeout=2.0)
            await srv.handle_gateway_push("Hintergrund fertig.")
            await _drain_until(ws, "audio.end", timeout=5.0)
            # Client reports playback finished → current behavior reopens window.
            await ws.send_json({"type": "playback.done"})
            win, seen, _ = await _drain_until(ws, "wake.window", timeout=3.0)
            assert win is not None, (
                f"push playback.done currently reopens the wake window; saw {seen}")
            assert win.get("reason") == "done"
            await ws.close()

    asyncio.run(run())


def test_push_under_streaming_off_uses_synth_fallback():
    """STREAMING=0 with a synth-only TTS (no synth_stream): a push always goes
    through the streaming machinery, degrading via _tts_synth_stream's synth()
    fallback. End to end: a push reply frame plus streamed VCT2 audio."""
    _configure(streaming=False, tts=FakeTTS())   # FakeTTS has no synth_stream

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            await srv.handle_gateway_push("Auch ohne Streaming.")
            reply, seen, b1 = await _drain_until(ws, "reply", timeout=5.0)
            assert reply is not None, f"no reply; saw {seen}"
            assert reply.get("push") is True
            assert reply["text"] == "Auch ohne Streaming."
            end, seen2, b2 = await _drain_until(ws, "audio.end", timeout=5.0)
            assert end is not None, f"no audio.end; saw {seen + seen2}"
            # Streaming machinery is used even under STREAMING=0 → VCT2 chunks.
            assert (b1 or b2)[:4] == b"VCT2"
            await ws.close()

    asyncio.run(run())


def test_push_reaches_both_clients_bargein_cancels_only_one():
    """Multi-client: a fresh push reaches BOTH connected browsers; A's barge-in
    cancels ONLY A's delivery (A gets turn.discarded), while B keeps playing
    (no turn.discarded, its push reply completes normally)."""
    llm = RecordingLLM()
    _configure(tts=SlowTTS(chunk_delay=0.05, chunks=8), llm=llm)
    push_text = "An alle Geräte."

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            wsA = await client.ws_connect("/ws")
            await wsA.receive_json()  # hello
            wsB = await client.ws_connect("/ws")
            await wsB.receive_json()  # hello
            await srv.handle_gateway_push(push_text)
            # Ensure A's push is audibly streaming, then barge in on A only.
            # (wsB is intentionally NOT drained here: its whole push stream —
            # including the early `reply` frame — is collected in one pass below.)
            aA, _ = await _collect(
                wsA, until=lambda f: f.get("type") == "audio.start", timeout=4.0)
            assert aA and aA[-1]["type"] == "audio.start"
            await wsA.send_json({"type": "text.message", "text": "stopp mal"})
            dA, _ = await _collect(
                wsA, until=lambda f: f.get("type") == "turn.discarded", timeout=4.0)
            assert dA and dA[-1]["type"] == "turn.discarded"
            # B is unaffected: no turn.discarded, its push reply finishes.
            fB, _ = await _collect(
                wsB, until=lambda f: f.get("type") == "audio.end", timeout=6.0)
            assert fB and fB[-1]["type"] == "audio.end", (
                f"B's push must complete with audio.end; saw {_types(fB)}")
            assert "turn.discarded" not in _types(fB), (
                f"B must not be cancelled by A's barge-in; saw {_types(fB)}")
            assert any(f.get("type") == "reply" and f.get("push")
                       and f.get("text") == push_text for f in fB), (
                f"B's push reply must complete; saw {_types(fB)}")
            await wsA.close()
            await wsB.close()

    asyncio.run(run())


def test_flush_pending_pushes_drains_to_first_connector_only():
    """Drain-once contract: an offline-queued push is spoken by the FIRST
    browser that connects; a later second browser gets nothing from the queue."""
    _configure()

    async def run():
        await srv.handle_gateway_push("Wartende Nachricht.")   # queued, no client
        assert len(srv._PENDING_PUSHES) == 1

        async with TestClient(TestServer(srv.build_app())) as client:
            wsA = await client.ws_connect("/ws")
            await wsA.receive_json()  # hello
            rA, seenA, _ = await _drain_until(wsA, "reply", timeout=5.0)
            assert rA is not None, f"first connector must speak it; saw {seenA}"
            assert rA["text"] == "Wartende Nachricht." and rA.get("push")
            assert len(srv._PENDING_PUSHES) == 0
            # A second, later client gets nothing from the (already drained) queue.
            wsB = await client.ws_connect("/ws")
            await wsB.receive_json()  # hello
            rB, seenB, _ = await _drain_until(wsB, "reply", timeout=1.5)
            assert rB is None, (
                f"queue drains ONCE to the first connector only; B saw {seenB}")
            await wsA.close()
            await wsB.close()

    asyncio.run(run())


def test_queued_silent_notice_delivered_as_system_message_on_connect():
    """A speak=False push queued while no client is connected → on connect it
    is delivered as an external.message (source 'system'), with NO
    reply.start / audio frames (a silent notice, never synthesized)."""
    _configure()

    async def run():
        await srv.handle_gateway_push("♻️ Wartung erledigt", speak=False)
        assert len(srv._PENDING_PUSHES) == 1

        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.receive_json()  # hello
            evt, seen, binary = await _drain_until(
                ws, "external.message", timeout=4.0)
            assert evt is not None, f"no external.message; saw {seen}"
            assert evt["text"] == "♻️ Wartung erledigt"
            assert evt["source"] == "system"
            assert "reply.start" not in seen
            assert "audio.start" not in seen
            assert binary is None
            assert len(srv._PENDING_PUSHES) == 0
            await ws.close()

    asyncio.run(run())
