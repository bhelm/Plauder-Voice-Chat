"""Speaker-lock gate in the WS pipeline: only the enrolled owner voice is
transcribed; any other voice is dropped as transcript.ignored (speaker_mismatch),
without starting or cancelling a turn.
"""
import asyncio
import dataclasses

from aiohttp.test_utils import TestClient, TestServer

from plauder import server as srv
from plauder.config import Config
from plauder.sanitizer import HallucinationFilter
from plauder.session import ConversationManager
from plauder.speaker_verify import SpeakerVerifier, VerifyResult, WindowAnalysis

from test_wake_pipeline import (GatedStreamingLLM, ScriptedSTT, FakeTTS,
                                _drain_until, _collect, _send_voice)


class ScriptedVerifier:
    """Duck-typed SpeakerVerifier: one accept/reject decision per verify() call.
    ``windows`` (a WindowAnalysis or None) scripts the mixed-segment analysis."""
    loaded = True
    threshold = 0.5
    window_threshold = 0.3
    min_dur_s = 0.0
    windows = None

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = 0
        self._count = 1

    def active(self):
        return True

    def has_profile(self):
        return True

    def verify(self, pcm, sample_rate=None, duration_s=None):
        i = self.calls
        self.calls += 1
        d = self.decisions[i] if i < len(self.decisions) else True
        # A decision may be a plain bool or an (ok, score) tuple.
        if isinstance(d, tuple):
            ok, score = d
        else:
            ok, score = d, (0.9 if d else 0.1)
        return VerifyResult(ok, score, "match" if ok else "mismatch")

    def analyze_windows(self, pcm, sample_rate=None, **_kw):
        return self.windows

    # Equal-length block scores for the trim/rescue gate.
    blocks = None

    def analyze_blocks(self, pcm, sample_rate=None, **_kw):
        return self.blocks or []

    # Block-level second opinion: scripted scores consumed in call order
    # (kept spans first, then cut gaps); empty list → 0.0 (refine disabled).
    region_scores = None

    def score_region(self, pcm, sample_rate=None, start_s=0.0, end_s=None):
        if self.region_scores:
            return self.region_scores.pop(0)
        return 0.0

    # Owner-watch: scripted per-window verdicts (True/False/None), consumed in
    # order; empty/exhausted list → False (foreign).
    window_owner = None

    def window_is_owner(self, pcm, sample_rate=None, **_kw):
        if self.window_owner:
            return self.window_owner.pop(0)
        return False


def _configure_speaker(stt_texts, deltas, decisions, *, gate_open=True, **cfg_overrides):
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=20, streaming=True,
                              **cfg_overrides)
    llm = GatedStreamingLLM(deltas)
    if gate_open:
        llm.gate.set()  # don't block the reply
    conv = ConversationManager(llm, system_prompt="sys")
    srv.configure(cfg, stt=ScriptedSTT(stt_texts), tts=FakeTTS(), conv=conv,
                  bridge=None, ghost=HallucinationFilter(enabled=False),
                  speaker=ScriptedVerifier(decisions))
    return cfg, llm


def test_impostor_voice_is_ignored_owner_passes():
    _configure_speaker(
        stt_texts=["ein fremder satz", "erzähl mir was"],
        deltas=["Gerne. ", "Hier kommt es."],
        decisions=[False, True],   # 1st segment: not the owner; 2nd: owner
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")

            # Segment 1: an impostor → ignored, no reply.
            await _send_voice(ws, "s1")
            ig, seen1 = await _drain_until(ws, "transcript.ignored")
            assert ig is not None and ig["reason"] == "speaker_mismatch", seen1
            assert "reply.start" not in seen1, seen1

            # Segment 2: the owner → normal turn.
            await _send_voice(ws, "s2")
            rs, seen2 = await _drain_until(ws, "reply.start")
            assert rs is not None, seen2
            await ws.close()

    asyncio.run(run())


# ~1.0 s of 16 kHz f32 audio — BELOW SPEAKER_BARGE_MIN_S (1.5 s): streaming it
# does NOT trigger an early speaker check (prefix scores are unreliable).
_ONE_SECOND_F32 = b"\x00" * (16000 * 4)
# ~1.6 s — above the early-check floor: exactly one mid-stream barge check.
_BARGE_LEN_F32 = b"\x00" * int(16000 * 4 * 1.6)


def test_foreign_voice_does_not_interrupt_inflight_reply():
    """Core of the reported bug: a reply is playing, a FOREIGN voice speaks →
    the reply must NOT be cancelled (no turn.discarded/audio.stop); the foreign
    segment is dropped at commit and the reply finishes unharmed."""
    _cfg, llm = _configure_speaker(
        stt_texts=["erzähl eine geschichte", "fremder einwurf"],
        deltas=["Erster Teil. ", "Zweiter Teil."],
        # s1 commit-verify: owner; s2 early check + commit-verify: foreign.
        decisions=[True, False, False],
        gate_open=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")                    # owner starts a turn
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            # Foreign voice streams in while the reply is in flight (B1).
            await ws.send_json({"type": "segment.stream.start", "segmentId": "s2"})
            await ws.send_bytes(_ONE_SECOND_F32)           # → early check: mismatch
            seen = await _collect(ws, 0.5)
            assert "turn.discarded" not in seen, f"Fremdstimme hat unterbrochen: {seen}"
            assert "audio.stop" not in seen, f"Audio gestoppt trotz Fremdstimme: {seen}"

            await ws.send_json({"type": "segment.stream.commit", "segmentId": "s2"})
            ig, seen2 = await _drain_until(ws, "transcript.ignored")
            assert ig is not None and ig["reason"] == "speaker_mismatch", seen2

            llm.gate.set()                                  # release the reply
            reply, seen3 = await _drain_until(ws, "reply")
            assert reply is not None, f"Antwort abgebrochen? {seen3}"
            assert reply["text"] == "Erster Teil. Zweiter Teil."
            await ws.close()

    asyncio.run(run())


def test_rejected_segment_does_not_redispatch_running_turn():
    """Regression: a foreign segment dropped while the owner's turn is running
    must NOT re-arm the debounce — that would dispatch the SAME user input to
    the LLM a second time (duplicate turn.commit/reply.start, garbled replies,
    duplicate history entries)."""
    _cfg, llm = _configure_speaker(
        stt_texts=["erzähl was", "kindergeplapper"],
        deltas=["Antwort läuft. ", "Fertig."],
        decisions=[True, False],   # s1: owner; s2: foreign (non-stream, no early check)
        gate_open=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            # Foreign voice while the reply is in flight → rejected at commit.
            await _send_voice(ws, "s2")
            ig, _seen = await _drain_until(ws, "transcript.ignored")
            assert ig is not None and ig["reason"] == "speaker_mismatch"

            # Wait well past the 20 ms debounce: no SECOND dispatch may happen.
            seen = await _collect(ws, 0.4)
            assert "turn.commit" not in seen, f"Turn doppelt dispatcht: {seen}"
            assert "reply.start" not in seen, f"Zweiter reply.start: {seen}"

            llm.gate.set()
            reply, seen2 = await _drain_until(ws, "reply")
            assert reply is not None and reply["text"] == "Antwort läuft. Fertig.", seen2
            await ws.close()

    asyncio.run(run())


def test_owner_voice_interrupts_mid_stream():
    """Counter-check: the OWNER's voice must still interrupt — and already
    mid-stream (early check on the growing buffer), before the segment commits."""
    _cfg, llm = _configure_speaker(
        stt_texts=["frag was", "erzähl was anderes"],
        deltas=["Lange Antwort Teil eins. ", "Teil zwei."],
        # s1 commit-verify: owner; s2 early check: owner; s2 commit RE-verifies
        # (the gate never trusts prefix scores alone anymore).
        decisions=[True, True, True],
        gate_open=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            # Owner speaks while the reply is in flight → cancel BEFORE commit.
            await ws.send_json({"type": "segment.stream.start", "segmentId": "s2"})
            await ws.send_bytes(_BARGE_LEN_F32)
            disc, seen = await _drain_until(ws, "turn.discarded", timeout=3.0)
            assert disc is not None, f"Owner-Stimme unterbrach nicht (mid-stream): {seen}"

            # The interrupting utterance becomes the next command.
            llm.gate.set()
            await ws.send_json({"type": "segment.stream.commit", "segmentId": "s2"})
            reply, seen2 = await _drain_until(ws, "reply", timeout=3.0)
            assert reply is not None, f"Folge-Kommando ohne Antwort: {seen2}"
            await ws.close()

    asyncio.run(run())


def test_vad_barge_in_message_ignored_manual_allowed():
    """With the lock engaged, a client 'barge_in' from the VAD (defense in
    depth / old clients) must not cancel; the deliberate stop button must."""
    _cfg, llm = _configure_speaker(
        stt_texts=["frag was"], deltas=["Teil eins. ", "Teil zwei."],
        decisions=[True], gate_open=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            await ws.send_json({"type": "barge_in", "reason": "vad-speech", "playing": True})
            seen = await _collect(ws, 0.4)
            assert "turn.discarded" not in seen, f"VAD-barge_in unterbrach trotz Lock: {seen}"

            await ws.send_json({"type": "barge_in", "reason": "manual", "playing": True})
            disc, seen2 = await _drain_until(ws, "turn.discarded")
            assert disc is not None, f"Stop-Button unterbrach nicht: {seen2}"
            await ws.close()

    asyncio.run(run())


# A 4 s segment (≥ SPEAKER_TRIM_MIN_S) — triggers the windowed mixed-voice path.
_FOUR_SECONDS_F32 = b"\x00" * (16000 * 4 * 4)


def test_mixed_segment_trimmed_to_owner_spans():
    """Sequential mix (owner speaks, kids keep talking into the segment): the
    full-segment verify fails (diluted), the windowed RESCUE crops the owner's
    spans and re-transcribes just those."""
    _configure_speaker(
        stt_texts=["mein teil und kindergeplapper", "nur mein teil"],
        deltas=["Ok!"], decisions=[False])   # full verify fails → rescue path
    # Equal-length blocks: owner block clears the floor (anchor), the
    # foreign tail ≥ 2.5 s scores far below best → rescued by cropping.
    srv.SPEAKER.blocks = [(0.0, 3.0, 0.45), (1.5, 4.5, 0.15), (3.0, 6.0, 0.12)]

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "segment.start", "segmentId": "s1"})
            await ws.send_bytes(b"\x00" * (16000 * 4 * 6))
            tr, seen = await _drain_until(ws, "transcript")
            assert tr is not None and tr["text"] == "nur mein teil", seen
            assert tr.get("speakerTrimmed") is True
            commit, seen2 = await _drain_until(ws, "turn.commit")
            assert commit is not None and commit["text"] == "nur mein teil", seen2
            await ws.close()

    asyncio.run(run())


def test_accepted_segment_relative_trim_cuts_interjection():
    """Accept path: the owner's voice dominates (full verify passes), but a
    contiguous low-scoring block (someone chiming in) is cut out via the
    RELATIVE window rule."""
    _configure_speaker(
        stt_texts=["mein satz und ein zwischenruf", "nur mein satz"],
        deltas=["Ok!"], decisions=[True])   # full verify passes
    # 6 s segment: equal-length blocks — owner at the start (0.50), the
    # foreign passage ≥ 2.5 s scores far below best (block-relative).
    srv.SPEAKER.blocks = [(0.0, 3.0, 0.50), (1.5, 4.5, 0.20), (3.0, 6.0, 0.15)]

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "segment.start", "segmentId": "s1"})
            await ws.send_bytes(b"\x00" * (16000 * 4 * 6))
            tr, seen = await _drain_until(ws, "transcript")
            assert tr is not None and tr["text"] == "nur mein satz", seen
            assert tr.get("speakerTrimmed") is True
            assert srv.STT.calls == 2          # full + cropped re-STT
            await ws.close()

    asyncio.run(run())


def test_short_low_scoring_word_passes_while_composing():
    """Field case: the owner speaks (part 1 pending), says one more short word
    that scores badly (short-clip embeddings are erratic) — it must PASS and
    join the turn instead of being rejected (which let the debounce fire
    mid-thought → premature submit)."""
    _cfg, llm = _configure_speaker(
        stt_texts=["erster teil", "voicetest"],
        deltas=["Ok!"],
        # part 1: strict match; the short word: badly scoring mismatch.
        decisions=[(True, 0.55), (False, 0.15)])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")                    # part 1 → pending
            tr1, _ = await _drain_until(ws, "transcript")
            assert tr1 is not None and tr1["text"] == "erster teil"
            # Short word (~1.2 s) while composing — before the debounce fires.
            await ws.send_json({"type": "segment.start", "segmentId": "s2"})
            await ws.send_bytes(b"\x00" * int(16000 * 4 * 1.2))
            tr2, seen = await _drain_until(ws, "transcript")
            assert tr2 is not None and tr2["text"] == "voicetest", seen
            assert "transcript.ignored" not in seen, f"Kurzwort verworfen: {seen}"
            commit, seen2 = await _drain_until(ws, "turn.commit", timeout=3.0)
            assert commit is not None and commit["text"] == "erster teil. Voicetest", seen2
            await ws.close()

    asyncio.run(run())


def test_owner_watch_veto_when_tail_is_same_speaker():
    """Field regression: window scores dipped mid-sentence and the watch cut
    the owner after 2.4 s WHILE they were still talking. The block-level veto
    (tail scores close to head) must prevent the premature auto-commit."""
    _configure_speaker(stt_texts=["ganzer satz am stück"], deltas=["Ok!"],
                       decisions=[True])
    srv.SPEAKER.window_owner = [True, False, False]    # noisy dip after 1.2 s
    srv.SPEAKER.region_scores = [0.55, 0.50]           # head ≈ tail → same speaker

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "segment.stream.start", "segmentId": "s1"})
            frame = b"\x00" * int(16000 * 4 * 0.6)
            for _ in range(6):                          # 3.6 s, no client commit
                await ws.send_bytes(frame)
                await asyncio.sleep(0.05)
            seen = await _collect(ws, 0.5)
            assert "transcript" not in seen, f"Veto versagt — vorzeitiger Cut: {seen}"
            # Real client commit → the whole utterance processed as one piece.
            await ws.send_json({"type": "segment.stream.commit", "segmentId": "s1"})
            tr, seen2 = await _drain_until(ws, "transcript", timeout=3.0)
            assert tr is not None and tr["text"] == "ganzer satz am stück", seen2
            await ws.close()

    asyncio.run(run())


def test_own_followup_before_audio_coalesces_not_drops():
    """Field case: turn A is committed, the LLM is still thinking (no audio
    yet), the owner keeps talking → turn A must NOT vanish. Its input is
    re-queued (turn.discarded carries coalesced=true) and the next turn
    combines old + new text."""
    _cfg, llm = _configure_speaker(
        stt_texts=["erster teil", "zweiter teil"],
        deltas=["Moment"],           # no sentence end → no TTS audio before the gate
        decisions=[(True, 0.6), (True, 0.6)],
        gate_open=False)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")
            c1, seen0 = await _drain_until(ws, "turn.commit")
            assert c1 is not None and c1["text"] == "erster teil", seen0
            assert (await _drain_until(ws, "reply.start"))[0] is not None

            await _send_voice(ws, "s2")      # owner keeps talking, no audio yet
            disc, seen = await _drain_until(ws, "turn.discarded")
            assert disc is not None and disc.get("coalesced") is True, seen

            c2, seen2 = await _drain_until(ws, "turn.commit", timeout=3.0)
            assert c2 is not None and c2["text"] == "erster teil. Zweiter teil", seen2

            llm.gate.set()
            assert (await _drain_until(ws, "reply"))[0] is not None
            await ws.close()

    asyncio.run(run())


def test_continuity_accepts_own_tail_after_owner_watch_split():
    """The reported field case: the owner's sentence tail becomes its own
    segment (owner-watch split) and scores a hair under the threshold — the
    temporal-continuity rule (last strict match − Δ) must accept it, while a
    genuinely foreign segment stays rejected."""
    _configure_speaker(
        stt_texts=["mein anfang", "und mein nachsatz", "fremde stimme"],
        deltas=["Eins. ", "Zwei. ", "Drei."],
        # s1: strict match at 0.52; s2: 0.45 (< thr 0.5, but ≥ 0.52−0.12);
        # s3: foreign at 0.20 (< bar) → rejected despite continuity window.
        decisions=[(True, 0.52), (False, 0.45), (False, 0.20)])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")
            assert (await _drain_until(ws, "reply"))[0] is not None

            await _send_voice(ws, "s2")           # tail, slightly under threshold
            tr, seen = await _drain_until(ws, "transcript")
            assert tr is not None and tr["text"] == "und mein nachsatz", seen
            assert (await _drain_until(ws, "reply"))[0] is not None

            await _send_voice(ws, "s3")           # foreign → still rejected
            ig, seen3 = await _drain_until(ws, "transcript.ignored")
            assert ig is not None and ig["reason"] == "speaker_mismatch", seen3
            await ws.close()

    asyncio.run(run())


def test_block_refine_uncuts_own_trailing_speech():
    """Windows propose cutting the tail, but the block-level second opinion
    scores the tail close to the kept block → un-cut, full text kept, no
    extra STT call."""
    _configure_speaker(stt_texts=["alles von mir gesprochen"], deltas=["Ok!"],
                       decisions=[(True, 0.6)])
    # Equal-length blocks all within Δ of each other → same speaker, no cut.
    srv.SPEAKER.blocks = [(0.0, 3.0, 0.50), (1.5, 4.0, 0.42)]

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "segment.start", "segmentId": "s1"})
            await ws.send_bytes(_FOUR_SECONDS_F32)
            tr, seen = await _drain_until(ws, "transcript")
            assert tr is not None and tr["text"] == "alles von mir gesprochen", seen
            assert not tr.get("speakerTrimmed"), seen
            assert srv.STT.calls == 1, "un-cut must not trigger a re-STT"
            await ws.close()

    asyncio.run(run())


def test_long_foreign_segment_rejected_via_windows():
    """Long foreign segment: full verify fails AND the rescue finds no owner
    windows → rejected, no extra STT."""
    _configure_speaker(stt_texts=["nur die kinder"], deltas=["x"],
                       decisions=[(False, 0.2)])
    # Best block 0.18 < floor (window_threshold 0.3) → no rescue at all.
    srv.SPEAKER.blocks = [(0.0, 3.0, 0.15), (1.5, 4.0, 0.18)]

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "segment.start", "segmentId": "s1"})
            await ws.send_bytes(_FOUR_SECONDS_F32)
            ig, seen = await _drain_until(ws, "transcript.ignored")
            assert ig is not None and ig["reason"] == "speaker_mismatch", seen
            assert srv.STT.calls == 1        # no re-STT on a full reject
            await ws.close()

    asyncio.run(run())


def test_long_pure_owner_segment_keeps_full_text():
    """Long segment, every voiced window is the owner → full text, ONE STT call."""
    _configure_speaker(stt_texts=["alles von mir"], deltas=["Ok!"], decisions=[])
    srv.SPEAKER.windows = WindowAnalysis(
        spans=[(0.0, 4.0)], owner_ratio=1.0, voiced=6, score=0.9)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "segment.start", "segmentId": "s1"})
            await ws.send_bytes(_FOUR_SECONDS_F32)
            tr, seen = await _drain_until(ws, "transcript")
            assert tr is not None and tr["text"] == "alles von mir", seen
            assert not tr.get("speakerTrimmed"), seen
            assert srv.STT.calls == 1
            await ws.close()

    asyncio.run(run())


def test_owner_watch_auto_commits_when_others_keep_talking():
    """Segmentation fix: the owner speaks, then OTHER voices keep the client
    VAD open — no client commit ever arrives. The server must detect the
    owner's end on the streaming buffer and auto-commit the utterance."""
    _configure_speaker(stt_texts=["mein satz"], deltas=["Ok!"], decisions=[True])
    srv.SPEAKER.window_owner = [True, False, False]   # [0,1.2)=owner, then foreign
    # Block veto consulted before the cut: head clearly owner, tail clearly
    # foreign → the cut proceeds.
    srv.SPEAKER.region_scores = [0.60, 0.20]

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "segment.stream.start", "segmentId": "s1"})
            # Stream 3.6 s in 0.6 s frames — deliberately NO stream.commit
            # (the "kids keep talking" case). The watch cut needs ≥ 2 s of
            # foreign tail past the owner's last window (lazier than debounce).
            frame = b"\x00" * int(16000 * 4 * 0.6)
            for _ in range(6):
                await ws.send_bytes(frame)
                await asyncio.sleep(0.05)   # let the watcher steps run
            tr, seen = await _drain_until(ws, "transcript", timeout=3.0)
            assert tr is not None and tr["text"] == "mein satz", seen
            assert tr["segmentId"] == "s1"
            reply, seen2 = await _drain_until(ws, "reply", timeout=3.0)
            assert reply is not None, seen2
            await ws.close()

    asyncio.run(run())


def test_ptt_segment_never_hard_rejected():
    """PTT = deliberate button press: a failed verify must fail OPEN (the
    command goes through), never drop the segment."""
    _configure_speaker(stt_texts=["mein befehl"], deltas=["Ok!"], decisions=[False])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "segment.start", "segmentId": "s1", "ptt": True})
            await ws.send_bytes(b"\x00" * 16000)     # 0.25 s → full-verify path
            tr, seen = await _drain_until(ws, "transcript")
            assert tr is not None and tr["text"] == "mein befehl", seen
            reply, seen2 = await _drain_until(ws, "reply", timeout=3.0)
            assert reply is not None, seen2
            await ws.close()

    asyncio.run(run())


def test_hello_advertises_speaker_lock():
    _configure_speaker(stt_texts=[], deltas=["x"], decisions=[])

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            hello, _ = await _drain_until(ws, "hello")
            sl = hello.get("speakerLock")
            assert sl and sl["available"] is True and sl["enrolled"] is True
            await ws.close()

    asyncio.run(run())


def test_enrollment_over_ws_updates_profile(tmp_path):
    """The enroll.start/frames/commit path builds a profile and acks ok."""
    cfg = dataclasses.replace(Config.from_env(), debounce_ms=20, streaming=True)
    conv = ConversationManager(GatedStreamingLLM(["x"]), system_prompt="sys")

    # Real SpeakerVerifier with a fake extractor (no model file needed).
    from test_speaker_verify import _FakeExtractor
    sv = SpeakerVerifier(model_path="fake.onnx",
                         profile_path=str(tmp_path / "p.json"), min_dur_s=0.1)
    sv._extractor = _FakeExtractor()
    sv._dim = 4
    srv.configure(cfg, stt=ScriptedSTT([]), tts=FakeTTS(), conv=conv, bridge=None,
                  ghost=HallucinationFilter(enabled=False), speaker=sv)

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await ws.send_json({"type": "speaker.enroll.start"})
            # ~0.5 s of 16 kHz f32 audio (well above min_dur_s).
            await ws.send_bytes(b"\x01\x00\x00\x00" * 8000)
            await ws.send_json({"type": "speaker.enroll.commit"})
            ack, seen = await _drain_until(ws, "speaker.enroll.ack")
            assert ack is not None and ack["ok"] is True and ack["count"] == 1, seen
            assert sv.has_profile()
            await ws.close()

    asyncio.run(run())


def test_speaker_dump_writes_wavs(tmp_path):
    """With SPEAKER_DUMP_DIR set, every gated segment lands as a WAV whose
    filename carries the decision (ok/rej) — the raw material for replaying
    gate decisions offline."""
    _configure_speaker(
        stt_texts=["fremder satz", "mein satz"],
        deltas=["Ok. "],
        decisions=[False, True],
        speaker_dump_dir=str(tmp_path / "dump"),
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")                       # impostor → rej
            await _drain_until(ws, "transcript.ignored")
            await _send_voice(ws, "s2")                       # owner → ok
            await _drain_until(ws, "reply.start")
            await ws.close()

    asyncio.run(run())
    names = sorted(p.name for p in (tmp_path / "dump").glob("*.wav"))
    assert len(names) == 2, names
    assert any("_rej_" in n for n in names) and any("_ok_" in n for n in names)
    # WAV header sanity: RIFF magic present.
    for p in (tmp_path / "dump").glob("*.wav"):
        assert p.read_bytes()[:4] == b"RIFF"


def test_owner_barge_stops_client_buffered_playback():
    """Regression (real field failure): a fast TTS ships ALL chunks within
    ~1-2 s (audio.end sent, audio_ids emptied) while the client keeps PLAYING
    its buffer for many more seconds. When the owner then speaks, the server
    used to see "nothing to interrupt" — no early barge, no audio.stop, the
    reply talked over the user. Now the server tracks client playback until
    playback.done and stops it with a bare audio.stop."""
    _configure_speaker(
        stt_texts=["erzähl was langes", "stopp mal, neue frage"],
        deltas=["Ein sehr langer Satz. "],
        decisions=[True, True],
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")
            # Reply fully generated + all chunks sent; NO playback.done sent —
            # the client is still playing its buffer.
            assert (await _drain_until(ws, "audio.end"))[0] is not None

            # Owner speaks again → gate passes → buffered playback must stop.
            await _send_voice(ws, "s2")
            stop, seen = await _drain_until(ws, "audio.stop")
            assert stop is not None, f"kein audio.stop trotz Owner-Barge: {seen}"
            await ws.close()

    asyncio.run(run())


def test_playback_done_clears_barge_target():
    """After playback.done nothing is playing — a later owner segment must NOT
    emit a spurious audio.stop."""
    _configure_speaker(
        stt_texts=["frage eins", "frage zwei"],
        deltas=["Antwort eins. ", "Antwort zwei. "],
        decisions=[True, True],
    )

    async def run():
        async with TestClient(TestServer(srv.build_app())) as client:
            ws = await client.ws_connect("/ws")
            await _drain_until(ws, "hello")
            await _send_voice(ws, "s1")
            end1, _ = await _drain_until(ws, "audio.end")
            await ws.send_json({"type": "playback.done", "turnId": end1["turnId"],
                                "audioId": end1.get("audioId"), "ts": 0})
            await _send_voice(ws, "s2")
            reply, seen = await _drain_until(ws, "reply")
            assert reply is not None
            assert "audio.stop" not in seen, f"unnötiger audio.stop: {seen}"
            await ws.close()

    asyncio.run(run())
