"""Audio utilities: PCM/WAV conversion, frame wrapping, TTS chunking."""
import numpy as np

from plauder import audio


def test_pcm_bytes_roundtrip_float32():
    arr = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    out = audio.pcm_bytes_to_float32_array(arr.tobytes())
    assert np.allclose(out, arr)


def test_float32_to_wav_has_riff_header():
    samples = np.zeros(160, dtype=np.float32)
    wav = audio.float32_to_pcm16_wav_bytes(samples, 16000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_pcm16_to_wav_bytes():
    pcm = np.array([0, 16384, -16384], dtype=np.int16).tobytes()
    wav = audio.pcm16_to_wav_bytes(pcm, 24000)
    assert wav[:4] == b"RIFF"
    # 3 frames * 2 bytes data
    assert len(wav) == 44 + 6


def test_pcm16_bytes_to_float32_scaling():
    pcm = np.array([0, 16384, -16384], dtype=np.int16).tobytes()
    out = audio.pcm16_bytes_to_float32(pcm)
    assert abs(out[1] - 0.5) < 1e-3
    assert abs(out[2] + 0.5) < 1e-3


def test_wrap_wav_with_turn_id():
    wav = b"RIFFxxxx"
    framed = audio.wrap_wav_with_turn_id(wav, "abc123")
    assert framed.startswith(audio.WAV_FRAME_MAGIC)
    idlen = framed[4]
    assert framed[5:5 + idlen] == b"abc123"
    assert framed[5 + idlen:] == wav


def test_split_text_short_returns_single():
    out = audio.split_text_for_tts("Hallo Welt.", 200)
    assert out == ["Hallo Welt."]


def test_split_text_empty():
    assert audio.split_text_for_tts("", 200) == []


def test_split_text_splits_long_text():
    long = ". ".join([f"Satz nummer {i} ist hier" for i in range(20)]) + "."
    out = audio.split_text_for_tts(long, 60)
    assert len(out) > 1
    assert all(len(p) <= 60 for p in out)


def test_split_text_merges_short_sentences():
    out = audio.split_text_for_tts("Ja. Nein. Vielleicht.", 200)
    # Short sentences get merged into a single chunk.
    assert out == ["Ja. Nein. Vielleicht."]


# --- VCT2 PCM-Chunk-Frame ---------------------------------------------------
def test_wrap_pcm_chunk_header():
    pcm = b"\x01\x02\x03\x04"
    framed = audio.wrap_pcm_chunk("turn42", 5, pcm)
    assert framed[:4] == audio.PCM_CHUNK_MAGIC
    idlen = framed[4]
    assert framed[5:5 + idlen] == b"turn42"
    seq = (framed[5 + idlen] << 8) | framed[6 + idlen]
    assert seq == 5
    assert framed[7 + idlen:] == pcm


def test_iter_pcm_frames_even_sizes_and_coverage():
    pcm = bytes(range(20))           # 20 bytes
    frames = list(audio.iter_pcm_frames(pcm, 7))   # 7 is odd → becomes 6
    assert all(len(f) % 2 == 0 for f in frames)
    assert b"".join(frames) == pcm   # lossless


def test_iter_pcm_frames_single_when_frame_large():
    pcm = bytes(8)
    assert list(audio.iter_pcm_frames(pcm, 1000)) == [pcm]


# --- Streaming sentence splitter --------------------------------------------
def test_split_stream_emits_complete_sentences_keeps_tail():
    sents, rest = audio.split_stream_sentences("Hallo Welt. Wie geht es dir", 200)
    assert sents == ["Hallo Welt."]
    assert rest == "Wie geht es dir"


def test_split_stream_holds_incomplete_sentence():
    # No punctuation + whitespace at the end → flush nothing, all in the rest.
    sents, rest = audio.split_stream_sentences("Noch kein Satzende", 200)
    assert sents == []
    assert rest == "Noch kein Satzende"


def test_split_stream_force_flush_when_too_long():
    buf = "ein sehr langer satz ohne jedes satzzeichen der einfach weiterlaeuft und laeuft"
    sents, rest = audio.split_stream_sentences(buf, 30)
    assert sents, "zu langer Satz muss zwangs-geflusht werden"
    assert all(len(s) <= 30 for s in sents)


def test_split_stream_multiple_sentences_at_once():
    sents, rest = audio.split_stream_sentences("Eins. Zwei! Drei? ", 200)
    assert sents == ["Eins.", "Zwei!", "Drei?"]
    assert rest == ""


def test_split_stream_sentences_keeps_abbreviations():
    """Abbreviations/ordinals are not sentence boundaries mid-stream: "z.B."
    must not ship as its own TTS sentence."""
    from plauder.audio import split_stream_sentences
    sents, rest = split_stream_sentences("z.B. eine Sache, die noch geht ", 220)
    assert sents == [] and rest.startswith("z.B.")
    sents, rest = split_stream_sentences("Das gilt z.B. hier. Und weiter ", 220)
    assert sents == ["Das gilt z.B. hier."]
    sents, rest = split_stream_sentences("1. Erstens kommt das. 2. Zweitens ", 220)
    assert sents == ["1. Erstens kommt das."] and rest.startswith("2.")


def test_pcm_bytes_to_float32_tolerates_truncated_buffer():
    import numpy as np
    from plauder.audio import pcm_bytes_to_float32_array, crop_f32_spans
    buf = np.zeros(10, np.float32).tobytes() + b"\x01\x02"   # 2 trailing bytes
    assert pcm_bytes_to_float32_array(buf).shape[0] == 10
    assert crop_f32_spans(buf, 10, [(0.0, 1.0)]) != b""


def test_crop_f32_spans_merges_overlaps():
    import numpy as np
    from plauder.audio import crop_f32_spans
    seg = np.arange(10, dtype=np.float32)
    out = np.frombuffer(crop_f32_spans(seg.tobytes(), 1, [(4.0, 8.0), (0.0, 5.0)]),
                        dtype=np.float32)
    assert out.tolist() == list(range(8))   # sorted + merged, no duplication


def test_split_stream_force_cut_prefers_clause_boundary():
    buf = ("Das ist ein sehr langer Satz ohne Punkt, aber mit einem Komma "
           "und dann geht es immer weiter ohne jede Pause")
    sents, rest = audio.split_stream_sentences(buf, 60)
    assert sents[0].endswith(",")            # cut at the clause boundary
    assert sents[1].startswith("aber")       # nothing lost across the cut


def test_split_stream_force_cut_word_boundary_without_clause():
    buf = "wort " * 30                        # no punctuation at all
    sents, rest = audio.split_stream_sentences(buf.strip(), 30)
    assert sents                              # still force-flushed
    assert all(len(s) <= 30 for s in sents)


# --- StreamLeadGate (mutes the TTS lead-noise blob) ---------------------------
def _pcm16(samples):
    return np.asarray(samples, dtype=np.int16).tobytes()


def test_lead_gate_mutes_blob_keeps_speech():
    sr = 16000
    gate = audio.StreamLeadGate(sr, threshold_db=-45.0, prepad_ms=50)
    silence = _pcm16(np.zeros(sr // 10))                    # 100 ms silence
    blob = _pcm16((np.random.default_rng(1).standard_normal(sr // 10)
                   * 32767 * 0.002).astype(np.int16))       # ~-54 dB noise
    speech = _pcm16((np.sin(np.linspace(0, 300, sr // 2))
                     * 32767 * 0.3).astype(np.int16))       # loud tone
    out = gate.process(silence) + gate.process(blob) + gate.process(speech)
    out += gate.flush()
    total_in = len(silence) + len(blob) + len(speech)
    assert len(out) == total_in                             # duration preserved
    arr = np.frombuffer(out, dtype=np.int16)
    n_lead = (len(silence) + len(blob)) // 2
    # Blob region is silenced (up to the 50 ms pre-pad before the onset).
    prepad = int(sr * 0.05)
    assert np.max(np.abs(arr[: n_lead - prepad])) == 0
    # Speech region passes unmodified.
    assert np.array_equal(arr[n_lead:], np.frombuffer(speech, dtype=np.int16))
    assert gate.opened


def test_lead_gate_all_quiet_becomes_silence():
    sr = 16000
    gate = audio.StreamLeadGate(sr)
    blob = _pcm16((np.random.default_rng(2).standard_normal(sr)
                   * 32767 * 0.001).astype(np.int16))       # ~-60 dB only
    out = gate.process(blob) + gate.flush()
    assert len(out) == len(blob)
    assert np.max(np.abs(np.frombuffer(out, dtype=np.int16))) == 0
    assert not gate.opened


def test_lead_gate_open_passes_through_verbatim():
    sr = 16000
    gate = audio.StreamLeadGate(sr)
    speech = _pcm16((np.sin(np.linspace(0, 100, sr // 4))
                     * 32767 * 0.5).astype(np.int16))
    first = gate.process(speech)
    assert gate.opened
    tail = _pcm16([1, 2, 3, 4])
    assert gate.process(tail) == tail                       # verbatim once open
    assert gate.flush() == b""


# --- voice-clone reference cleanup ------------------------------------------
_SR = 16000


def _tone(dur_s: float, amp: float = 0.25) -> np.ndarray:
    n = int(_SR * dur_s)
    return (np.sin(2 * np.pi * 220.0 * np.arange(n) / _SR) * amp).astype(np.float32)


def _sil(dur_s: float) -> np.ndarray:
    return np.zeros(int(_SR * dur_s), dtype=np.float32)


def _trim(parts):
    raw = np.concatenate(parts).astype(np.float32).tobytes()
    return audio.trim_clone_reference(raw, _SR)


def test_trim_clean_recording_keeps_speech_trims_silence():
    out, info = _trim([_sil(0.8), _tone(3.0), _sil(0.8)])
    assert out is not None
    assert not info["dropped_head"] and not info["dropped_tail"]
    dur = len(out) / 4 / _SR
    assert abs(dur - 3.4) < 0.15                 # speech + 2×200 ms pad
    assert abs(info["head_cut_s"] - 0.6) < 0.1   # 800 ms silence → 200 ms pad


def test_trim_drops_half_word_at_start():
    out, info = _trim([_tone(0.4), _sil(0.6), _tone(3.0), _sil(0.5)])
    assert out is not None
    assert info["dropped_head"] and not info["dropped_tail"]
    dur = len(out) / 4 / _SR
    assert abs(dur - 3.4) < 0.15                 # fragment + its gap are gone
    samples = np.frombuffer(out, dtype=np.float32)
    assert np.abs(samples[: int(_SR * 0.1)]).max() < 1e-3   # starts in the pad


def test_trim_drops_half_word_at_end():
    out, info = _trim([_sil(0.5), _tone(3.0), _sil(0.6), _tone(0.4)])
    assert out is not None
    assert info["dropped_tail"] and not info["dropped_head"]
    assert abs(len(out) / 4 / _SR - 3.4) < 0.15


def test_trim_rejects_when_only_edge_speech_remains():
    out, info = _trim([_tone(1.0), _sil(0.5), _tone(1.0)])   # touches both edges
    assert out is None
    assert info["reason"] == "edge_speech"
    assert info["dropped_head"] and info["dropped_tail"]


def test_trim_rejects_silence_only():
    out, info = _trim([_sil(4.0)])
    assert out is None
    assert info["reason"] == "no_speech"


def test_trim_rejects_too_little_speech_left():
    out, info = _trim([_sil(1.0), _tone(0.5), _sil(1.0)])
    assert out is None
    assert info["reason"] == "too_short"


def test_trim_ignores_click_blips():
    parts = [_sil(0.5), _tone(0.04, amp=0.9), _sil(0.5), _tone(3.0), _sil(0.5)]
    out, info = _trim(parts)
    assert out is not None
    # the 40 ms click neither counts as speech nor extends the kept span
    assert abs(len(out) / 4 / _SR - 3.4) < 0.15


def test_trim_overrun_cuts_at_breath_pause_inside_region():
    # Speech runs over the END with no >=120 ms pause — but an 80 ms breath
    # pause sits before the final (cut-off) word. Only that word must go,
    # not the whole region.
    out, info = _trim([_sil(1.0), _tone(3.0), _sil(0.08), _tone(0.4)])
    assert out is not None
    assert info["dropped_tail"] and not info["dropped_head"]
    # head pad 200 ms + 3 s speech + tail pad clamped to ~60% of the 80 ms gap
    assert abs(len(out) / 4 / _SR - 3.25) < 0.15
    assert abs(info["kept_s"] - 3.0) < 0.15


def test_trim_overrun_at_start_cuts_at_breath_pause():
    out, info = _trim([_tone(0.4), _sil(0.08), _tone(3.0), _sil(1.0)])
    assert out is not None
    assert info["dropped_head"] and not info["dropped_tail"]
    assert abs(info["kept_s"] - 3.0) < 0.15


def test_trim_pad_never_reaches_into_dropped_material():
    # Loud fragment only 80 ms behind the kept speech: the tail pad must stop
    # inside the gap, not pull the fragment's onset back into the reference.
    out, info = _trim([_sil(1.0), _tone(3.0), _sil(0.08), _tone(0.4, amp=0.9)])
    assert out is not None and info["dropped_tail"]
    arr = np.frombuffer(out, dtype=np.float32)
    # guard = 40% of the 80 ms gap (~32 ms) + 20 ms fade → the very end is calm
    assert np.abs(arr[-int(_SR * 0.03):]).max() < 0.05, "no silence guard at the cut"
    # and the loud fragment (amp 0.9) itself must be gone entirely
    assert np.abs(arr[-int(_SR * 0.25):]).max() < 0.5, "cut-off fragment leaked into the tail pad"


def test_trim_output_has_edge_fades():
    out, _ = _trim([_sil(0.5), _tone(3.0), _sil(0.5)])
    arr = np.frombuffer(out, dtype=np.float32)
    assert abs(arr[0]) < 1e-4 and abs(arr[-1]) < 1e-4
