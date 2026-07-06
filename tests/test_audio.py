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
