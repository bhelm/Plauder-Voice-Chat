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
