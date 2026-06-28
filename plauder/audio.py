"""Audio utilities: PCM/WAV/numpy conversion + TTS text chunking.

Pure, stateless helpers. Browser ↔ server exchange 16 kHz mono; TTS delivers
24 kHz depending on the backend. These functions encapsulate the format
conversion.
"""

from __future__ import annotations

import io
import re
import wave

import numpy as np

# Magic header for WAV frames with an embedded turn ID.
# Format: 4 bytes "VCT1" + 1 byte ID length + N bytes ID (ASCII) + WAV bytes.
WAV_FRAME_MAGIC = b"VCT1"

# Magic header for streamed PCM chunks (progressive playback).
# Format: 4 bytes "VCT2" + 1 byte ID length + N bytes turn ID (ASCII)
#         + 2 bytes sequence (big-endian uint16) + raw 16-bit LE mono PCM.
# The sample rate comes separately in the audio.start event; the client builds
# AudioBuffers from it and stitches them together gaplessly.
PCM_CHUNK_MAGIC = b"VCT2"


def pcm_bytes_to_float32_array(pcm_bytes: bytes) -> np.ndarray:
    """Raw float32 PCM bytes (as from the browser) → numpy array."""
    return np.frombuffer(pcm_bytes, dtype=np.float32)


def float32_to_pcm16_bytes(samples):
    """Clip float32 samples to [-1, 1] and pack them as little-endian int16 PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def float32_to_pcm16_wav_bytes(samples_f32, sample_rate: int) -> bytes:
    """float32 [-1,1] → 16-bit mono WAV bytes."""
    pcm16 = float32_to_pcm16_bytes(samples_f32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """16-bit signed LE PCM (e.g. OpenAI TTS) → float32 [-1,1]."""
    pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def time_stretch(samples_f32, rate: float, sample_rate: int) -> np.ndarray:
    """Changes the speaking tempo by the factor `rate` (rate>1 = faster,
    rate<1 = slower), WITHOUT shifting the pitch (no chipmunk effect).

    Method: WSOLA (Waveform Similarity Overlap-Add). Analysis frames are
    aligned phase-continuously in the input via cross-correlation and added
    overlapping with a fixed synthesis hop. Good enough for TTS speed-up up to
    ~3×. Needed because some OpenAI-compatible TTS servers (e.g. local XTTS)
    ignore the `speed` parameter.
    """
    x = np.asarray(samples_f32, dtype=np.float32)
    if rate <= 0 or abs(rate - 1.0) < 1e-3 or x.size < 4:
        return x
    N = max(256, int(sample_rate * 0.046))   # window length (~46 ms)
    Hs = N // 2                              # synthesis hop (output)
    Ha = Hs * rate                          # analysis hop (input, factor rate)
    seek = max(1, N // 16)                   # WSOLA search range (±)
    win = np.hanning(N).astype(np.float32)
    max_off = x.size - N
    if max_off <= 0:
        return x
    out = np.zeros(int(x.size / rate) + 2 * N, dtype=np.float32)
    wsum = np.zeros_like(out)
    natural = None      # the natural continuation of the previous frame
    a = 0.0
    out_pos = 0
    while True:
        ana = int(round(a))
        if ana > max_off:
            break
        if natural is None:
            off = min(ana, max_off)
        else:
            lo = max(0, ana - seek)
            hi = min(max_off, ana + seek)
            if hi <= lo:
                off = min(ana, max_off)
            else:
                region = x[lo:hi + N]
                corr = np.correlate(region, natural, "valid")
                off = lo + int(np.argmax(corr))
        end = out_pos + N
        if end > out.size:
            break
        out[out_pos:end] += x[off:off + N] * win
        wsum[out_pos:end] += win
        nf = off + Hs                       # what naturally follows this frame
        natural = x[nf:nf + N]
        if natural.size < N:
            break
        out_pos += Hs
        a += Ha
    nz = wsum > 1e-6
    out[nz] /= wsum[nz]
    return out[:out_pos + N]


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Raw 16-bit mono PCM bytes → WAV container bytes (without re-quantization)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def wrap_wav_with_turn_id(wav_bytes: bytes, turn_id: str) -> bytes:
    """Prepends a frame header with the turn ID to the WAV payload so the
    client can reliably assign the audio to the correct turn.
    """
    tid = turn_id.encode("ascii", errors="replace")[:255]
    return WAV_FRAME_MAGIC + bytes([len(tid)]) + tid + wav_bytes


def wrap_pcm_chunk(turn_id: str, seq: int, pcm_bytes: bytes) -> bytes:
    """Packs a 16-bit PCM chunk into the VCT2 streaming frame (see
    PCM_CHUNK_MAGIC). ``seq`` is the consecutive chunk number (from 1).
    """
    tid = turn_id.encode("ascii", errors="replace")[:255]
    seq16 = max(0, min(0xFFFF, int(seq)))
    header = PCM_CHUNK_MAGIC + bytes([len(tid)]) + tid + bytes([(seq16 >> 8) & 0xFF, seq16 & 0xFF])
    return header + pcm_bytes


def iter_pcm_frames(pcm_bytes: bytes, frame_bytes: int):
    """Splits a PCM buffer into frames of a fixed byte size. Ensures that every
    frame has an even byte count (16-bit sample alignment).
    """
    if frame_bytes <= 0:
        frame_bytes = len(pcm_bytes)
    if frame_bytes % 2:
        frame_bytes -= 1
    if frame_bytes <= 0:
        return
    for off in range(0, len(pcm_bytes), frame_bytes):
        chunk = pcm_bytes[off:off + frame_bytes]
        if chunk:
            yield chunk


# --------------------------------------------------------------------------- #
# TTS text chunking
# --------------------------------------------------------------------------- #
# Sentence boundaries: after . ! ? … (also repeated) + whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_text_for_tts(text: str, max_chars: int) -> list[str]:
    """Splits text into TTS-suitable pieces (≤ max_chars).

    1. Split at punctuation (. ! ? …).
    2. Split overly long sentences further at comma/semicolon/colon.
    3. If a piece is still too long, hard-cut at word boundaries.
    4. Merge consecutive short pieces back together (≤ max_chars).
    """
    text = (text or "").strip()
    if not text:
        return []
    raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
    if not raw:
        raw = [text]

    def _hard_split(s: str) -> list[str]:
        if len(s) <= max_chars:
            return [s]
        out: list[str] = []
        parts = re.split(r"(?<=[,;:])\s+", s)
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) > max_chars:
                if buf:
                    out.append(buf)
                    buf = ""
                words = p.split()
                cur = ""
                for w in words:
                    if cur and len(cur) + 1 + len(w) > max_chars:
                        out.append(cur)
                        cur = w
                    else:
                        cur = (cur + " " + w).strip()
                if cur:
                    out.append(cur)
            elif buf and len(buf) + 1 + len(p) > max_chars:
                out.append(buf)
                buf = p
            else:
                buf = (buf + " " + p).strip()
        if buf:
            out.append(buf)
        return out

    pieces: list[str] = []
    for s in raw:
        pieces.extend(_hard_split(s))

    merged: list[str] = []
    for p in pieces:
        if merged and len(merged[-1]) + 1 + len(p) <= max_chars:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


# Complete sentence boundary in the (still growing) stream: up to the first
# punctuation mark, optionally followed by a closing quote/bracket, then
# whitespace. The trailing whitespace is mandatory so that a sentence does not
# flush prematurely while more characters may still follow (e.g. "z.B.").
_STREAM_BOUNDARY_RE = re.compile(r"(.*?[.!?…]['\"\)\]»]?)\s", re.S)


def split_stream_sentences(buffer: str, max_chars: int) -> tuple[list[str], str]:
    """Cuts all already-complete sentences out of a continuously growing LLM
    stream buffer.

    Returns ``(finished_sentences, rest)``. ``rest`` stays in the caller's
    buffer until more tokens arrive. A sentence without punctuation is
    force-flushed as soon as the buffer exceeds ``max_chars`` (cut at a word
    boundary), so that very long sentences do not drive up the first-audio
    latency.
    """
    sentences: list[str] = []
    buf = buffer
    while buf:
        m = _STREAM_BOUNDARY_RE.match(buf)
        if m:
            sent = m.group(1).strip()
            if sent:
                sentences.append(sent)
            buf = buf[m.end():]
            continue
        if len(buf) > max_chars > 0:
            cut = buf.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            head = buf[:cut].strip()
            if head:
                sentences.append(head)
            buf = buf[cut:].lstrip()
            continue
        break
    return sentences, buf
