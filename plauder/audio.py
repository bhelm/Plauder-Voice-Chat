"""Audio-Utilities: PCM/WAV/numpy-Konvertierung + TTS-Text-Chunking.

Reine, zustandslose Helfer. Browser ↔ Server tauschen 16 kHz Mono aus; TTS
liefert je nach Backend 24 kHz. Diese Funktionen kapseln die Formatwandlung.
"""

from __future__ import annotations

import io
import re
import wave

import numpy as np

# Magic-Header für WAV-Frames mit eingebetteter Turn-ID.
# Format: 4 Bytes "VCT1" + 1 Byte ID-Länge + N Bytes ID (ASCII) + WAV-Bytes.
WAV_FRAME_MAGIC = b"VCT1"

# Magic-Header für gestreamte PCM-Chunks (progressive Wiedergabe).
# Format: 4 Bytes "VCT2" + 1 Byte ID-Länge + N Bytes Turn-ID (ASCII)
#         + 2 Bytes Sequenz (big-endian uint16) + roh-16-bit-LE-Mono-PCM.
# Die Sample-Rate kommt separat im audio.start-Event; der Client baut daraus
# AudioBuffers und reiht sie lückenlos aneinander.
PCM_CHUNK_MAGIC = b"VCT2"


def pcm_bytes_to_float32_array(pcm_bytes: bytes) -> np.ndarray:
    """Roh-float32-PCM-Bytes (wie vom Browser) → numpy-Array."""
    return np.frombuffer(pcm_bytes, dtype=np.float32)


def float32_to_pcm16_wav_bytes(samples_f32, sample_rate: int) -> bytes:
    """float32 [-1,1] → 16-bit Mono-WAV-Bytes."""
    clipped = np.clip(samples_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """16-bit signed LE PCM (z.B. OpenAI-TTS) → float32 [-1,1]."""
    pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def time_stretch(samples_f32, rate: float, sample_rate: int) -> np.ndarray:
    """Ändert das Sprech-Tempo um den Faktor `rate` (rate>1 = schneller,
    rate<1 = langsamer), OHNE die Tonhöhe zu verschieben (kein Mickey-Maus).

    Verfahren: WSOLA (Waveform Similarity Overlap-Add). Analyse-Frames werden
    im Eingang per Kreuzkorrelation phasen-stetig ausgerichtet und mit festem
    Synthese-Hop überlappend addiert. Genug für TTS-Beschleunigung bis ~3×.
    Wird gebraucht, weil manche OpenAI-kompatiblen TTS-Server (z.B. lokales
    XTTS) den `speed`-Parameter ignorieren.
    """
    x = np.asarray(samples_f32, dtype=np.float32)
    if abs(rate - 1.0) < 1e-3 or x.size < 4:
        return x
    N = max(256, int(sample_rate * 0.046))   # Fensterlänge (~46 ms)
    Hs = N // 2                              # Synthese-Hop (Ausgabe)
    Ha = Hs * rate                          # Analyse-Hop (Eingabe, Faktor rate)
    seek = max(1, N // 16)                   # WSOLA-Suchbereich (±)
    win = np.hanning(N).astype(np.float32)
    max_off = x.size - N
    if max_off <= 0:
        return x
    out = np.zeros(int(x.size / rate) + 2 * N, dtype=np.float32)
    wsum = np.zeros_like(out)
    natural = None      # die natürliche Fortsetzung des vorigen Frames
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
        nf = off + Hs                       # was nach diesem Frame natürlich folgt
        natural = x[nf:nf + N]
        if natural.size < N:
            break
        out_pos += Hs
        a += Ha
    nz = wsum > 1e-6
    out[nz] /= wsum[nz]
    return out[:out_pos + N]


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Roh-16-bit-Mono-PCM-Bytes → WAV-Container-Bytes (ohne Re-Quantisierung)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def wrap_wav_with_turn_id(wav_bytes: bytes, turn_id: str) -> bytes:
    """Stellt der WAV-Payload einen Frame-Header mit der Turn-ID voran, damit
    der Client das Audio sicher dem richtigen Turn zuordnen kann.
    """
    tid = turn_id.encode("ascii", errors="replace")[:255]
    return WAV_FRAME_MAGIC + bytes([len(tid)]) + tid + wav_bytes


def wrap_pcm_chunk(turn_id: str, seq: int, pcm_bytes: bytes) -> bytes:
    """Verpackt einen 16-bit-PCM-Chunk in das VCT2-Streaming-Frame (siehe
    PCM_CHUNK_MAGIC). ``seq`` ist die fortlaufende Chunk-Nummer (ab 1).
    """
    tid = turn_id.encode("ascii", errors="replace")[:255]
    seq16 = max(0, min(0xFFFF, int(seq)))
    header = PCM_CHUNK_MAGIC + bytes([len(tid)]) + tid + bytes([(seq16 >> 8) & 0xFF, seq16 & 0xFF])
    return header + pcm_bytes


def iter_pcm_frames(pcm_bytes: bytes, frame_bytes: int):
    """Zerlegt einen PCM-Buffer in Frames fester Byte-Größe. Stellt sicher, dass
    jeder Frame eine gerade Byte-Anzahl hat (16-bit-Sample-Ausrichtung).
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
# TTS-Text-Chunking
# --------------------------------------------------------------------------- #
# Satzgrenzen: nach . ! ? … (auch mehrfach) + Whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_text_for_tts(text: str, max_chars: int) -> list[str]:
    """Zerlegt Text in TTS-taugliche Stücke (≤ max_chars).

    1. An Satzzeichen splitten (. ! ? …).
    2. Zu lange Sätze an Komma/Semikolon/Doppelpunkt weiter zerlegen.
    3. Bleibt ein Stück zu lang, hart an Wortgrenzen schneiden.
    4. Aufeinanderfolgende kurze Stücke wieder zusammenfassen (≤ max_chars).
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


# Vollständige Satzgrenze im (noch wachsenden) Stream: bis zum ersten
# Satzzeichen, optional gefolgt von schließendem Anführungszeichen/Klammer,
# dann Whitespace. Der nachgestellte Whitespace ist Pflicht, damit ein Satz
# nicht vorzeitig flusht, solange evtl. noch Zeichen nachkommen (z.B. "z.B.").
_STREAM_BOUNDARY_RE = re.compile(r"(.*?[.!?…]['\"\)\]»]?)\s", re.S)


def split_stream_sentences(buffer: str, max_chars: int) -> tuple[list[str], str]:
    """Schneidet aus einem fortlaufend wachsenden LLM-Stream-Puffer alle bereits
    vollständigen Sätze heraus.

    Gibt ``(fertige_sätze, rest)`` zurück. ``rest`` bleibt im Aufrufer-Puffer,
    bis weitere Token kommen. Ein Satz ohne Satzzeichen wird zwangs-geflusht,
    sobald der Puffer ``max_chars`` überschreitet (Schnitt an Wortgrenze), damit
    sehr lange Sätze die Erst-Latenz nicht in die Höhe treiben.
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
