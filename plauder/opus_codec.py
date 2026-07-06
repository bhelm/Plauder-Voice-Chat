"""Optional Opus codec support (bandwidth-reduced browser link).

Wraps ``opuslib`` (a tiny ctypes binding to the system libopus) for
- decoding the browser's opus mic uplink packets to 16 kHz float32 PCM, and
- encoding the TTS PCM16 downlink into opus packets (VCT3 frames).

The dependency is imported ONLY lazily inside :func:`is_available` and the
codec constructors (same invariant as the GPU backends: no heavy/optional deps
at module import). When ``opuslib`` or the system ``libopus`` is missing,
``is_available()`` returns False and the server silently keeps the raw PCM
paths — ``hello`` simply does not advertise opus.

libopus decodes/encodes natively at 8/12/16/24/48 kHz, so both the 16 kHz mic
uplink and the 24 kHz TTS downlink need no resampling.
"""

from __future__ import annotations

import logging

LOG = logging.getLogger("voice-chat")

# Default bitrates (bits/s). Uplink is encoded client-side (WebCodecs); the
# client uses its own constant — this one only feeds the loopback tests.
UPLINK_BITRATE = 24_000
DOWNLINK_BITRATE = 48_000

# Largest legal opus frame is 120 ms — decode buffers are sized for it.
_MAX_FRAME_MS = 120

_AVAILABLE: bool | None = None


def is_available() -> bool:
    """True when opuslib + the system libopus are importable and usable.
    Probed once (constructing a decoder exercises the ctypes binding), then
    cached for the process lifetime."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import opuslib

            opuslib.Decoder(16000, 1)
            _AVAILABLE = True
        except Exception as exc:  # ImportError, missing libopus, ABI issues …
            LOG.info("opus codec unavailable (%s) — raw PCM paths only", exc)
            _AVAILABLE = False
    return _AVAILABLE


class OpusDecoder:
    """Decodes a stream of raw opus packets to float32 LE PCM.

    One instance per streamed segment (libopus decoder state is stateful
    across packets). ``decode_packet`` raises on a corrupt packet — the caller
    logs and drops it, the decoder stays usable for the following packets."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        import opuslib

        self._dec = opuslib.Decoder(sample_rate, channels)
        self._max_frame = sample_rate * _MAX_FRAME_MS // 1000

    def decode_packet(self, packet: bytes) -> bytes:
        """One opus packet → float32 LE mono PCM bytes at the decoder rate."""
        return self._dec.decode_float(packet, self._max_frame)


class OpusEncoder:
    """Encodes 16-bit LE PCM into raw opus packets (20 ms frames).

    PCM may arrive in arbitrary chunk sizes; a partial trailing frame is
    buffered until enough samples exist. ``flush()`` zero-pads and emits the
    remainder (adds < 20 ms of silence at the very end of a stream)."""

    FRAME_MS = 20

    def __init__(self, sample_rate: int, bitrate: int = DOWNLINK_BITRATE,
                 channels: int = 1, application: str = "audio"):
        import opuslib

        app = (opuslib.APPLICATION_VOIP if application == "voip"
               else opuslib.APPLICATION_AUDIO)
        self._enc = opuslib.Encoder(sample_rate, channels, app)
        try:
            self._enc.bitrate = int(bitrate)
        except Exception:
            LOG.debug("opus encoder: could not set bitrate, using default")
        self.sample_rate = sample_rate
        self._frame_samples = sample_rate * self.FRAME_MS // 1000
        self._frame_bytes = self._frame_samples * 2 * channels
        self._buf = bytearray()

    def encode_pcm16(self, pcm: bytes) -> list[bytes]:
        """Append PCM16 bytes, return every completed opus packet."""
        self._buf.extend(pcm)
        out: list[bytes] = []
        while len(self._buf) >= self._frame_bytes:
            frame = bytes(self._buf[:self._frame_bytes])
            del self._buf[:self._frame_bytes]
            out.append(self._enc.encode(frame, self._frame_samples))
        return out

    def flush(self) -> list[bytes]:
        """Emit the buffered remainder as one final zero-padded packet."""
        if not self._buf:
            return []
        frame = bytes(self._buf) + b"\x00" * (self._frame_bytes - len(self._buf))
        self._buf.clear()
        return [self._enc.encode(frame, self._frame_samples)]
