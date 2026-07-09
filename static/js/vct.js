/* Binary WS frame parsers for the three audio framings the server ships:
   VCT1 full WAV (classic path), VCT2 streamed PCM chunk, VCT3 streamed
   opus chunk (see plauder/audio.py for the writers). Pure ArrayBuffer→
   object parsers — unit-tested via tests/client/pure_modules.test.mjs. */
// Parses the server's WAV frame format:
//   Magic "VCT1" + 1 byte ID length + N bytes turn ID (ASCII) + WAV bytes
// Returns: { turnId, wav } or null if no valid header.
const WAV_FRAME_MAGIC = [0x56, 0x43, 0x54, 0x31]; // "VCT1"
function parseFramedWav(arrayBuffer) {
  if (arrayBuffer.byteLength < 6) return null;
  const view = new Uint8Array(arrayBuffer);
  for (let i = 0; i < 4; i++) if (view[i] !== WAV_FRAME_MAGIC[i]) return null;
  const idLen = view[4];
  if (idLen === 0 || arrayBuffer.byteLength < 5 + idLen) return null;
  const idBytes = view.subarray(5, 5 + idLen);
  const turnId = new TextDecoder('ascii').decode(idBytes);
  const wav = arrayBuffer.slice(5 + idLen);
  return { turnId, wav };
}

// Parses the VCT2 streaming frame (progressive PCM chunks):
//   "VCT2" + 1 byte ID length + turn ID + 2 byte seq (BE) + PCM16LE
const PCM_CHUNK_MAGIC = [0x56, 0x43, 0x54, 0x32]; // "VCT2"
function parsePcmChunk(arrayBuffer) {
  if (arrayBuffer.byteLength < 7) return null;
  const view = new Uint8Array(arrayBuffer);
  for (let i = 0; i < 4; i++) if (view[i] !== PCM_CHUNK_MAGIC[i]) return null;
  const idLen = view[4];
  if (idLen === 0 || arrayBuffer.byteLength < 7 + idLen) return null;
  const turnId = new TextDecoder('ascii').decode(view.subarray(5, 5 + idLen));
  const seq = (view[5 + idLen] << 8) | view[6 + idLen];
  // 16-bit PCM from offset (7+idLen). Int16Array needs 2-byte alignment,
  // hence via DataView/copy, not directly on the ArrayBuffer.
  const pcmBytes = new Uint8Array(arrayBuffer, 7 + idLen);
  const n = pcmBytes.length >> 1;
  const int16 = new Int16Array(n);
  for (let i = 0; i < n; i++) int16[i] = (pcmBytes[2 * i] | (pcmBytes[2 * i + 1] << 8)) << 16 >> 16;
  return { turnId, seq, int16 };
}

// Parses the VCT3 streaming frame (opus-compressed downlink):
//   "VCT3" + 1 byte ID length + turn ID + 2 byte seq (BE)
//   + repeated (2 byte packet length BE + opus packet).
const OPUS_CHUNK_MAGIC = [0x56, 0x43, 0x54, 0x33]; // "VCT3"
function parseOpusChunk(arrayBuffer) {
  if (arrayBuffer.byteLength < 7) return null;
  const view = new Uint8Array(arrayBuffer);
  for (let i = 0; i < 4; i++) if (view[i] !== OPUS_CHUNK_MAGIC[i]) return null;
  const idLen = view[4];
  if (idLen === 0 || arrayBuffer.byteLength < 7 + idLen) return null;
  const turnId = new TextDecoder('ascii').decode(view.subarray(5, 5 + idLen));
  const seq = (view[5 + idLen] << 8) | view[6 + idLen];
  const packets = [];
  let off = 7 + idLen;
  while (off + 2 <= view.length) {
    const len = (view[off] << 8) | view[off + 1];
    off += 2;
    if (off + len > view.length) break;   // truncated record → ignore the rest
    packets.push(view.slice(off, off + len));
    off += len;
  }
  return { turnId, seq, packets };
}
