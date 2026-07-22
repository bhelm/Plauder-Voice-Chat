#!/usr/bin/env python3
"""Derive HeadScratch.vrma from Blush.vrma (body animation only).

The Blush clip from tk256ailab/vrm-viewer carries the classic embarrassed
head-scratch gesture as its BODY animation — the smiling face of the 'blush'
emote is procedural in waifu.js, not part of the file (Blush.vrma has no
expression/lookAt tracks at all). For the thinking stage we reuse exactly
that gesture; the 'scratch' emote has no facial overlay, so the face stays
neutral.

  1. read Blush.vrma (GLB)
  2. keep only rotation channels (drops hips root motion; expression/lookAt
     channels are filtered too, defensively — the source has none)
  3. rename the animation to 'HeadScratch', declare specVersion 1.0
  4. write HeadScratch.vrma (JSON chunk rewritten, binary chunk unchanged)
"""
import json
import struct

SRC = 'static/anims/Blush.vrma'
DST = 'static/anims/HeadScratch.vrma'

data = open(SRC, 'rb').read()
clen, = struct.unpack('<I', data[12:16])
js = json.loads(data[20:20 + clen])
off = 20 + clen
blen, = struct.unpack('<I', data[off:off + 4])
bin_chunk = data[off + 8:off + 8 + blen]

anim = js['animations'][0]
keep = [ch for ch in anim['channels'] if ch['target'].get('path') == 'rotation']
dropped = len(anim['channels']) - len(keep)
# compact the samplers to the kept channels (accessors/bufferViews keep their
# indices; a few become unreferenced, which glTF loaders simply never read)
new_samplers, remap = [], {}
for ch in keep:
    si = ch['sampler']
    if si not in remap:
        remap[si] = len(new_samplers)
        new_samplers.append(anim['samplers'][si])
    ch['sampler'] = remap[si]
js['animations'] = [{'name': 'HeadScratch', 'channels': keep, 'samplers': new_samplers}]
# source pack omits specVersion -> three-vrm-animation warns; declare 1.0
js['extensions']['VRMC_vrm_animation']['specVersion'] = '1.0'

jbytes = json.dumps(js, separators=(',', ':')).encode()
jbytes += b' ' * ((4 - len(jbytes) % 4) % 4)
bbytes = bin_chunk + b'\0' * ((4 - len(bin_chunk) % 4) % 4)
glb = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(jbytes) + 8 + len(bbytes))
glb += struct.pack('<II', len(jbytes), 0x4E4F534A) + jbytes
glb += struct.pack('<II', len(bbytes), 0x004E4942) + bbytes
open(DST, 'wb').write(glb)
print('wrote %s: kept %d rotation channels, dropped %d other(s)'
      % (DST, len(keep), dropped))
