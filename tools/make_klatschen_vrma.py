#!/usr/bin/env python3
"""Derive Klatschen.vrma from Clapping.vrma (arms + body only, no head).

Counterpart to make_lachen_vrma.py: where Lachen keeps the laughing body and
drops the applause, Klatschen keeps the applause (arms, hands, fingers) plus
the body motion carrying it, and drops the head/neck channels so the clip
leaves the face free for the procedural layer of waifu.js (mode overlays,
lookAt, lip-sync). The source has no expression/lookAt tracks at all, so
there is no facial animation to strip beyond that.

  1. read Clapping.vrma (GLB)
  2. keep only rotation channels (drops hips root motion) whose target bone is
     NOT head or neck
  3. rename the animation to 'Klatschen', declare specVersion 1.0
  4. write Klatschen.vrma (JSON chunk rewritten, binary chunk unchanged)
"""
import json
import struct

SRC = 'static/anims/Clapping.vrma'
DST = 'static/anims/Klatschen.vrma'

# head motion: 'neck' is dropped too — it swings the head just as much as
# 'head' does, so keeping it would defeat the purpose
HEAD_BONES = ('head', 'neck')

data = open(SRC, 'rb').read()
clen, = struct.unpack('<I', data[12:16])
js = json.loads(data[20:20 + clen])
off = 20 + clen
blen, = struct.unpack('<I', data[off:off + 4])
bin_chunk = data[off + 8:off + 8 + blen]

human = js['extensions']['VRMC_vrm_animation']['humanoid']['humanBones']
node2bone = {v['node']: k for k, v in human.items()}

anim = js['animations'][0]
keep, drop_head, drop_other = [], 0, 0
for ch in anim['channels']:
    tgt = ch['target']
    if tgt.get('path') != 'rotation':
        drop_other += 1
        continue
    if node2bone.get(tgt.get('node'), '') in HEAD_BONES:
        drop_head += 1
        continue
    keep.append(ch)

# compact the samplers to the kept channels (accessors/bufferViews keep their
# indices; the now-unreferenced ones are simply never read by glTF loaders)
new_samplers, remap = [], {}
for ch in keep:
    si = ch['sampler']
    if si not in remap:
        remap[si] = len(new_samplers)
        new_samplers.append(anim['samplers'][si])
    ch['sampler'] = remap[si]
js['animations'] = [{'name': 'Klatschen', 'channels': keep, 'samplers': new_samplers}]
# source pack omits specVersion -> three-vrm-animation warns; declare 1.0
js['extensions']['VRMC_vrm_animation']['specVersion'] = '1.0'

jbytes = json.dumps(js, separators=(',', ':')).encode()
jbytes += b' ' * ((4 - len(jbytes) % 4) % 4)
bbytes = bin_chunk + b'\0' * ((4 - len(bin_chunk) % 4) % 4)
glb = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(jbytes) + 8 + len(bbytes))
glb += struct.pack('<II', len(jbytes), 0x4E4F534A) + jbytes
glb += struct.pack('<II', len(bbytes), 0x004E4942) + bbytes
open(DST, 'wb').write(glb)
print('wrote %s: kept %d channels, dropped %d head/neck + %d non-rotation'
      % (DST, len(keep), drop_head, drop_other))
