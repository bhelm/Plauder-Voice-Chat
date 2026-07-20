#!/usr/bin/env python3
"""Derive Lachen.vrma from Clapping.vrma (laughing body motion, no arms).

The 'laugh' emote uses the Clapping clip — happy body bounce PLUS the clapping
arm/hand motion. For a laugh without the applause we keep the same clip but
drop every arm channel (shoulder/upper arm/lower arm/hand/fingers); those bones
then stay on the procedural layer of waifu.js (REST pose, arms down, idle sway)
because the clip layer only touches nodes it has tracks for.

  1. read unused/Clapping.vrma (GLB)
  2. keep only rotation channels (drops hips root motion) whose target bone is
     NOT part of an arm
  3. rename the animation to 'Lachen', declare specVersion 1.0
  4. write Lachen.vrma (JSON chunk rewritten, binary chunk unchanged)
"""
import json
import struct

SRC = 'static/anims/unused/Clapping.vrma'
DST = 'static/anims/Lachen.vrma'

# substrings that mark a humanoid bone as "arm" (fingers included)
ARM_PARTS = ('Shoulder', 'UpperArm', 'LowerArm', 'Hand',
             'Thumb', 'Index', 'Middle', 'Ring', 'Little')

data = open(SRC, 'rb').read()
clen, = struct.unpack('<I', data[12:16])
js = json.loads(data[20:20 + clen])
off = 20 + clen
blen, = struct.unpack('<I', data[off:off + 4])
bin_chunk = data[off + 8:off + 8 + blen]

human = js['extensions']['VRMC_vrm_animation']['humanoid']['humanBones']
node2bone = {v['node']: k for k, v in human.items()}


def is_arm(bone):
    b = bone[0].upper() + bone[1:]
    return any(p in b for p in ARM_PARTS)


anim = js['animations'][0]
keep, drop_arm, drop_other = [], 0, 0
for ch in anim['channels']:
    tgt = ch['target']
    if tgt.get('path') != 'rotation':
        drop_other += 1
        continue
    bone = node2bone.get(tgt.get('node'), '')
    if is_arm(bone):
        drop_arm += 1
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
js['animations'] = [{'name': 'Lachen', 'channels': keep, 'samplers': new_samplers}]
# source pack omits specVersion -> three-vrm-animation warns; declare 1.0
js['extensions']['VRMC_vrm_animation']['specVersion'] = '1.0'

jbytes = json.dumps(js, separators=(',', ':')).encode()
jbytes += b' ' * ((4 - len(jbytes) % 4) % 4)
bbytes = bin_chunk + b'\0' * ((4 - len(bin_chunk) % 4) % 4)
glb = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(jbytes) + 8 + len(bbytes))
glb += struct.pack('<II', len(jbytes), 0x4E4F534A) + jbytes
glb += struct.pack('<II', len(bbytes), 0x004E4942) + bbytes
open(DST, 'wb').write(glb)
print('wrote %s: kept %d channels, dropped %d arm + %d non-rotation'
      % (DST, len(keep), drop_arm, drop_other))
