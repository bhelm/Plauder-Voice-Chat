#!/usr/bin/env python3
"""Author Wave.vrma from scratch: classic right-hand wave (greeting/farewell).

Unlike the other make_*_vrma tools this one does not filter an existing clip —
no source clip in the pack contains a one-armed wave (Goodbye waves BOTH arms).
Instead it synthesizes the keyframes: the upper arm rises only slightly (the
elbow stays near shoulder height), the forearm goes upright so the hand ends
up NEXT TO THE HEAD, the hand twists about the forearm axis so the palm faces
the camera, and the hand waves in the frontal plane before the arm comes back
down. Fingers and thumb are left untouched (they keep the model's rest pose).

The phases are strictly separated: the swing ends BEFORE the arm starts its
descent, so the forearm holds its ~90-deg raise for the whole wave. This only
works together with the `Wave` entry in CLIP_END (waifu.js): the generic
once-clip fade would start CLIP_FADE_OUT (0.9 s) before the clip ends and
blend the still-waving arm toward the arms-down procedural pose — the exact
"forearm sags while waving" artifact this layout avoids. The CLIP_END entry
keeps full weight through the wave and fades only over the authored descent.

An existing VRMA (Goodbye) serves only as the STRUCTURAL template: its node
hierarchy and VRMC_vrm_animation humanoid mapping are copied verbatim, the
animation/buffer data is built fresh (30 fps LINEAR keys). The script verifies
its own output numerically — including the hand's WORLD position vs. the head
(computed from the template's skeleton offsets) — because headless-browser
testing is not available in this environment.

Tunables live in the constants below; regenerate with:
  .venv/bin/python tools/make_wave_vrma.py
"""
import json
import math
import struct

TEMPLATE = 'static/anims/unused/Goodbye.vrma'
DST = 'static/anims/Wave.vrma'

DUR = 3.6       # s, total clip length
FPS = 30        # keyframe rate (LINEAR interpolation between keys)

# Geometry (negative Z = raise the right arm, frontal plane):
UPPER_DEG = -10.0     # shoulder barely above horizontal -> elbow stays low
LOWER_DEG = -85.0     # forearm upright -> hand next to the head, not above it
TWIST_DEG = -90.0     # hand twist about the forearm axis: palm to the camera
WAVE_HAND_DEG = -25.0  # hand swing amplitude (frontal plane, pre-twist)
WAVE_HZ = 2.2
# NB: the swing lives ONLY on the hand. An earlier version added a +-8 deg
# elbow component — on the real model that read as the forearm sinking toward
# the body during the wave and popping back up afterwards, so the forearm now
# holds perfectly still (user request).

BONES = ['rightUpperArm', 'rightLowerArm', 'rightHand']


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


# ---- quaternions, glTF order [x, y, z, w] ---------------------------------
def _axis_q(deg, ax):
    a = math.radians(deg) / 2
    q = [0.0, 0.0, 0.0, math.cos(a)]
    q[ax] = math.sin(a)
    return tuple(q)


def qx(deg): return _axis_q(deg, 0)
def qy(deg): return _axis_q(deg, 1)
def qz(deg): return _axis_q(deg, 2)


def qmul(a, b):
    """Hamilton product: rotation b is applied FIRST, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def qrot(q, v):
    x, y, z, w = q
    cx, cy, cz = (y * v[2] - z * v[1] + w * v[0],
                  z * v[0] - x * v[2] + w * v[1],
                  x * v[1] - y * v[0] + w * v[2])
    return (v[0] + 2 * (y * cz - z * cy),
            v[1] + 2 * (z * cx - x * cz),
            v[2] + 2 * (x * cy - y * cx))


def qangle_id(q):
    return 2 * math.degrees(math.acos(min(1.0, abs(q[3]))))


def qangle(a, b):
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2 * math.degrees(math.acos(min(1.0, dot)))


# ---- the animation itself --------------------------------------------------
def envelopes(t):
    # Strict phase order: raise (0-0.7) -> wave (0.8-2.6) -> hold -> lower
    # (2.7-3.55). The swing envelope closes before the lift envelope opens
    # downwards, so the forearm never sinks while still waving.
    lift = smoothstep(0.0, 0.7, t) - smoothstep(2.7, 3.55, t)
    wave = smoothstep(0.8, 1.1, t) - smoothstep(2.3, 2.6, t)
    osc = math.sin(2 * math.pi * WAVE_HZ * (t - 0.8))
    return lift, wave, osc


def pose(t):
    lift, wave, osc = envelopes(t)
    return {
        'rightUpperArm': qz(UPPER_DEG * lift),
        'rightLowerArm': qz(LOWER_DEG * lift),
        # Twist about the bone's long axis first (palm to camera), then the
        # frontal-plane swing on top — order matters.
        'rightHand': qmul(qz(WAVE_HAND_DEG * wave * osc),
                          qx(TWIST_DEG * lift)),
    }


def main():
    data = open(TEMPLATE, 'rb').read()
    ln = struct.unpack('<I', data[12:16])[0]
    js = json.loads(data[20:20 + ln])

    hb = js['extensions']['VRMC_vrm_animation']['humanoid']['humanBones']
    node_of = {b: hb[b]['node'] for b in BONES}

    n = int(DUR * FPS) + 1
    times = [i / FPS for i in range(n)]
    poses = [pose(t) for t in times]

    # ---- fresh binary buffer: one shared time accessor + quat accessors ----
    blob = bytearray()
    views, accessors = [], []

    def add_accessor(values, ncomp, atype, with_minmax=False):
        off = len(blob)
        flat = [c for v in values for c in (v if ncomp > 1 else (v,))]
        blob.extend(struct.pack('<%df' % len(flat), *flat))
        views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(flat) * 4})
        acc = {'bufferView': len(views) - 1, 'componentType': 5126,
               'count': len(values), 'type': atype}
        if with_minmax:
            acc['min'] = [min(flat)]
            acc['max'] = [max(flat)]
        accessors.append(acc)
        return len(accessors) - 1

    t_acc = add_accessor(times, 1, 'SCALAR', with_minmax=True)
    samplers, channels = [], []
    for b in BONES:
        q_acc = add_accessor([p[b] for p in poses], 4, 'VEC4')
        samplers.append({'input': t_acc, 'output': q_acc,
                         'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1,
                         'target': {'node': node_of[b], 'path': 'rotation'}})

    js['buffers'] = [{'byteLength': len(blob)}]
    js['bufferViews'] = views
    js['accessors'] = accessors
    js['animations'] = [{'name': 'Wave', 'samplers': samplers,
                         'channels': channels}]
    js['asset']['generator'] = 'plauder tools/make_wave_vrma.py'

    # ---- GLB assembly (JSON chunk space-padded, BIN chunk zero-padded) -----
    jbytes = json.dumps(js, separators=(',', ':')).encode()
    jbytes += b' ' * (-len(jbytes) % 4)
    bbytes = bytes(blob) + b'\0' * (-len(blob) % 4)
    total = 12 + 8 + len(jbytes) + 8 + len(bbytes)
    out = struct.pack('<III', 0x46546C67, 2, total)
    out += struct.pack('<II', len(jbytes), 0x4E4F534A) + jbytes
    out += struct.pack('<II', len(bbytes), 0x004E4942) + bbytes
    open(DST, 'wb').write(out)
    print('wrote %s (%d bytes, %d keys/channel, %.2fs)'
          % (DST, len(out), n, DUR))

    # ---- numeric self-verification -----------------------------------------
    # Rest world positions from the template hierarchy (translations only —
    # the template's nodes carry no rest rotations).
    parent = {}
    for i, node in enumerate(js['nodes']):
        for c in node.get('children', []):
            parent[c] = i

    def world(idx):
        p = [0.0, 0.0, 0.0]
        while idx is not None:
            t = js['nodes'][idx].get('translation', [0, 0, 0])
            p = [a + b for a, b in zip(p, t)]
            idx = parent.get(idx)
        return p

    t_probe = 1.6
    pz = pose(t_probe)
    q_u = pz['rightUpperArm']
    q_ul = qmul(q_u, pz['rightLowerArm'])
    q_ulh = qmul(q_ul, pz['rightHand'])
    shoulder = world(node_of['rightUpperArm'])
    t_lower = js['nodes'][node_of['rightLowerArm']]['translation']
    t_hand = js['nodes'][node_of['rightHand']]['translation']
    elbow = [a + b for a, b in zip(shoulder, qrot(q_u, t_lower))]
    hand = [a + b for a, b in zip(elbow, qrot(q_ul, t_hand))]
    head = world(hb['head']['node'])

    # 1) the wave happens at head height, on the right side of the body
    dy = hand[1] - head[1]
    assert abs(dy) < 0.2, 'hand not at head height: dy=%.2f' % dy
    assert hand[0] < -0.1, 'hand not on the right side: x=%.2f' % hand[0]
    # 2) palm faces the camera (+Z), fingers point up
    palm = qrot(q_ulh, (0.0, -1.0, 0.0))
    fingers = qrot(q_ulh, (-1.0, 0.0, 0.0))
    assert palm[2] > 0.9, 'palm not facing camera: z=%.2f' % palm[2]
    assert fingers[1] > 0.7, 'fingers not pointing up: y=%.2f' % fingers[1]
    # 3) the hand actually waves: sign changes of the swing angle
    swing = [envelopes(t)[1] * envelopes(t)[2]
             for t in times if 0.9 < t < 2.3]
    flips = sum(1 for i in range(1, len(swing)) if swing[i - 1] * swing[i] < 0)
    assert flips >= 5, 'too few wave swings: %d' % flips
    # 4) upper arm AND forearm are perfectly STILL for the entire swing
    #    window — the arm must not move at all while waving (user request;
    #    an oscillating elbow read as the forearm sinking on the real model)
    hold = [p for t, p in zip(times, poses) if 0.8 <= t <= 2.6]
    for b in ('rightUpperArm', 'rightLowerArm'):
        drift = max(qangle(hold[0][b], p[b]) for p in hold)
        assert drift < 0.01, '%s moves during the wave: %.3f deg' % (b, drift)
    # 5) the clip ends exactly on the rest pose (procedural layer takes over)
    end = pose(DUR)
    worst = max(qangle_id(q) for q in end.values())
    assert worst < 0.5, 'end pose not rest: %.2f deg' % worst
    print('verified: hand at head height (dy=%+.2f m, x=%.2f), palm to camera '
          '(z=%.2f), fingers up (y=%.2f), %d swings, clean rest-pose end'
          % (dy, hand[0], palm[2], fingers[1], flips))


if __name__ == '__main__':
    main()
