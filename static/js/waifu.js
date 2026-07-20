/*
 * waifu.js — VTuber-Avatar-Renderer (VRM 1.0 / three-vrm)
 *
 * ISOLIERT: Dieses Modul fasst KEINE bestehenden VC-Pfade (mic/playback/vct) an.
 * Bricht der Avatar, laeuft der Voice-Chat normal weiter.
 *
 * Public API (window.Waifu):
 *   Waifu.mount(canvas)      -> initialisiert Szene + laedt VRM in den gegebenen Canvas
 *   Waifu.unmount()          -> stoppt Loop, gibt GPU-Ressourcen frei
 *   Waifu.setMouth(v)        -> Lip-Sync: v in [0..1] auf 'aa'-Expression
 *   Waifu.setState(s)        -> 'idle' | 'listening' | 'speaking' | 'thinking'
 *   Waifu.emote(n)           -> kurze Emote-Animation: 'lachen'|'sigh'|'surprise'|
 *                               'nod'|'grumble'|'smile'|'scratch'; liefert Dauer (s)
 *   Waifu.stopEmote()        -> laufendes Emote (inkl. VRMA-Clip) weich abbrechen
 *   Waifu.emoteActive()      -> bool: Emote oder Clip laeuft gerade noch
 *   Waifu.emoteNames()       -> Liste aller Emote-Namen (fuer den Anim-Tester)
 *   Waifu.setExpression(n,v) -> beliebige VRM-Expression setzen (happy/angry/...)
 *   Waifu.isReady()          -> bool
 *
 * Animations-Layer (jeder Frame, additiv uebereinander):
 *   1. Rest-Pose (Arme unten)          4. Modus-Overlay (thinking gestaffelt, ...)
 *   2. Atmen                           5. Emote-Overlay (laugh/sigh/... mit Envelope)
 *   3. Idle-Randomness (Kopf+Arme)     6. Expressions (Mund/Blink/Emotion) gedaempft
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from 'three-vrm';
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from 'three-vrm-animation';

const MODEL_URL = (window.__BASE_PATH__ || '') + '/static/models/joy.vrm';
const ANIM_URL = (window.__BASE_PATH__ || '') + '/static/anims/';

// Emote -> VRMA-Clip (aus tk256ailab/vrm-viewer, MIT). Nicht gemappte Emotes
// (nod) bleiben prozedural.
const EMOTE_CLIPS = {
  lachen: 'Lachen',       // Lach-Koerperbewegung ohne Klatschen, happy/Lach-Augen
                          // darueber (tools/make_lachen_vrma.py)
  sigh: 'Sad',
  surprise: 'Surprised',
  grumble: 'Angry',
  smile: 'Blush',
  scratch: 'HeadScratch', // Blush-Koerpergeste ohne Mimik (tools/make_head_scratch_vrma.py)
};

// Ruhepose: Arme entspannt nach unten (VRM-Grundpose ist T-Pose)
const REST = {
  leftUpperArm:  { x: 0, y: 0, z: -1.15 },
  rightUpperArm: { x: 0, y: 0, z:  1.15 },
  leftLowerArm:  { x: 0, y: 0, z: -0.15 },
  rightLowerArm: { x: 0, y: 0, z:  0.15 },
};

// Bones, die der Animations-Layer anfasst
const BONES = ['head', 'neck', 'spine', 'chest', 'hips',
  'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'rightHand'];

// Emote-Dauern in Sekunden
const EMOTE_DUR = { lachen: 2.4, sigh: 2.8, surprise: 1.6, nod: 1.4, grumble: 2.0, smile: 2.5, scratch: 5.3 };

const state = {
  renderer: null, scene: null, camera: null, controls: null,
  vrm: null, clock: null, raf: null, rafIsTimeout: false, ready: false,
  mixer: null,               // THREE.AnimationMixer fuer VRMA-Clips
  clip: null,                // aktiver Clip: { name, action, clip, bindings, w, target, once, t }
  clipPending: false,
  clipCache: {},             // name -> { clip, bindings } (pro VRM-Instanz)
  mouth: 0, mouthTarget: 0,
  blinkTimer: 0, nextBlink: 2 + Math.random() * 3,
  mode: 'idle', modeTime: 0,
  thinkScratchAt: 7,         // naechster Kopf-Kratz-Zeitpunkt (s in thinking)
  emote: null,               // { name, t, dur }
  t: 0,                      // globale Animationszeit
  // Idle-Randomness: Kopf schaut alle paar Sekunden woanders hin
  idleHead: { x: 0, y: 0, z: 0 }, idleHeadTimer: 0, idleHeadNext: 1.5,
  swayPhase: Math.random() * 6.28, swaySpeed: 0.45 + Math.random() * 0.15,
  // gedaempfte Ist-Pose (bone -> {x,y,z})
  pose: {},
  // gedaempfte Expression-Werte
  expr: { happy: 0, sad: 0, angry: 0, surprised: 0, relaxed: 0 },
  squint: 0,                 // Ziel fuer nachdenkliches Augen-Zukneifen
  squintCur: 0,
  // Blickziel (Augen): normal Kamera, beim langen Nachdenken "Sterne gucken"
  look: null, lookWanderTimer: 0, lookWanderNext: 1,
  lookWander: null,            // aktueller Fixpunkt ueber dem Kopf (Weltkoord.)
};

for (const b of BONES) state.pose[b] = { x: 0, y: 0, z: 0 };

const IDENT_Q = new THREE.Quaternion();
const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();

const damp = (c, t, l, dt) => THREE.MathUtils.damp(c, t, l, dt);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const smoothstep = (a, b, v) => {
  const x = clamp01((v - a) / (b - a));
  return x * x * (3 - 2 * x);
};

function initScene(canvas) {
  const w = canvas.clientWidth || 320;
  const h = canvas.clientHeight || 480;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(30, w / h, 0.1, 20);
  camera.position.set(0, 1.35, 1.6);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 1.3, 0);
  controls.enablePan = false;
  controls.minDistance = 0.8;
  controls.maxDistance = 4;
  controls.update();

  const key = new THREE.DirectionalLight(0xffffff, Math.PI);
  key.position.set(1, 2, 1.5);
  scene.add(key);
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));

  // Blickziel fuer die Augen (vrm.lookAt)
  const look = new THREE.Object3D();
  look.position.copy(camera.position);
  scene.add(look);
  state.look = look;

  state.renderer = renderer;
  state.scene = scene;
  state.camera = camera;
  state.controls = controls;
  state.clock = new THREE.Clock();
}

async function loadVRM() {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(MODEL_URL);
  const vrm = gltf.userData.vrm;

  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);

  vrm.springBoneManager?.reset();
  state.scene.add(vrm.scene);
  if (vrm.lookAt) vrm.lookAt.target = state.look;
  state.vrm = vrm;
  // Nodes, die der prozedurale Layer besitzt (fuer Clip-Blending)
  state.boneNodes = new Set();
  for (const b of BONES) {
    const n = vrm.humanoid?.getNormalizedBoneNode(b);
    if (n) state.boneNodes.add(n);
  }
  state.ready = true;
}

/* ---- VRMA-Clips (three-vrm-animation) ------------------------------------
   Clips laufen ueber einen AnimationMixer auf den normalisierten Bones.
   Pro Frame: erst prozedurale Pose anwenden, deren Quaternions als "pre"
   sichern, Mixer drueberschreiben lassen, dann pre->clip mit Gewicht w
   slerpen. So faden Clips weich ueber die Idle-Pose und zurueck. */
const CLIP_FADE_IN = 0.35;   // s: Einblenden ueber die prozedurale Pose
const CLIP_FADE_OUT = 0.9;   // s: Ausblenden am Clip-Ende bzw. nach stopEmote()

// Clip-Ende-Behandlung: Clips, deren einanimierte Rueckfuehrung zu ruckartig
// ist (HeadScratch: der Arm-Drop ab ~2.6s beschleunigt auf ~235 deg/s),
// laufen am Ende in Zeitlupe weiter — timeScale rampt ab slowAt auf slowTo,
// die authored Bewegung (Arme, Kopf, Koerper zusammen) bleibt kontinuierlich,
// nur langsamer; parallel blendet ein Smoothstep (fadeAt -> fadeEnd) auf die
// prozedurale Idle-Pose. Alle Zeiten in CLIP-Zeit (action.time); realDur ist
// die resultierende Echtzeit-Dauer (fuer die Emote-Buchhaltung, simuliert).
const CLIP_END = { HeadScratch: {
  slowAt: 2.45, slowEnd: 2.75, slowTo: 0.4, fadeAt: 2.9, fadeEnd: 3.7, realDur: 5.3,
} };

async function loadClip(name) {
  if (state.clipCache[name]) return state.clipCache[name];
  const loader = new GLTFLoader();
  loader.register((p) => new VRMAnimationLoaderPlugin(p));
  const gltf = await loader.loadAsync(ANIM_URL + name + '.vrma');
  const anim = gltf.userData.vrmAnimations && gltf.userData.vrmAnimations[0];
  if (!anim) throw new Error('no vrmAnimation in ' + name);
  const clip = createVRMAnimationClip(anim, state.vrm);
  // Nur Rotations-Tracks: hips.position (Root-Motion) wuerde das Modell
  // versetzen, Expression-Tracks kollidieren mit Lip-Sync/Emotionen.
  clip.tracks = clip.tracks.filter((tr) => tr.name.endsWith('.quaternion'));
  const nodes = [];
  for (const tr of clip.tracks) {
    const nodeName = THREE.PropertyBinding.parseTrackName(tr.name).nodeName;
    const node = THREE.PropertyBinding.findNode(state.vrm.scene, nodeName);
    if (node && !nodes.includes(node)) nodes.push(node);
  }
  const entry = { clip, nodes };
  state.clipCache[name] = entry;
  return entry;
}

async function playClip(name, opts) {
  if (!state.vrm || state.clipPending) return;
  if (state.clip && state.clip.name === name && !opts.once) return; // laeuft schon (Loop)
  state.clipPending = true;
  try {
    const entry = await loadClip(name);
    if (!state.vrm) return;
    if (!state.mixer) state.mixer = new THREE.AnimationMixer(state.vrm.scene);
    if (state.clip) state.clip.action.stop();
    const action = state.mixer.clipAction(entry.clip);
    action.reset();
    action.setLoop(opts.once ? THREE.LoopOnce : THREE.LoopRepeat, opts.once ? 1 : Infinity);
    action.clampWhenFinished = true;
    action.play();
    const end = CLIP_END[name] || null;
    const effDur = end ? end.realDur : entry.clip.duration;
    action.timeScale = 1;   // Actions werden gecacht — CLIP_END-Rest zuruecksetzen
    state.clip = {
      name, action, nodes: entry.nodes,
      pre: entry.nodes.map(() => new THREE.Quaternion()),
      w: 0, once: !!opts.once, dur: effDur, end, t: 0,
      fade: null, fadeFrom: 0,   // stopEmote(): zeitbasierter Abbruch-Fade
    };
    // Emote-Expressions so lange halten wie der Clip laeuft
    if (opts.emoteName && state.emote && state.emote.name === opts.emoteName) {
      state.emote.dur = Math.max(state.emote.dur, effDur);
    }
  } catch (e) {
    console.warn('[waifu] clip failed:', name, e);
  } finally {
    state.clipPending = false;
  }
}

function updateClip(dt) {
  const c = state.clip;
  if (!c || !state.mixer) return;
  c.t += dt;
  // Envelope: zeitbasierter Smoothstep statt exponentiellem damp. Der
  // exponentielle Fade startete mit maximaler Geschwindigkeit — sichtbarer
  // Ruck beim Uebergang zurueck in die Idle-Pose; Smoothstep hat an beiden
  // Enden Geschwindigkeit 0.
  if (c.fade != null) {
    // stopEmote(): weicher Abbruch aus der aktuellen Pose heraus
    c.fade += dt;
    c.w = c.fadeFrom * (1 - smoothstep(0, CLIP_FADE_OUT, c.fade));
  } else {
    c.w = smoothstep(0, CLIP_FADE_IN, c.t);
    if (c.once) {
      if (c.end) {
        // Rueckfuehrung in Zeitlupe (siehe CLIP_END): Bewegung bleibt
        // kontinuierlich, der Blend landet weich auf der Idle-Pose.
        const tau = c.action.time;
        c.action.timeScale =
          1 - (1 - c.end.slowTo) * smoothstep(c.end.slowAt, c.end.slowEnd, tau);
        c.w *= 1 - smoothstep(c.end.fadeAt, c.end.fadeEnd, tau);
      } else {
        c.w *= 1 - smoothstep(c.dur - CLIP_FADE_OUT, c.dur, c.t);
      }
    }
  }
  if (c.w <= 0.001 && c.t > CLIP_FADE_IN) {
    c.action.stop();
    state.clip = null;
    return;
  }
  for (let i = 0; i < c.nodes.length; i++) {
    const n = c.nodes[i];
    // Baseline fuer Bones ausserhalb des prozeduralen Layers: normalisierte
    // Ruhepose (Identity) — sonst bleibt Clip-Residuum haengen.
    if (!state.boneNodes || !state.boneNodes.has(n)) n.quaternion.copy(IDENT_Q);
    c.pre[i].copy(n.quaternion);
  }
  state.mixer.update(dt);
  for (let i = 0; i < c.nodes.length; i++) {
    const n = c.nodes[i];
    _q.copy(c.pre[i]).slerp(n.quaternion, c.w);
    n.quaternion.copy(_q);
  }
}

/* ---- Pose: Ziel-Rotationen pro Frame berechnen -------------------------- */
function computeTargets(dt) {
  const T = {};
  for (const b of BONES) T[b] = { x: 0, y: 0, z: 0 };
  const add = (b, x, y, z) => { T[b].x += x; T[b].y += y; T[b].z += z; };
  const t = state.t;

  // 1) Rest-Pose
  for (const b in REST) add(b, REST[b].x, REST[b].y, REST[b].z);

  // 2) Atmen (dezent — der Rest der Bewegung kommt aus Kopf/Armen)
  add('chest', Math.sin(t * 1.5) * 0.012, 0, 0);

  // 3) Idle-Randomness — immer aktiv, bei speaking reduziert
  const idleAmt = state.mode === 'speaking' ? 0.4 : 1.0;
  //   Kopf: alle paar Sekunden ein neues zufaelliges "Hinschauen/Neigen"
  state.idleHeadTimer += dt;
  if (state.idleHeadTimer >= state.idleHeadNext) {
    state.idleHeadTimer = 0;
    state.idleHeadNext = 3 + Math.random() * 5;      // 3–8s bis zur naechsten Pose
    state.idleHead = {
      x: -0.03 + Math.random() * 0.09,               // leicht rauf/runter
      y: (Math.random() - 0.5) * 0.36,               // links/rechts schauen
      z: (Math.random() - 0.5) * 0.22,               // Kopf neigen
    };
  }
  add('head', state.idleHead.x * idleAmt, state.idleHead.y * idleAmt, state.idleHead.z * idleAmt);
  add('neck', 0, state.idleHead.y * 0.3 * idleAmt, state.idleHead.z * 0.3 * idleAmt);
  //   Arme: langsames Schlendern (gegenphasig), plus Mikro-Drift
  const sp = state.swayPhase, sv = state.swaySpeed;
  add('leftUpperArm',  Math.sin(t * sv + sp) * 0.05 * idleAmt, 0, Math.sin(t * sv * 0.7 + sp) * 0.035 * idleAmt);
  add('rightUpperArm', Math.sin(t * sv + sp + Math.PI) * 0.05 * idleAmt, 0, -Math.sin(t * sv * 0.7 + sp) * 0.035 * idleAmt);
  add('leftLowerArm', 0, 0, Math.sin(t * sv * 1.3 + sp) * 0.02 * idleAmt);
  add('rightLowerArm', 0, 0, -Math.sin(t * sv * 1.3 + sp) * 0.02 * idleAmt);
  //   Koerper: sanfte Gewichtsverlagerung + langsames Links/Rechts-Pendeln
  const sway = Math.sin(t * 0.35 + sp) * idleAmt;
  add('spine', 0, Math.sin(t * 0.3 + sp) * 0.03, Math.sin(t * 0.23 + sp) * 0.015 + sway * 0.035);
  add('hips', 0, sway * 0.04, sway * 0.03);
  add('chest', 0, 0, -sway * 0.025);   // Oberkoerper haelt leicht gegen -> natuerlich
  add('head', 0, 0, -sway * 0.02);

  // 4) Modus-Overlays
  let squint = 0;
  let eyeWander = false;
  if (state.mode === 'thinking') {
    const tt = state.modeTime;
    // Stufe 1 (sofort): Kopf zur Seite neigen
    const s1 = smoothstep(0.2, 1.4, tt);
    add('head', 0.04 * s1, 0.08 * s1, 0.20 * s1);
    add('neck', 0, 0, 0.06 * s1);
    // Stufe 2 (ab ~2.5s): Augen nachdenklich zukneifen, Kopf senkt sich leicht
    const s2 = smoothstep(2.5, 3.8, tt);
    squint = 0.35 * s2;
    add('head', 0.05 * s2, 0.04 * s2, 0);
    // Stufe 3 (ab ~5s): "nach den Sternen sehen" — die Augen fixieren
    // Punkte ueber dem Kopf (updateLook). Der Squint oeffnet sich dafuer
    // wieder und der Kopf hebt sich leicht, sonst liest der Blick nicht.
    const s3 = smoothstep(5, 6, tt);
    eyeWander = tt > 5;
    squint *= 1 - 0.75 * s3;
    add('head', -0.12 * s3, 0, 0);
    add('neck', -0.04 * s3, 0, 0);
    // Stufe 4 (ab ~7s): EINMAL kurz am Kopf kratzen (VRMA-Clip, ~3.9s).
    // Eine Wiederholung erst, wenn nach dem ENDE des Clips weitere 5s
    // thinking vergangen sind (kommt praktisch selten vor).
    if (tt >= state.thinkScratchAt) {
      const dur = Waifu.emote('scratch') || 5.3;
      state.thinkScratchAt = tt + dur + 5;
    }
  } else if (state.mode === 'listening') {
    add('head', -0.02, 0, 0.07);           // aufmerksam, leicht geneigt
  } else if (state.mode === 'speaking') {
    add('head', state.mouth * 0.04, 0, 0); // Sprech-Emphase folgt dem Mund
  }

  // 5) Emote-Overlay — Koerperbewegung kommt aus den VRMA-Clips (EMOTE_CLIPS),
  //    hier laufen nur noch Mimik/Mund und das rein prozedurale Nicken.
  const exprT = { happy: 0, sad: 0, angry: 0, surprised: 0, relaxed: 0 };
  let mouthExtra = 0;
  const em = state.emote;
  if (em) {
    em.t += dt;
    const p = clamp01(em.t / em.dur);
    const amp = Math.sin(Math.min(1, p) * Math.PI);          // weich rein/raus
    const fast = clamp01(p * 4) * (1 - smoothstep(0.6, 1, p)); // schneller Attack
    switch (em.name) {
      case 'lachen':
        exprT.happy = amp;
        squint = Math.max(squint, 0.7 * amp);                 // lachende Augen
        add('head', Math.sin(t * 14) * 0.02 * amp, 0, 0);     // Lach-Zittern
        mouthExtra = Math.abs(Math.sin(t * 12)) * 0.6 * amp;
        break;
      case 'sigh':
        exprT.relaxed = 0.4 * amp; exprT.sad = 0.35 * amp;
        if (p > 0.35) mouthExtra = 0.18 * amp;                // hoerbares Ausatmen
        break;
      case 'surprise':
        exprT.surprised = fast;
        break;
      case 'nod':
        exprT.happy = 0.25 * amp;
        add('head', Math.sin(p * Math.PI * 4) * 0.12 * (1 - p), 0, 0);
        break;
      case 'grumble':
        exprT.angry = 0.5 * amp;
        break;
      case 'smile':
        exprT.happy = 0.6 * amp;
        add('head', -0.02 * amp, 0, 0.04 * amp);
        break;
    }
    if (p >= 1) state.emote = null;
  }

  state.squint = squint;
  return { T, exprT, mouthExtra, eyeWander };
}

/* ---- Augen: Blickziel setzen (Kamera oder "nach den Sternen sehen") ------ */
function updateLook(dt, eyeWander) {
  const look = state.look;
  if (!look) return;
  if (eyeWander) {
    state.lookWanderTimer += dt;
    if (!state.lookWander || state.lookWanderTimer >= state.lookWanderNext) {
      state.lookWanderTimer = 0;
      state.lookWanderNext = 0.9 + Math.random() * 0.4;    // ~1 s fixieren
      // Fixpunkt deutlich ueber dem Kopf (Weltkoordinaten); der naechste
      // liegt immer klar woanders (Seitenwechsel), wie beim Absuchen des
      // Sternenhimmels.
      _v.set(0, 1.4, 0);
      const head = state.vrm?.humanoid?.getRawBoneNode('head');
      if (head) head.getWorldPosition(_v);
      const side = state.lookWander && state.lookWander.x >= 0 ? -1 : 1;
      state.lookWander = {
        x: side * (0.25 + Math.random() * 0.6),
        y: _v.y + 0.7 + Math.random() * 0.7,
        z: _v.z + 0.25 + Math.random() * 0.55,
      };
    }
    // Sakkade: schnell zum neuen Punkt springen, dort ruhig fixieren
    const w = state.lookWander;
    look.position.x = damp(look.position.x, w.x, 14, dt);
    look.position.y = damp(look.position.y, w.y, 14, dt);
    look.position.z = damp(look.position.z, w.z, 14, dt);
    return;
  }
  state.lookWander = null;
  state.lookWanderTimer = 0;
  const cam = state.camera.position;
  look.position.x = damp(look.position.x, cam.x, 5, dt);
  look.position.y = damp(look.position.y, cam.y, 5, dt);
  look.position.z = damp(look.position.z, cam.z, 5, dt);
}

/* ---- Expressions: Blink + Squint + Mund + Emotionen ----------------------- */
function updateExpressions(dt, exprT, mouthExtra) {
  const mgr = state.vrm?.expressionManager;
  if (!mgr) return;

  // Auto-Blink
  state.blinkTimer += dt;
  let blink = 0;
  const bt = state.blinkTimer - state.nextBlink;
  if (bt >= 0 && bt < 0.2) {
    blink = bt < 0.1 ? bt / 0.1 : 1 - (bt - 0.1) / 0.1;
  } else if (bt >= 0.2) {
    state.blinkTimer = 0;
    state.nextBlink = 2 + Math.random() * 3;
  }
  state.squintCur = damp(state.squintCur, state.squint, 6, dt);
  mgr.setValue('blink', Math.max(blink, state.squintCur));

  // Mund: Lip-Sync + Emote-Anteil
  state.mouth += (state.mouthTarget - state.mouth) * Math.min(1, dt * 18);
  mgr.setValue('aa', clamp01(Math.max(state.mouth, mouthExtra)));

  // Emotionen weich nachziehen
  for (const k in state.expr) {
    state.expr[k] = damp(state.expr[k], exprT[k], 8, dt);
    mgr.setValue(k, clamp01(state.expr[k]));
  }
}

function applyPose(dt, T) {
  const h = state.vrm?.humanoid;
  if (!h) return;
  for (const b of BONES) {
    const node = h.getNormalizedBoneNode(b);
    if (!node) continue;
    const p = state.pose[b];
    p.x = damp(p.x, T[b].x, 6, dt);
    p.y = damp(p.y, T[b].y, 6, dt);
    p.z = damp(p.z, T[b].z, 6, dt);
    node.rotation.set(p.x, p.y, p.z);
  }
}

function loop() {
  const dt = Math.min(state.clock.getDelta(), 0.1);
  state.t += dt;
  state.modeTime += dt;
  if (state.vrm) {
    const { T, exprT, mouthExtra, eyeWander } = computeTargets(dt);
    applyPose(dt, T);
    updateClip(dt);          // VRMA-Clip ueber die prozedurale Pose blenden
    updateExpressions(dt, exprT, mouthExtra);
    updateLook(dt, eyeWander);
    state.vrm.update(dt); // Spring Bones + Expressions anwenden
  }
  state.controls?.update();
  state.renderer.render(state.scene, state.camera);
  scheduleFrame();
}

// requestAnimationFrame wird in Hintergrund-Tabs von den meisten Browsern
// gedrosselt oder komplett angehalten (Power-Saving) -> Avatar wuerde
// einfrieren. Waehrend das Dokument hidden ist, treibt ein setTimeout-Fallback
// den Loop weiter (selbst gedrosselt, aber er laeuft); sobald der Tab wieder
// sichtbar ist, wechselt der naechste Schedule-Aufruf automatisch zurueck.
function scheduleFrame() {
  if (typeof document !== 'undefined' && document.hidden) {
    state.rafIsTimeout = true;
    state.raf = setTimeout(loop, 200);
  } else {
    state.rafIsTimeout = false;
    state.raf = requestAnimationFrame(loop);
  }
}

function onResize() {
  const canvas = state.renderer?.domElement;
  if (!canvas) return;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w === 0 || h === 0) return;
  state.camera.aspect = w / h;
  state.camera.updateProjectionMatrix();
  state.renderer.setSize(w, h, false);
}

const Waifu = {
  async mount(canvas) {
    if (state.renderer) this.unmount();
    initScene(canvas);
    loop();
    window.addEventListener('resize', onResize);
    try {
      await loadVRM();
    } catch (e) {
      console.error('[waifu] VRM load failed:', e);
      throw e;
    }
    return this;
  },
  unmount() {
    if (state.raf) {
      if (state.rafIsTimeout) clearTimeout(state.raf); else cancelAnimationFrame(state.raf);
    }
    window.removeEventListener('resize', onResize);
    if (state.mixer) { try { state.mixer.stopAllAction(); } catch (_) {} }
    if (state.vrm) { VRMUtils.deepDispose(state.vrm.scene); state.vrm = null; }
    state.renderer?.dispose();
    Object.assign(state, { renderer: null, scene: null, camera: null, controls: null, look: null,
      mixer: null, clip: null, clipCache: {}, boneNodes: null, ready: false });
  },
  setMouth(v) { state.mouthTarget = clamp01(v || 0); },
  setState(s) {
    if (state.mode !== s) {
      state.mode = s; state.modeTime = 0;
      if (s === 'thinking') state.thinkScratchAt = 7;
    }
  },
  emote(name) {
    const dur = EMOTE_DUR[name];
    if (!dur) return 0;
    state.emote = { name, t: 0, dur };
    // Koerperanimation als VRMA-Clip, falls gemappt (Mimik laeuft parallel)
    if (EMOTE_CLIPS[name]) playClip(EMOTE_CLIPS[name], { once: true, emoteName: name });
    return dur;
  },
  stopEmote() {
    state.emote = null;
    const c = state.clip;                    // laufenden VRMA-Clip weich ausblenden
    if (c && c.fade == null) { c.fade = 0; c.fadeFrom = c.w; }
  },
  emoteActive() { return !!(state.emote || state.clip); },
  emoteNames() { return Object.keys(EMOTE_DUR); },
  setExpression(name, v) { state.vrm?.expressionManager?.setValue(name, v); },
  resize: onResize,
  isReady() { return state.ready; },
};

window.Waifu = Waifu;
export default Waifu;
