/* Audio OUTPUT engine: progressive stream player (VCT2 PCM / VCT3 opus
   via WebCodecs), classic full-WAV playback, per-turn replay cache and
   the per-message speaker button, plus volume/duck. Owns all playback
   state (stream player, contexts, replay). Classic script — shares the
   global lexical scope with index.html (ws, t(), turn maps, …) at call
   time; only declarations run at load time. */
/* ===== Audio playback ===== */
let currentAudioEl = null;
let currentAudioId = null;
let currentTurnId = null;

/* ===== Echo guard: echo-cancellable output via WebRTC loopback =====
   Browser AEC only subtracts audio it knows as "far end": <audio> elements
   and WebRTC tracks. Our TTS is raw Web-Audio scheduling (BufferSources into
   ctx.destination) — Firefox's AEC never sees it, so on speakers the mic
   hears the assistant and the VAD barge-ins on its own reply. (Chrome mostly
   gets away with it because its AEC references the whole tab output —
   platform-dependent.) Fix: route the master gain through a local
   RTCPeerConnection pair into an <audio> element; the reply then counts as
   "call audio", which every browser's AEC cancels. Volume/duck/suspend logic
   is untouched — it all acts on the gain node BEFORE this sink.
   Preference: settings → "Echo guard"; default ON in Firefox, OFF elsewhere
   (persisted as 'echoLoopback'). Resolved lazily — the settings helpers live
   in index.html and don't exist yet when this file loads. */
let _echoLoopbackPref = null;
const _loopbackSinks = [];   // live sinks: { ctx, gain, label, sink }

function echoLoopbackEnabled() {
  if (_echoLoopbackPref === null) {
    let saved = null;
    try { saved = loadSavedSetting('echoLoopback'); } catch (_) {}
    _echoLoopbackPref = saved === '1' ? true
      : saved === '0' ? false
      : /firefox/i.test(navigator.userAgent || '');
  }
  return _echoLoopbackPref;
}

function _closeLoopbackSink(sink) {
  if (!sink) return;
  sink.dead = true;
  try { sink.audioEl.pause(); sink.audioEl.srcObject = null; } catch (_) {}
  try { if (sink.pc1) sink.pc1.close(); } catch (_) {}
  try { if (sink.pc2) sink.pc2.close(); } catch (_) {}
}

function _createLoopbackSink(ctx, gain, label) {
  const sink = {
    input: ctx.createMediaStreamDestination(),
    audioEl: new Audio(),
    pc1: null, pc2: null, ready: false, dead: false,
    // On any failure: fall back to direct output so audio is never lost.
    // (After fallback the dead sink may still be connected — harmless, its
    // peer connection is closed and ontrack is gated on `dead`.)
    fallback(reason) {
      if (this.dead) return;
      console.warn('[echo-guard] loopback unavailable (' + label + '): ' + reason
                   + ' — direct output');
      _closeLoopbackSink(this);
      try { gain.connect(ctx.destination); } catch (_) {}
    },
  };
  sink.audioEl.autoplay = true;
  (async () => {
    try {
      const pc1 = sink.pc1 = new RTCPeerConnection();
      const pc2 = sink.pc2 = new RTCPeerConnection();
      pc1.onicecandidate = (e) => { if (e.candidate) pc2.addIceCandidate(e.candidate).catch(() => {}); };
      pc2.onicecandidate = (e) => { if (e.candidate) pc1.addIceCandidate(e.candidate).catch(() => {}); };
      pc2.ontrack = (ev) => {
        if (sink.dead) return;
        sink.audioEl.srcObject = ev.streams[0] || new MediaStream([ev.track]);
        // Autoplay may be blocked before the first user gesture; retried in
        // _ensureLoopbackAlive() on every playback start.
        sink.audioEl.play().catch(() => {});
        sink.ready = true;
        console.log('[echo-guard] loopback active (' + label + ')');
      };
      for (const tr of sink.input.stream.getAudioTracks()) pc1.addTrack(tr, sink.input.stream);
      const offer = await pc1.createOffer();
      await pc1.setLocalDescription(offer);
      await pc2.setRemoteDescription(offer);
      const answer = await pc2.createAnswer();
      await pc2.setLocalDescription(answer);
      await pc1.setRemoteDescription(answer);
    } catch (e) {
      sink.fallback((e && e.message) || e);
    }
  })();
  // Local negotiation normally completes in well under a second.
  setTimeout(() => { if (!sink.ready) sink.fallback('negotiation timeout'); }, 3000);
  return sink;
}

/* Wire `gain` to the context's real output — direct, or through the loopback
   when the echo guard is on. Idempotent per (ctx, gain): re-routes live when
   the setting changes. */
function routeMasterOut(ctx, gain, label) {
  let entry = _loopbackSinks.find((s) => s.ctx === ctx && s.gain === gain);
  if (!entry) { entry = { ctx, gain, label, sink: null }; _loopbackSinks.push(entry); }
  try { gain.disconnect(); } catch (_) {}
  _closeLoopbackSink(entry.sink);
  entry.sink = null;
  const supported = typeof RTCPeerConnection === 'function'
    && typeof ctx.createMediaStreamDestination === 'function';
  if (echoLoopbackEnabled() && supported) {
    entry.sink = _createLoopbackSink(ctx, gain, label);
    gain.connect(entry.sink.input);
  } else {
    gain.connect(ctx.destination);
  }
}

function _ensureLoopbackAlive() {
  // Autoplay-blocked elements start once the user has interacted; playback
  // starts are always user-adjacent, so retry here.
  for (const e of _loopbackSinks) {
    if (e.sink && !e.sink.dead && e.sink.audioEl.paused && e.sink.audioEl.srcObject) {
      e.sink.audioEl.play().catch(() => {});
    }
  }
}

function setEchoLoopback(on) {
  _echoLoopbackPref = !!on;
  try { saveSetting('echoLoopback', on ? '1' : '0'); } catch (_) {}
  for (const e of _loopbackSinks) routeMasterOut(e.ctx, e.gain, e.label);
}

function loopbackAudioEls() {
  return _loopbackSinks.filter((e) => e.sink && !e.sink.dead).map((e) => e.sink.audioEl);
}

/* ===== Progressive PCM playback (A2, VCT2 chunks via Web Audio) =====
   Instead of a finished WAV file, raw PCM chunks arrive as soon as the
   server has them. We chain them gaplessly through a dedicated AudioContext
   (one AudioBufferSourceNode per chunk, scheduled in time). */
let playbackCtx = null;
let playbackGain = null;
// Current streaming player: { turnId, audioId, sampleRate, nextTime,
//   sources:Set, ended:bool, donePending:bool }
let streamPlayer = null;

function ensurePlaybackCtx() {
  if (!playbackCtx) {
    playbackCtx = new (window.AudioContext || window.webkitAudioContext)();
    playbackGain = playbackCtx.createGain();
    routeMasterOut(playbackCtx, playbackGain, 'playback');
  }
  _ensureLoopbackAlive();
  // Do NOT auto-resume if the user paused live playback via the
  // speaker button (otherwise incoming chunks would unintentionally
  // restart it).
  if (playbackCtx.state === 'suspended' && !playbackUserPaused) { try { playbackCtx.resume(); } catch (_) {} }
  if (playbackGain) playbackGain.gain.value = parseInt(volume.value, 10) / 100 * duckFactor;
  return playbackCtx;
}

function startStreamPlayback(turnId, audioId, sampleRate, codec) {
  // Hard-stop any running playback (a different turn) — replays too.
  stopStreamPlayback();
  stopCurrentAudio();
  stopReplaySource();
  playbackUserPaused = false;   // new playback → release a possibly paused ctx
  ensurePlaybackCtx();
  streamPlayer = {
    turnId, audioId, sampleRate: sampleRate || 24000,
    nextTime: 0, sources: new Set(), ended: false, donePending: true,
    lead: 0.04,   // scheduling lead-in; grows after underruns (jitter buffer)
    chunks: [],   // collect PCM for the replay cache
    cachedSamples: 0,
    decoder: null,       // opus downlink: WebCodecs AudioDecoder (VCT3)
    decoderTs: 0,        // running packet timestamp (ordering only)
    coalesce: [],        // decoded opus packets awaiting one merged push
    coalesceSamples: 0,
  };
  if (codec === 'opus') {
    try {
      streamPlayer.decoder = makeDownlinkOpusDecoder(turnId, sampleRate || 24000);
    } catch (e) {
      // Should not happen (we only requested opus when supported); chunks
      // of this turn would be undecodable — surface it once.
      console.warn('[opus] downlink decoder init failed', e);
    }
  }
  currentTurnId = turnId;
  currentAudioId = audioId;
  if (micActive) setMicUi('playing');
  refreshSpeakerButtons();
}

// Opus downlink (VCT3): decoded AudioData is converted to int16 and fed
// into the SAME gapless scheduling as raw VCT2 chunks — barge-in stop,
// turn gating, jitter lead-in, replay cache and playback.done all reuse
// the existing path. WebCodecs guarantees output ordering.
function makeDownlinkOpusDecoder(turnId, sampleRate) {
  const dec = new AudioDecoder({
    output: (audioData) => {
      try {
        // Gate at decode-OUTPUT time: a barge-in may have stopped this
        // stream while the decode was still in flight.
        if (!streamPlayer || streamPlayer.turnId !== turnId) { audioData.close(); return; }
        const n = audioData.numberOfFrames;
        const f32 = new Float32Array(n);
        audioData.copyTo(f32, { planeIndex: 0, format: 'f32-planar' });
        // Some decoders output 48 kHz regardless of the configured rate —
        // trust the decoded frame, not audio.start (before the 1st chunk).
        if (audioData.sampleRate && streamPlayer.sampleRate !== audioData.sampleRate
            && streamPlayer.nextTime === 0 && !streamPlayer.cachedSamples) {
          streamPlayer.sampleRate = audioData.sampleRate;
        }
        audioData.close();
        const int16 = new Int16Array(n);
        for (let i = 0; i < n; i++) {
          const v = Math.max(-1, Math.min(1, f32[i]));
          int16[i] = v < 0 ? v * 32768 : v * 32767;
        }
        queueDecodedChunk(turnId, int16);
      } catch (_) { try { audioData.close(); } catch (_) {} }
    },
    error: (e) => console.warn('[opus] downlink decoder error', e),
  });
  dec.configure({ codec: 'opus', sampleRate, numberOfChannels: 1 });
  return dec;
}

function feedOpusChunk(chunk) {
  if (!streamPlayer || streamPlayer.turnId !== chunk.turnId) return;
  const dec = streamPlayer.decoder;
  if (!dec || dec.state !== 'configured') return;
  for (const pkt of chunk.packets) {
    try {
      dec.decode(new EncodedAudioChunk({
        type: 'key', timestamp: streamPlayer.decoderTs, data: pkt }));
      streamPlayer.decoderTs += 20000;   // nominal 20 ms/packet (ordering only)
    } catch (e) { console.warn('[opus] decode failed', e); }
  }
}

// Coalesce decoded opus packets (20 ms each) into ~300 ms buffers before
// scheduling. One BufferSource per packet means ~50 nodes/s — a long reply
// synthesizes much faster than realtime, so thousands of scheduled nodes
// pile up and load the audio thread (audible as crackle from mid-reply to
// the end). The PCM path ships ~400 ms per node; this matches that. No
// latency cost: the server already batches ~tts_chunk_ms of packets per
// VCT3 frame, so packets arrive in bursts of that size anyway.
const OPUS_COALESCE_S = 0.3;
function queueDecodedChunk(turnId, int16) {
  if (!streamPlayer || streamPlayer.turnId !== turnId) return;
  streamPlayer.coalesce.push(int16);
  streamPlayer.coalesceSamples += int16.length;
  if (streamPlayer.coalesceSamples >= streamPlayer.sampleRate * OPUS_COALESCE_S) {
    flushDecodedChunks(turnId);
  }
}

function flushDecodedChunks(turnId) {
  if (!streamPlayer || streamPlayer.turnId !== turnId) return;
  const parts = streamPlayer.coalesce;
  if (!parts.length) return;
  const merged = new Int16Array(streamPlayer.coalesceSamples);
  let off = 0;
  for (const p of parts) { merged.set(p, off); off += p.length; }
  streamPlayer.coalesce = [];
  streamPlayer.coalesceSamples = 0;
  pushStreamChunk(turnId, merged);
}

function closeStreamDecoder(sp) {
  if (sp && sp.decoder) {
    try { sp.decoder.close(); } catch (_) {}
    sp.decoder = null;
  }
}

function pushStreamChunk(turnId, int16) {
  if (!streamPlayer || streamPlayer.turnId !== turnId) return;
  if (!int16 || !int16.length) return;
  if (streamPlayer.chunks) {   // for the replay cache — capped (long replies)
    streamPlayer.chunks.push(int16);
    streamPlayer.cachedSamples += int16.length;
    if (streamPlayer.cachedSamples > streamPlayer.sampleRate * 300) {
      streamPlayer.chunks = null;   // >5 min: skip caching, free the PCM
    }
  }
  const ctx = ensurePlaybackCtx();
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
  const buf = ctx.createBuffer(1, f32.length, streamPlayer.sampleRate);
  buf.getChannelData(0).set(f32);
  const node = ctx.createBufferSource();
  node.buffer = buf;
  node.connect(playbackGain);
  const now = ctx.currentTime;
  // Lead-in on the first chunk so nothing gets swallowed; after a real
  // underrun (a chunk arrived after its slot passed) grow the lead so a
  // jittery link converges on a stable buffer instead of gapping forever.
  if (streamPlayer.nextTime < now) {
    if (streamPlayer.nextTime > 0) {
      streamPlayer.lead = Math.min(0.25, streamPlayer.lead * 2);
    }
    streamPlayer.nextTime = now + streamPlayer.lead;
  }
  const startAt = streamPlayer.nextTime;
  node.start(startAt);
  streamPlayer.nextTime = startAt + buf.duration;
  streamPlayer.sources.add(node);
  node.onended = () => {
    if (!streamPlayer) return;
    streamPlayer.sources.delete(node);
    maybeFinishStream();
  };
}

function maybeFinishStream() {
  if (!streamPlayer || !streamPlayer.ended) return;
  if (streamPlayer.sources.size > 0) return;
  const turnId = streamPlayer.turnId, audioId = streamPlayer.audioId;
  const fireDone = streamPlayer.donePending;
  const chunks = streamPlayer.chunks || [];
  const sr = streamPlayer.sampleRate;
  closeStreamDecoder(streamPlayer);   // opus: normally already closed at audio.end
  streamPlayer = null;
  if (currentTurnId === turnId) { currentTurnId = null; currentAudioId = null; }
  setDuck(false);        // playback over → release a possible voice-lock duck
  // Played to the end normally → cache the complete audio of this turn (replay).
  if (chunks.length) {
    try { cacheTts(turnId, URL.createObjectURL(pcmChunksToWavBlob(chunks, sr))); } catch (_) {}
  }
  setAgentBusy(false);   // reply fully played → mic back to 🎤
  if (micActive && !currentAudioEl) setMicUi('listening');
  refreshSpeakerButtons();
  // House-Mode speaker-ID: ended normally → open the conversation window.
  if (fireDone) {
    try {
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: 'playback.done', turnId, audioId, ts: Date.now() }));
      }
    } catch (_) {}
  }
}

function endStreamPlayback(turnId) {
  if (!streamPlayer || streamPlayer.turnId !== turnId) return;
  flushDecodedChunks(turnId);   // opus: ship the sub-threshold tail
  streamPlayer.ended = true;
  maybeFinishStream();
}

function stopStreamPlayback() {
  setDuck(false);
  if (!streamPlayer) return;
  const sp = streamPlayer;
  streamPlayer = null;            // detach first, so onended does nothing anymore
  closeStreamDecoder(sp);         // opus: drop in-flight decodes with the stream
  for (const n of sp.sources) { try { n.stop(); } catch (_) {} try { n.disconnect(); } catch (_) {} }
  if (currentTurnId === sp.turnId) { currentTurnId = null; currentAudioId = null; }
  // Aborted (barge-in/stop) → don't cache; possibly clear the pause flag.
  if (playbackUserPaused) { playbackUserPaused = false; if (playbackCtx) { try { playbackCtx.resume(); } catch (_) {} } }
  refreshSpeakerButtons();
}

function anyAudioPlaying() {
  return !!currentAudioEl || !!streamPlayer;
}

/* Voice lock: while the server checks WHO is speaking, playback is only
   DUCKED (lowered), not stopped. Owner confirmed → server sends audio.stop;
   foreign voice → transcript.ignored restores the volume. A safety timer
   guarantees the duck can never get stuck. */
let duckFactor = 1.0;
let duckTimer = null;
function applyPlaybackVolume() {
  const vol = parseInt(volume.value, 10) / 100;
  if (playbackGain) playbackGain.gain.value = vol * duckFactor;
  if (currentAudioEl) currentAudioEl.volume = Math.min(1, vol * duckFactor);
  if (replayGain) replayGain.gain.value = vol * duckFactor;
}
function setDuck(on) {
  const f = on ? 0.2 : 1.0;
  if (duckTimer) { clearTimeout(duckTimer); duckTimer = null; }
  if (on) duckTimer = setTimeout(() => setDuck(false), 5000);
  if (duckFactor === f) return;
  duckFactor = f;
  applyPlaybackVolume();
}
// Active audio IDs per turn, so we can stop them all on discardTurn.
const turnAudioIds = new Map(); // turnId -> Set<audioId>


function playWavBytes(turnId, audioId, arrayBuffer) {
  try {
    const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    if (currentAudioEl) {
      try { currentAudioEl.pause(); } catch(_) {}
      // Do NOT revoke the URL: it now belongs to the replay cache (see below).
    }
    const a = new Audio(url);
    a.volume = parseInt(volume.value, 10) / 100;
    currentAudioEl = a;
    currentAudioId = audioId;
    currentTurnId = turnId;
    cacheTts(turnId, url);   // for the speaker button (replay)
    if (micActive) setMicUi('playing');
    refreshSpeakerButtons();
    a.addEventListener('ended', () => {
      // URL stays in the cache (for replay) → do NOT revoke here.
      if (currentAudioEl === a) {
        currentAudioEl = null; currentAudioId = null; currentTurnId = null;
        setAgentBusy(false);
        if (micActive) setMicUi('listening');
      }
      refreshSpeakerButtons();
      // House-Mode speaker-ID: TTS played to the end normally -> server
      // starts the 3-s conversation window. Send ONLY here, NOT in
      // stopCurrentAudio() (barge-in/stop) -> no window start there.
      try {
        if (ws && ws.readyState === 1) {
          ws.send(JSON.stringify({
            type: 'playback.done', turnId: turnId, audioId: audioId, ts: Date.now(),
          }));
        }
      } catch (_) {}
    });
    a.addEventListener('error', () => {
      addBubble('err', t('err.audio_playback'));
      ttsCache.delete(turnId);
      try { URL.revokeObjectURL(url); } catch (_) {}
      refreshSpeakerButtons();
    });
    a.play().catch(err => {
      addBubble('err', t('err.autoplay_blocked', { err: err.message || err }));
    });
  } catch (err) {
    addBubble('err', t('err.audio', { err: err.message || err }));
  }
}

function stopCurrentAudio() {
  setDuck(false);
  if (currentAudioEl) {
    try { currentAudioEl.pause(); } catch(_) {}
    // Do NOT revoke the URL: it belongs to the replay cache.
    currentAudioEl = null; currentAudioId = null; currentTurnId = null;
    refreshSpeakerButtons();
  }
}

function stopAudioForTurn(turnId) {
  if (!turnId) return;
  if (currentTurnId === turnId) stopCurrentAudio();
  if (streamPlayer && streamPlayer.turnId === turnId) stopStreamPlayback();
  turnAudioIds.delete(turnId);
}

/* ===== Per-message speaker button: pause / resume / replay of the TTS ==
   - Short tap: if this message's TTS is running → pause; again → resume;
     if it's finished → play again.
   - Hold: restart from the beginning (regardless of whether it's currently playing).
   Each agent turn is cached as WAV (streaming chunks or finished WAV),
   so replay works instantly & without the server. The LIVE streaming playback is
   paused via AudioContext.suspend()/resume(). */
const ttsCache = new Map();        // turnId -> { url }
const TTS_CACHE_MAX = 40;
let playbackUserPaused = false;    // LIVE streaming paused via suspend()?
// Replays run through Web Audio, NOT an <audio> element: HTMLAudio uses a
// different audio route than the Web-Audio live path on mobile (iOS media
// channel vs. context routing) — replays came out of the wrong speaker.
// Own context (not playbackCtx), so pauseLiveStream()'s suspend() and a
// paused replay stay independent of each other.
let replayCtx = null, replayGain = null, replaySrc = null;
let replayTurnId = null;
let replayPlaying = false, replayPaused = false;
let replayGen = 0;                 // guards async decode against re-taps

function ensureReplayCtx() {
  if (!replayCtx) {
    replayCtx = new (window.AudioContext || window.webkitAudioContext)();
    replayGain = replayCtx.createGain();
    routeMasterOut(replayCtx, replayGain, 'replay');
  }
  _ensureLoopbackAlive();
  return replayCtx;
}

function stopReplaySource() {
  if (replaySrc) {
    const s = replaySrc; replaySrc = null;
    try { s.onended = null; s.stop(); } catch (_) {}
    try { s.disconnect(); } catch (_) {}
  }
  replayPlaying = false; replayPaused = false;
}

function cacheTts(turnId, url) {
  if (!turnId || !url) return;
  const old = ttsCache.get(turnId);
  if (old && old.url && old.url !== url) { try { URL.revokeObjectURL(old.url); } catch (_) {} }
  ttsCache.set(turnId, { url });
  while (ttsCache.size > TTS_CACHE_MAX) {
    const k = ttsCache.keys().next().value;
    const v = ttsCache.get(k); ttsCache.delete(k);
    if (v && v.url && k !== replayTurnId && k !== currentTurnId) {
      try { URL.revokeObjectURL(v.url); } catch (_) {}
    }
  }
  refreshSpeakerButtons();
}

function pcmChunksToWavBlob(chunks, sampleRate) {
  let total = 0; for (const c of chunks) total += c.length;
  const dataLen = total * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const dv = new DataView(buf);
  const wstr = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  wstr(0, 'RIFF'); dv.setUint32(4, 36 + dataLen, true); wstr(8, 'WAVE');
  wstr(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true);
  dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  wstr(36, 'data'); dv.setUint32(40, dataLen, true);
  let off = 44;
  for (const c of chunks) { for (let i = 0; i < c.length; i++) { dv.setInt16(off, c[i], true); off += 2; } }
  return new Blob([buf], { type: 'audio/wav' });
}

function pauseLiveStream() {
  if (playbackCtx) { try { playbackCtx.suspend(); } catch (_) {} }
  playbackUserPaused = true; refreshSpeakerButtons();
}
function resumeLiveStream() {
  playbackUserPaused = false;
  if (replayPlaying) pauseReplay();   // never both at once
  if (playbackCtx) { try { playbackCtx.resume(); } catch (_) {} }
  refreshSpeakerButtons();
}

async function startReplay(turnId, fromStart) {
  const c = ttsCache.get(turnId);
  if (!c) return;
  // Mute other playback so nothing sounds twice.
  if (streamPlayer && !playbackUserPaused) pauseLiveStream();
  if (currentAudioEl && !currentAudioEl.paused) { try { currentAudioEl.pause(); } catch (_) {} }
  ensureReplayCtx();
  stopReplaySource();
  const gen = ++replayGen;
  try {
    if (!c.buf) {
      const ab = await (await fetch(c.url)).arrayBuffer();
      c.buf = await replayCtx.decodeAudioData(ab);
    }
  } catch (_) { return; }
  if (gen !== replayGen) return;   // superseded by a newer tap mid-decode
  const src = replayCtx.createBufferSource();
  src.buffer = c.buf;
  src.connect(replayGain);
  src.onended = () => {
    if (replaySrc === src) { replaySrc = null; replayPlaying = false; replayPaused = false; }
    refreshSpeakerButtons();
  };
  replaySrc = src;
  replayTurnId = turnId;
  replayPlaying = true; replayPaused = false;
  applyPlaybackVolume();
  try { await replayCtx.resume(); } catch (_) {}
  src.start();
  refreshSpeakerButtons();
}

function pauseReplay() {
  if (replayCtx) { try { replayCtx.suspend(); } catch (_) {} }
  replayPlaying = false; replayPaused = true;
  refreshSpeakerButtons();
}
function resumeReplay() {
  replayPlaying = true; replayPaused = false;
  if (replayCtx) { try { replayCtx.resume(); } catch (_) {} }
  refreshSpeakerButtons();
}

function onSpeakerTap(turnId) {
  if (streamPlayer && streamPlayer.turnId === turnId) {        // LIVE (streaming)
    if (playbackUserPaused) resumeLiveStream(); else pauseLiveStream();
    return;
  }
  if (currentAudioEl && currentTurnId === turnId) {            // LIVE (classic WAV)
    if (currentAudioEl.paused) currentAudioEl.play().catch(() => {}); else currentAudioEl.pause();
    refreshSpeakerButtons(); return;
  }
  if (replayTurnId === turnId && replaySrc) {                  // replay playing / paused
    if (replayPlaying) { pauseReplay(); return; }
    if (replayPaused) { resumeReplay(); return; }
  }
  startReplay(turnId, false);                                  // otherwise: play again
}
function onSpeakerHold(turnId) {
  if (ttsCache.has(turnId)) startReplay(turnId, true);         // restart from the beginning
}

function refreshSpeakerButtons() {
  // Update legacy .spk-btn elements (kept for any edge-case callers)
  document.querySelectorAll('.spk-btn[data-turn]').forEach((btn) => {
    const T = btn.dataset.turn;
    const liveStream = !!(streamPlayer && streamPlayer.turnId === T);
    const liveWav = !!(currentAudioEl && currentTurnId === T);
    const hasAudio = ttsCache.has(T) || liveStream || liveWav;
    let playing = false, paused = false;
    if (liveStream) { playing = !playbackUserPaused; paused = playbackUserPaused; }
    else if (liveWav) { playing = !currentAudioEl.paused; paused = currentAudioEl.paused; }
    else if (replayTurnId === T) {
      playing = replayPlaying;
      paused = replayPaused;
    }
    btn.classList.toggle('disabled', !hasAudio);
    btn.textContent = playing ? '⏸' : (paused ? '▶️' : '🔊');
    btn.title = hasAudio ? t('speaker.tip') : t('speaker.no_audio');
  });
  // Update kebab menu buttons
  document.querySelectorAll('.msg-menu-wrap[data-turn]').forEach((wrap) => {
    const T = wrap.dataset.turn;
    const liveStream = !!(streamPlayer && streamPlayer.turnId === T);
    const liveWav = !!(currentAudioEl && currentTurnId === T);
    const hasAudio = ttsCache.has(T) || liveStream || liveWav;
    let playing = false, paused = false;
    if (liveStream) { playing = !playbackUserPaused; paused = playbackUserPaused; }
    else if (liveWav) { playing = !currentAudioEl.paused; paused = currentAudioEl.paused; }
    else if (replayTurnId === T) {
      playing = replayPlaying;
      paused = replayPaused;
    }
    const menuBtn = wrap.querySelector('.msg-menu-btn');
    if (menuBtn) menuBtn.classList.toggle('disabled', !hasAudio);
    const playItem = wrap.querySelector('.mm-play');
    const dlItem = wrap.querySelector('.mm-download');
    if (playItem) {
      playItem.disabled = !hasAudio;
      const icon = playing ? '⏸' : '▶️';
      const label = playing ? t('msg.pause_audio') : t('msg.play_audio');
      playItem.textContent = '';
      playItem.append(icon + ' ' + label);
    }
    if (dlItem) dlItem.disabled = !hasAudio;
  });
}

// Close any open message menu
let _openMenu = null;
function closeMessageMenu() {
  if (_openMenu) { _openMenu.classList.remove('open'); _openMenu = null; }
}
document.addEventListener('click', (e) => {
  if (_openMenu && !e.target.closest('.msg-menu-wrap')) closeMessageMenu();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeMessageMenu();
});

function downloadTtsAudio(turnId) {
  const c = ttsCache.get(turnId);
  if (!c || !c.url) return;
  const a = document.createElement('a');
  a.href = c.url;
  const prefix = AGENT_NAME.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'voice-reply';
  a.download = prefix + '-' + (turnId || 'audio').substr(0, 8) + '.wav';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function makeSpeakerButton(turnId) {
  // Returns a ⋮ menu wrapper (replaces the old bare speaker button).
  const wrap = document.createElement('div');
  wrap.className = 'msg-menu-wrap';
  if (turnId) wrap.dataset.turn = turnId;

  const btn = document.createElement('button');
  btn.className = 'msg-menu-btn';
  btn.textContent = '\u22EE';   // ⋮
  btn.title = t('msg.menu');
  btn.setAttribute('aria-label', t('msg.menu'));
  btn.addEventListener('click', (e) => {
    e.preventDefault(); e.stopPropagation();
    if (btn.classList.contains('disabled')) return;
    const drop = wrap.querySelector('.msg-menu-drop');
    if (!drop) return;
    if (drop.classList.contains('open')) { closeMessageMenu(); return; }
    closeMessageMenu();   // close any other open menu first
    drop.classList.add('open');
    _openMenu = drop;
  });

  const drop = document.createElement('div');
  drop.className = 'msg-menu-drop';

  // Play / Pause item
  const playItem = document.createElement('button');
  playItem.className = 'mm-play';
  playItem.textContent = '▶️ ' + t('msg.play_audio');
  playItem.addEventListener('click', (e) => {
    e.preventDefault(); e.stopPropagation();
    if (turnId) onSpeakerTap(turnId);
    closeMessageMenu();
  });
  drop.appendChild(playItem);

  // Replay from start
  const replayItem = document.createElement('button');
  replayItem.className = 'mm-replay';
  replayItem.textContent = '🔁 ' + t('msg.replay_audio');
  replayItem.addEventListener('click', (e) => {
    e.preventDefault(); e.stopPropagation();
    if (turnId) onSpeakerHold(turnId);
    closeMessageMenu();
  });
  drop.appendChild(replayItem);

  // Download item
  const dlItem = document.createElement('button');
  dlItem.className = 'mm-download';
  dlItem.textContent = '💾 ' + t('msg.download_audio');
  dlItem.addEventListener('click', (e) => {
    e.preventDefault(); e.stopPropagation();
    if (turnId) downloadTtsAudio(turnId);
    closeMessageMenu();
  });
  drop.appendChild(dlItem);

  wrap.appendChild(btn);
  wrap.appendChild(drop);

  // Also add a hidden .spk-btn so refreshSpeakerButtons legacy path works
  const legacyBtn = document.createElement('button');
  legacyBtn.className = 'spk-btn';
  if (turnId) legacyBtn.dataset.turn = turnId;
  legacyBtn.textContent = '🔊';
  wrap.appendChild(legacyBtn);

  return wrap;
}
