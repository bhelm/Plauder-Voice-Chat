/* Audio INPUT engine: mic capture + Silero VAD (B1 live uplink with the
   optional opus encoder), level meter, the timed one-shot recorder for
   enrollment/voice-clone takes, and push-to-talk. Owns all capture state
   incl. the `enrolling` binary-channel guard. Classic script — shares
   the global lexical scope with index.html at call time. */
let enrolling = false;

/* ===== Mikrofon + VAD ===== */
let micActive = false;
let vad = null;
let audioCtx = null;
let micStream = null;
let analyser = null;
let meterRaf = null;

/* ===== B1: input streaming (VAD mode) =====
   A ScriptProcessor taps the same mic stream as the level analyser
   and continuously sends 16 kHz frames to the server while speaking.
   This way almost nothing is left to upload at speech end → STT starts
   almost immediately. A ring buffer provides ~300 ms of lead-in before speech start. */
let inputTapNode = null;
let inputTapSink = null;
let inputDs16k = null;            // stateful resampler of the mic session
let inputPreRoll = [];            // ring buffer (Float32Array @16k)
let inputStreaming = false;       // true between stream.start and commit
let vadStreamSegId = null;        // Segment-ID des aktuell gestreamten Segments
const INPUT_PREROLL_SAMPLES = 16000 * 0.3;

// Continuous-phase linear resampler to 16 kHz. Stateful (one instance per
// capture session): carries the fractional read position and the previous
// buffer's last sample across calls. A stateless per-buffer loop resets
// the phase every ScriptProcessor buffer (~43 ms), silently dropping ~2
// input samples per buffer and creating a discontinuity at every seam —
// periodic clicks that degrade STT.
function makeDownsampler16k() {
  let phase = 0, last = 0;
  return function (float32, inRate) {
    if (!inRate || inRate === 16000) return new Float32Array(float32);
    if (!float32.length) return new Float32Array(0);
    const ratio = inRate / 16000;
    let t = phase;
    const out = new Float32Array(Math.ceil((float32.length - t) / ratio) + 1);
    let n = 0;
    for (;;) {
      const lo = Math.floor(t);
      if (lo + 1 >= float32.length) break;
      const frac = t - lo;
      const s0 = lo >= 0 ? float32[lo] : last;
      out[n++] = s0 * (1 - frac) + float32[lo + 1] * frac;
      t += ratio;
    }
    phase = t - float32.length;
    last = float32[float32.length - 1];
    return out.subarray(0, n);
  };
}

function streamInputActive() {
  return vadLike() && streamInputEnabled && ws && ws.readyState === 1;
}

// Backpressure: on a congested uplink (mobile, weak WLAN) the WS send
// buffer grows without bound and delays exactly the frames STT waits on —
// plus every control message queued behind them. Mic frames are the one
// thing we can shed: above this bound (~4 s raw f32, minutes of opus) new
// frames are dropped instead of queued. A short gap in the segment audio
// beats a commit that arrives half a minute late.
const UPLINK_MAX_BUFFERED = 256 * 1024;   // bytes
let _uplinkShedLogTs = 0;
function uplinkCongested() {
  if (!ws || ws.bufferedAmount <= UPLINK_MAX_BUFFERED) return false;
  const now = Date.now();
  if (now - _uplinkShedLogTs > 5000) {
    _uplinkShedLogTs = now;
    console.warn(`[uplink] congested (${ws.bufferedAmount} bytes buffered) — shedding mic frames`);
  }
  return true;
}

/* ===== Opus mic uplink (B1 segments) =====
   One AudioEncoder per streamed segment; each EncodedAudioChunk goes out
   as 0x4F ('O') + u16 BE length + packet. Because flush() is async, all
   opus start/commit/abort work is serialized on ONE promise chain so the
   wire order (start → frames → commit) can never interleave across
   segments. The chain normally resolves within a microtask, i.e. before
   the next mic callback; frames arriving before the queued start ran are
   still covered by the preroll ring buffer (flushed at start time).
   Enrollment and PTT/full segments stay raw f32 — only VAD/wake streaming
   segments are encoded. */
let inputOpusEnc = null;      // AudioEncoder of the ACTIVE opus segment
let inputOpusTs = 0;          // running sample counter → AudioData timestamps
let uplinkCodec = 'pcm';      // codec of the ACTIVE streamed segment
let uplinkGen = 0;            // segment generation (guards stale chained tasks)
let uplinkChain = Promise.resolve();
function chainUplink(fn) {
  uplinkChain = uplinkChain.then(fn, fn);
  return uplinkChain;
}

function makeUplinkOpusEncoder(gen) {
  const enc = new AudioEncoder({
    output: (chunk) => {
      if (!ws || ws.readyState !== 1) return;
      try {
        const pkt = new Uint8Array(chunk.byteLength);
        chunk.copyTo(pkt);
        const framed = new Uint8Array(3 + pkt.length);
        framed[0] = 0x4F;                       // 'O'
        framed[1] = (pkt.length >> 8) & 0xFF;   // u16 BE packet length
        framed[2] = pkt.length & 0xFF;
        framed.set(pkt, 3);
        ws.send(framed);
      } catch (_) {}
    },
    error: (e) => console.warn('[opus] uplink encoder error', e),
  });
  enc.configure({ codec: 'opus', sampleRate: 16000, numberOfChannels: 1, bitrate: 24000 });
  enc._gen = gen;   // frames may only enter the encoder of the CURRENT segment
  return enc;
}

function encodeUplinkFrame(f32) {
  if (!inputOpusEnc || inputOpusEnc._gen !== uplinkGen) return;
  if (inputOpusEnc.state !== 'configured' || !f32.length) return;
  try {
    const data = new AudioData({
      format: 'f32', sampleRate: 16000, numberOfFrames: f32.length,
      numberOfChannels: 1,
      timestamp: Math.round(inputOpusTs * 1e6 / 16000),
      data: f32,
    });
    inputOpusTs += f32.length;
    inputOpusEnc.encode(data);
    try { data.close(); } catch (_) {}
  } catch (e) { console.warn('[opus] uplink encode failed', e); }
}

function beginInputStream(segmentId) {
  if (!streamInputActive()) return false;
  const gen = ++uplinkGen;
  const meta = {
    type: 'segment.stream.start', segmentId,
    speechStartTs: lastSpeechStartTs, clientNow: Date.now(),
    bargeIn: pendingBargeIn,
  };
  if (!opusUplinkActive()) {
    // Raw f32 path — fully synchronous, exactly as before.
    uplinkCodec = 'pcm';
    try {
      ws.send(JSON.stringify(meta));
      inputStreaming = true;
      // Prepend the lead-in (ring buffer).
      for (const fr of inputPreRoll) ws.send(f32ToBytes(fr));
      lastSpeechStartTs = null;
      pendingBargeIn = false;
      return true;
    } catch (_) { inputStreaming = false; return false; }
  }
  uplinkCodec = 'opus';
  inputStreaming = true;
  lastSpeechStartTs = null;
  pendingBargeIn = false;
  chainUplink(() => {
    // Stale (aborted / superseded before the chain ran) → send nothing.
    if (uplinkGen !== gen || !inputStreaming) return;
    try {
      inputOpusEnc = makeUplinkOpusEncoder(gen);
      inputOpusTs = 0;
      meta.codec = 'opus';
    } catch (e) {
      console.warn('[opus] uplink encoder unavailable, falling back to PCM', e);
      inputOpusEnc = null;
      uplinkCodec = 'pcm';
    }
    try {
      ws.send(JSON.stringify(meta));
      // Prepend the lead-in — through the same encoder, order preserved
      // (it also holds any frames the tap saw before this task ran).
      for (const fr of inputPreRoll) {
        if (inputOpusEnc) encodeUplinkFrame(fr);
        else ws.send(f32ToBytes(fr));
      }
    } catch (_) { inputStreaming = false; }
  });
  return true;
}

function commitInputStream(segmentId) {
  if (!inputStreaming) return;
  inputStreaming = false;
  if (uplinkCodec !== 'opus') {
    try {
      ws.send(JSON.stringify({ type: 'segment.stream.commit', segmentId }));
      sttSent(segmentId);
    } catch (_) {}
    return;
  }
  chainUplink(async () => {
    const enc = inputOpusEnc;
    inputOpusEnc = null;
    if (enc) {
      // Every remaining packet must hit the wire BEFORE the commit JSON —
      // encoder outputs are delivered before flush() resolves.
      try { await enc.flush(); } catch (_) {}
      try { enc.close(); } catch (_) {}
    }
    try {
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: 'segment.stream.commit', segmentId }));
        sttSent(segmentId);
      }
    } catch (_) {}
  });
}

function abortInputStream() {
  if (!inputStreaming) { vadStreamSegId = null; return; }
  inputStreaming = false;
  vadStreamSegId = null;
  if (uplinkCodec !== 'opus') {
    try {
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'segment.stream.abort' }));
    } catch (_) {}
    return;
  }
  chainUplink(() => {
    const enc = inputOpusEnc;
    inputOpusEnc = null;
    if (enc) { try { enc.close(); } catch (_) {} }
    try {
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'segment.stream.abort' }));
    } catch (_) {}
  });
}

// VAD parameters maintained by the server (arrive via hello / settings.ack)
let currentVadParams = {
  redemptionFrames: 12,
  minSpeechFrames: 3,
  preSpeechPadFrames: 8,
  frameMs: 32,
};

// Barge-in: deliberately kept SIMPLE. As soon as the VAD detects speech or
// PTT is pressed, audio output is stopped immediately — no
// config, no ENV. The user controls VAD sensitivity directly via
// the threshold slider (positiveSpeechThreshold) in the web client.
// (vadActivationThreshold is declared further up — before the slider wiring.)
let vadRestartTimer = null;
let vadRestartInFlight = null;  // Promise-Mutex: serialisiert Restarts

function applyServerVadParams(params) {
  if (!params) return;
  const next = {
    redemptionFrames: Number.isFinite(params.redemptionFrames) ? params.redemptionFrames : currentVadParams.redemptionFrames,
    minSpeechFrames:  Number.isFinite(params.minSpeechFrames)  ? params.minSpeechFrames  : currentVadParams.minSpeechFrames,
    preSpeechPadFrames: Number.isFinite(params.preSpeechPadFrames) ? params.preSpeechPadFrames : currentVadParams.preSpeechPadFrames,
    frameMs: Number.isFinite(params.frameMs) ? params.frameMs : currentVadParams.frameMs,
  };
  // Direct comparison instead of JSON.stringify (more robust against field order).
  const changed =
       next.redemptionFrames   !== currentVadParams.redemptionFrames
    || next.minSpeechFrames    !== currentVadParams.minSpeechFrames
    || next.preSpeechPadFrames !== currentVadParams.preSpeechPadFrames
    || next.frameMs            !== currentVadParams.frameMs;
  const frameSizeChanged = next.frameMs !== currentVadParams.frameMs;
  currentVadParams = next;
  if (changed && micActive) {
    // frameMs maps to the worklet's frame size — that one genuinely needs
    // a rebuild. Everything else is patched on the running instance.
    if (frameSizeChanged || !applyVadOptionsLive({
          redemptionFrames: next.redemptionFrames,
          minSpeechFrames: next.minSpeechFrames,
          preSpeechPadFrames: next.preSpeechPadFrames,
        })) restartVadSoon();
  }
}

// Live-update of the RUNNING VAD's tuning: the library's frame processor
// reads its options object on every frame, so mutating it applies
// instantly — no stopMic/startMic (which re-runs getUserMedia + model
// init and leaves the mic deaf for ~0.5 s on every change). Returns false
// when there is no running instance to patch (caller falls back to a
// restart, or the values are simply picked up by the next startMic).
function applyVadOptionsLive(opts) {
  const fp = vad && vad.audioNodeVAD && vad.audioNodeVAD.frameProcessor;
  if (!fp || !fp.options) return false;
  Object.assign(fp.options, opts);
  return true;
}

// Debounced + serialized: if a restart is still running, the next one
// queues after it instead of starting a second mic in parallel.
// Only needed when live-patching is impossible (mic device change,
// frame size change, or no reachable frame processor).
function restartVadSoon() {
  if (!micActive) return;
  if (vadRestartTimer) clearTimeout(vadRestartTimer);
  vadRestartTimer = setTimeout(() => {
    vadRestartTimer = null;
    if (!micActive) return;
    const prev = vadRestartInFlight || Promise.resolve();
    vadRestartInFlight = prev.then(async () => {
      if (!micActive) return;
      try {
        await stopMic(true);
        await startMic(true);
      } catch (e) {
        addBubble('err', t('err.vad_restart', { err: e.message || e }));
      }
    }).finally(() => {
      if (vadRestartInFlight === prev) vadRestartInFlight = null;
    });
  }, 400);
}

function f32ToBytes(float32) {
  return new Uint8Array(float32.buffer, float32.byteOffset, float32.byteLength);
}

/* Shared capture plumbing: a ScriptProcessor tap wired through a muted
   gain node (audioprocess only fires when the chain reaches a
   destination; gain 0 so you do not hear yourself). One implementation
   for the VAD input tap, PTT and the enrollment/clone recorders. */
function connectMutedTap(ctx, src, frameSize, onFrame) {
  const tap = ctx.createScriptProcessor(frameSize, 1, 1);
  tap.onaudioprocess = onFrame;
  src.connect(tap);
  const sink = ctx.createGain(); sink.gain.value = 0;
  tap.connect(sink).connect(ctx.destination);
  return { tap, sink };
}
function disconnectMutedTap(chain) {
  if (!chain) return;
  try { chain.tap.disconnect(); } catch (_) {}
  try { chain.sink.disconnect(); } catch (_) {}
}

async function startMic(silent = false) {
  try {
    if (!silent) setMicUi('loading');
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: micConstraints(),
      video: false,
    });
    // Re-enumerate so device labels become visible after permission grant.
    refreshMicList();
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(micStream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);
    startMeter();

    // B1: tap for input streaming (ring buffer + live frames).
    inputPreRoll = [];
    inputStreaming = false;
    inputDs16k = makeDownsampler16k();
    try {
      const chain = connectMutedTap(audioCtx, src, 2048, (ev) => {
        if (enrolling) return;   // enrollment owns the binary channel
        const ch = ev.inputBuffer.getChannelData(0);
        const ds = inputDs16k(ch, audioCtx.sampleRate);
        // Maintain the ring buffer (always).
        inputPreRoll.push(ds);
        let total = 0;
        for (const f of inputPreRoll) total += f.length;
        while (total > INPUT_PREROLL_SAMPLES && inputPreRoll.length > 1) {
          total -= inputPreRoll.shift().length;
        }
        // During an active speech segment: stream the frame immediately.
        // Opus segments encode instead of sending raw; while the queued
        // stream start has not run yet (no encoder), the frame is simply
        // kept in the ring buffer above and flushed by the start task.
        if (inputStreaming && ws && ws.readyState === 1 && !uplinkCongested()) {
          if (uplinkCodec === 'opus') encodeUplinkFrame(ds);
          else try { ws.send(f32ToBytes(ds)); } catch (_) {}
        }
      });
      inputTapNode = chain.tap;
      inputTapSink = chain.sink;
    } catch (_) { inputTapNode = null; }

    if (!window.vad) throw new Error(t('err.vad_not_loaded'));
    if (!silent) setVad('warn', t('vad.loading'));
    vad = await window.vad.MicVAD.new({
      // Use the already-opened mic stream so the VAD captures from the
      // user-selected device instead of opening its own default-device stream.
      stream: micStream,
      // Local asset paths (no CDN)
      modelURL: `${BASE}/static/vendor/silero_vad.onnx`,
      workletURL: `${BASE}/static/vendor/vad.worklet.bundle.min.js`,
      // Vom User per Slider gewaehlte Aktivierungsschwelle. negative liegt
      // konventionell ~0.15 darunter (Hysterese), min. 0.05.
      positiveSpeechThreshold: vadActivationThreshold,
      negativeSpeechThreshold: Math.max(0.05, vadActivationThreshold - 0.15),
      minSpeechFrames: currentVadParams.minSpeechFrames,
      preSpeechPadFrames: currentVadParams.preSpeechPadFrames,
      redemptionFrames: currentVadParams.redemptionFrames,
      // BARGE-IN (simpel + schnell): onFrameProcessed feuert pro VAD-Frame
      // (~32ms) with the speech probability. As soon as it reaches the
      // threshold chosen by the user AND audio is currently playing, playback
      // is stopped IMMEDIATELY — no waiting for onSpeechStart or
      // the transcription. Latency ~ one frame (~32ms).
      onFrameProcessed: (probs) => {
        const p = probs && Number.isFinite(probs.isSpeech) ? probs.isSpeech : 0;
        // Show the live speech level in the UI (for tuning the threshold).
        if (vadProbBar) vadProbBar.style.width = Math.round(p * 100) + '%';
        // Barge-in: audio is playing AND level above threshold -> stop immediately.
        // (In wake mode only with an open window — otherwise not directed at Antonia.)
        if (anyAudioPlaying() && p >= vadActivationThreshold && bargeInAllowed()) bargeInStop('vad-frame');
      },
      onSpeechStart: () => {
        // Record speech start for the speaker-ID hold (before bargeInStop,
        // since this is the real start of speech).
        lastSpeechStartTs = Date.now();
        // Wake mode: while speaking, the window must NOT expire
        // (the chime timer is paused) — otherwise it closes mid-sentence.
        if (inputMode === 'wake' && wakeWindowActive) cancelWakeWindowTimer();
        // Safety net: in case the frame path does not kick in. In wake mode
        // with a closed window, do NOT abort (speech not directed at Antonia).
        // Only when audio is actually PLAYING: while Antonia merely thinks,
        // the abort is deferred to the server (post-STT + ghost filter), so
        // a hallucinated noise segment cannot kill the pending answer.
        if (bargeInAllowed() && anyAudioPlaying()) bargeInStop('vad-speech');
        // Voice lock: don't stop — duck the playback while the server
        // verifies the speaker. Owner → server stops it; foreign → restore.
        else if (voiceLockEngaged() && anyAudioPlaying()
                 && !(inputMode === 'wake' && !wakeWindowActive)) setDuck(true);
        // B1: start input streaming (frames now flow live).
        if (streamInputActive()) {
          segmentCounter += 1;
          segCount.textContent = String(segmentCounter);
          vadStreamSegId = String(segmentCounter);
          beginInputStream(vadStreamSegId);
        }
        setMicUi('speaking');
      },
      onSpeechEnd: (audio) => {
        if (inputStreaming && vadStreamSegId) {
          // Already streamed live → only commit now, do NOT send again.
          commitInputStream(vadStreamSegId);
          vadStreamSegId = null;
        } else {
          segmentCounter += 1;
          segCount.textContent = String(segmentCounter);
          sendSegment(String(segmentCounter), audio);
        }
        // Wake window was open → resume the timer (from now). For
        // accepted segments the server takes over shortly (reply.start →
        // pauses, playback.done → restart); this resume covers the
        // no-turn cases (empty/hallucinated) so nothing gets stuck.
        if (inputMode === 'wake' && wakeWindowActive && !isBusy()) armWakeWindowTimer(serverWake.windowS);
        // Only fall back to "Listening" when no segment is awaiting its
        // transcript — otherwise the "Transcribing …" state stays visible.
        setTimeout(() => { if (micActive && !sttPending.size) setMicUi('listening'); }, 100);
      },
      onVADMisfire: () => {
        // Speech too short → no segment. Discard the running input stream.
        setDuck(false);   // voice lock: release the duck, nothing to verify
        if (inputStreaming) abortInputStream();
        vadStreamSegId = null;
        if (inputMode === 'wake' && wakeWindowActive && !isBusy()) armWakeWindowTimer(serverWake.windowS);
        if (micActive && !sttPending.size) setMicUi('listening');
      },
    });
    await vad.start();
    setVad('ok', t('vad.active'));
    micActive = true;
    // On a silent restart, do not overwrite the previously running UI state
    // (e.g. 'playing') — otherwise it flickers.
    if (!silent) setMicUi('listening');
    else if (!currentAudioEl) setMicUi('listening');
  } catch (err) {
    addBubble('err', t('err.mic_vad', { err: err.message || err }));
    setVad('err', t('vad.error'));
    setMicUi('error');
    await stopMic(true);
  }
}

function sendSegment(segmentId, float32, opts) {
  if (!ws || ws.readyState !== 1) {
    addBubble('err', t('err.ws_not_connected_segment'));
    return;
  }
  // Während des Enrollments interpretiert der Server Binärframes als
  // Enroll-Aufnahme — ein Segment jetzt würde beide Ströme vermischen.
  if (enrolling) return;
  ws.send(JSON.stringify({
    type: 'segment.start',
    segmentId,
    samples: float32.length,
    sampleRate: 16000,
    format: 'f32le',
    // Speaker-ID hold: speech start + current client clock (both ms,
    // Date.now()). The server derives the speech start in its own
    // eigenen Zeitbasis. bargeIn markiert Ins-Wort-Fallen.
    speechStartTs: lastSpeechStartTs,
    clientNow: Date.now(),
    bargeIn: pendingBargeIn,
    // PTT = deliberate button press → the speaker lock trims but never
    // hard-rejects these segments.
    ptt: !!(opts && opts.ptt),
  }));
  ws.send(f32ToBytes(float32));
  sttSent(segmentId);
  // Consumed -> reset so the next segment gets fresh values
  // (otherwise an old speech-start/barge-in sticks to the next segment).
  lastSpeechStartTs = null;
  pendingBargeIn = false;
}

async function stopMic(silent = false) {
  micActive = false;
  if (inputStreaming) {
    // A user-initiated stop WHILE speech is streaming (spoke, then hit the
    // mic button before the VAD's silence timeout fired): flush the
    // utterance so it still gets transcribed + submitted, instead of
    // silently discarding it. On a silent restart (slider / device change)
    // a half-segment would be noise, so there we still abort.
    if (!silent && vadStreamSegId) commitInputStream(vadStreamSegId);
    else abortInputStream();
  }
  vadStreamSegId = null;
  try { if (vad) await vad.destroy(); } catch (_) {}
  vad = null;
  stopMeter();
  try { if (inputTapNode) inputTapNode.disconnect(); } catch (_) {}
  try { if (inputTapSink) inputTapSink.disconnect(); } catch (_) {}
  inputTapNode = null; inputTapSink = null; inputPreRoll = [];
  try { if (micStream) micStream.getTracks().forEach(t => t.stop()); } catch (_) {}
  micStream = null;
  try { if (audioCtx) await audioCtx.close(); } catch (_) {}
  audioCtx = null;
  analyser = null;
  meterBar.style.width = '0%';
  if (vadProbBar && !silent) vadProbBar.style.width = '0%';
  if (!silent) {
    // On a silent restart, do not throw the UI to 'stopped/idle' — that
    // caused visible flicker on every slider change.
    setVad('warn', t('vad.stopped'));
    setMicUi('idle');
  }
}

function startMeter() {
  const buf = new Uint8Array(analyser.fftSize);
  // Couple the gradient to the track width (once at startup) → see CSS.
  const trackW = (meterBar.parentElement && meterBar.parentElement.clientWidth) || 90;
  meterBar.style.backgroundSize = trackW + 'px 100%';
  const tick = () => {
    if (!analyser) return;
    analyser.getByteTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / buf.length);
    const pct = Math.min(100, Math.round(rms * 250));
    meterBar.style.width = pct + '%';
    meterRaf = requestAnimationFrame(tick);
  };
  tick();
}
function stopMeter() {
  if (meterRaf) cancelAnimationFrame(meterRaf);
  meterRaf = null;
}

/* Timed one-shot recording that OWNS the binary WS channel (enrollment
   and voice-clone reference takes share this one code path). Pauses the
   running VAD pipeline — its input tap would otherwise stream VAD-segment
   frames in parallel with the recording frames and the server can't tell
   them apart on the binary channel (garbled take) — opens its own capture
   chain, streams `seconds` of 16 kHz f32 frames, then sends `commitMsg()`;
   on failure `abortMsg` + an error bubble. Button/progress/label are the
   caller's UI elements. */
async function recordTimedTake({ seconds, btn, progEl, labelEl, labelKey,
                                 startMsg, commitMsg, abortMsg, errKey, onDone }) {
  enrolling = true;
  btn.disabled = true;
  btn.classList.add('recording');
  if (inputStreaming) abortInputStream();
  if (vad && micActive) { try { vad.pause(); } catch (_) {} }
  let ctx = null, stream = null, chain = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: micConstraints(), video: false });
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream);
    ws.send(JSON.stringify(startMsg));
    const target = 16000 * seconds;
    const ds16k = makeDownsampler16k();
    let sent = 0;
    await new Promise((resolve, reject) => {
      // Ohne Timeout hinge die Aufnahme für immer, wenn der
      // ScriptProcessor stumm stirbt (Tab im Hintergrund, Gerät weg) —
      // und das finally liefe nie: enrolling bliebe true, der VAD-Tap
      // dauerhaft unterdrückt (Spracheingabe tot bis zum Reload).
      const deadline = setTimeout(
        () => reject(new Error('recording timed out (no audio frames)')),
        seconds * 1000 + 8000);
      chain = connectMutedTap(ctx, src, 2048, (ev) => {
        if (sent >= target) return;
        const ds = ds16k(ev.inputBuffer.getChannelData(0), ctx.sampleRate);
        try { ws.send(f32ToBytes(ds)); } catch (_) {}
        sent += ds.length;
        const pct = Math.min(100, Math.round((sent / target) * 100));
        if (progEl) progEl.style.width = pct + '%';
        labelEl.textContent = t(labelKey, { pct });
        if (sent >= target) { clearTimeout(deadline); resolve(); }
      });
    });
    ws.send(JSON.stringify(commitMsg()));
    // Result comes back as the matching *.ack → handled in MSG_HANDLERS.
  } catch (e) {
    try { if (ws && ws.readyState === 1) ws.send(JSON.stringify(abortMsg)); } catch (_) {}
    addBubble('err', t(errKey, { err: (e && e.message) || e }));
  } finally {
    disconnectMutedTap(chain);
    try { if (stream) stream.getTracks().forEach(tr => tr.stop()); } catch (_) {}
    try { if (ctx) await ctx.close(); } catch (_) {}
    enrolling = false;
    btn.disabled = false;
    btn.classList.remove('recording');
    if (progEl) progEl.style.width = '0%';
    if (vad && micActive) { try { vad.start(); } catch (_) {} }
    if (onDone) onDone();
  }
}

/* ===== Push-to-Talk =====
   In PTT mode no VAD runs. Instead:
     mousedown / touchstart  -> startPttRecording()
     mouseup / mouseleave / touchend -> stopPttRecording() (sendet Segment)

   Important: the ScriptProcessor runs PERMANENTLY (once PTT mode
   is active), not only while the key is held. This solves two
   klassische PTT-Probleme:

     1. Pre-roll: the first few hundred ms after the key press
        would otherwise be cut off because the user starts talking right away.
        We keep a ring buffer of the last ~300ms and
        prepend it to the recording buffer on press.

     2. Post-roll: the last ~200ms after release would otherwise
        be cut off because the audio is still in the browser buffer.
        We let the processor keep running a bit longer.
*/
const PTT_PREROLL_MS = 300;
const PTT_POSTROLL_MS = 200;
const PTT_FRAME_SIZE = 4096;       // ~256 ms bei 16k
const PTT_PREROLL_FRAMES = Math.ceil((PTT_PREROLL_MS / 1000 * 16000) / PTT_FRAME_SIZE);
const PTT_POSTROLL_FRAMES = Math.ceil((PTT_POSTROLL_MS / 1000 * 16000) / PTT_FRAME_SIZE);

let pttCtx = null;
let pttStream = null;
let pttNode = null;
let pttSource = null;
let pttGainSink = null;
let pttPreRoll = [];               // ring buffer of the last N frames
let pttBuffers = [];               // active recording buffer
let pttRecording = false;
let pttPostRollFramesLeft = 0;     // if > 0, we keep collecting
let pttPendingFinalize = null;     // {segmentId, resolve} for post-roll
let pttArming = false;

async function ensurePttCtx() {
  // We start the audio stream + processor already on the mode switch,
  // so the ring buffer is pre-filled when the user presses the
  // key. Stream + context stay open during PTT mode.
  if (pttCtx && pttStream && pttSource && pttNode) return;
  pttStream = await navigator.mediaDevices.getUserMedia({
    audio: micConstraints(),
    video: false,
  });
  // Re-enumerate so device labels become visible after permission grant.
  refreshMicList();
  pttCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  if (pttCtx.state === 'suspended') {
    try { await pttCtx.resume(); } catch(_) {}
  }
  pttSource = pttCtx.createMediaStreamSource(pttStream);

  // analyser tap for the level meter
  analyser = pttCtx.createAnalyser();
  analyser.fftSize = 1024;
  pttSource.connect(analyser);
  startMeter();

  // Permanently running ScriptProcessor; collects into the ring buffer and
  // — when pttRecording — also into pttBuffers.
  const chain = connectMutedTap(pttCtx, pttSource, PTT_FRAME_SIZE, (ev) => {
    const ch = ev.inputBuffer.getChannelData(0);
    const copy = new Float32Array(ch);
    // maintain the ring buffer (always)
    pttPreRoll.push(copy);
    if (pttPreRoll.length > PTT_PREROLL_FRAMES) pttPreRoll.shift();
    // Aktiv aufnehmen
    if (pttRecording) {
      pttBuffers.push(copy);
    } else if (pttPostRollFramesLeft > 0) {
      // Post-roll: collect a few more frames after release
      // so the word endpoint is not cut off.
      pttBuffers.push(copy);
      pttPostRollFramesLeft -= 1;
      if (pttPostRollFramesLeft === 0 && pttPendingFinalize) {
        const fin = pttPendingFinalize;
        pttPendingFinalize = null;
        // finalize on the next tick so the running callback
        // returns cleanly
        setTimeout(() => fin(), 0);
      }
    }
  });
  pttGainSink = chain.sink;
  pttNode = chain.tap;
}

async function startPttRecording() {
  if (pttRecording || pttArming) return;
  if (enrolling) return;   // Enroll-Aufnahme läuft: PTT-Frames würden den binären Kanal korrumpieren
  // Speech start for the speaker-ID hold = moment of the key press (PTT has
  // keinen VAD-onSpeechStart). Vor bargeInStop setzen.
  lastSpeechStartTs = Date.now();
  // Barge-In Variante A: Tastendruck = klare Absicht "ich rede jetzt".
  // Stop ongoing TTS playback IMMEDIATELY, without debounce.
  bargeInStop('ptt-press');
  pttArming = true;
  try {
    await ensurePttCtx();
  } catch (err) {
    addBubble('err', t('err.mic', { err: err.message || err }));
    setMicUi('error');
    pttArming = false;
    return;
  }
  if (pttCtx.state === 'suspended') {
    try { await pttCtx.resume(); } catch(_) {}
  }
  // Pre-roll: use the ring-buffer frames collected so far as lead-in.
  // Take a copy so the ring buffer keeps running unhindered.
  pttBuffers = pttPreRoll.slice();
  pttPostRollFramesLeft = 0;
  pttPendingFinalize = null;
  pttRecording = true;
  pttArming = false;
  micBtn.classList.add('recording');
  setMicUi('recording');
}

async function stopPttRecording() {
  if (!pttRecording) return;
  pttRecording = false;
  micBtn.classList.remove('recording');

  // Instead of cutting off immediately: let post-roll frames be collected.
  // If no more callbacks arrive (e.g. context already gone),
  // the timeout fallback will trigger.
  const finalize = () => {
    let total = 0;
    for (const b of pttBuffers) total += b.length;
    const captured = pttBuffers;
    pttBuffers = [];
    if (total < 16000 * 0.20) {   // <200ms inkl. Pre-Roll: verwerfen
      setMicUi('listening');
      return;
    }
    const out = new Float32Array(total);
    let off = 0;
    for (const b of captured) { out.set(b, off); off += b.length; }
    segmentCounter += 1;
    segCount.textContent = String(segmentCounter);
    // sendSegment → sttSent shows "Transcribing …" (the honest state:
    // STT runs first; "thinking" follows via turn.commit).
    sendSegment(String(segmentCounter), out, { ptt: true });
  };
  pttPostRollFramesLeft = PTT_POSTROLL_FRAMES;
  pttPendingFinalize = finalize;
  // Fallback: if post-roll callbacks fail to arrive (context closed etc.)
  setTimeout(() => {
    if (pttPendingFinalize === finalize) {
      pttPendingFinalize = null;
      pttPostRollFramesLeft = 0;
      finalize();
    }
  }, PTT_POSTROLL_MS + 200);
}

async function teardownPttCtx() {
  pttRecording = false;
  pttPostRollFramesLeft = 0;
  // Eine noch ausstehende Post-Roll-Finalisierung SOFORT ausführen statt
  // wegwerfen — sonst verschluckt ein Mikro-/Moduswechsel direkt nach dem
  // Loslassen der PTT-Taste die komplette Aufnahme.
  if (pttPendingFinalize) {
    const fin = pttPendingFinalize;
    pttPendingFinalize = null;
    try { fin(); } catch (_) {}
  }
  pttBuffers = [];
  pttPreRoll = [];
  try { if (pttNode) pttNode.disconnect(); } catch(_) {}
  pttNode = null;
  try { if (pttGainSink) pttGainSink.disconnect(); } catch(_) {}
  pttGainSink = null;
  stopMeter();
  try { if (pttSource) pttSource.disconnect(); } catch(_) {}
  pttSource = null;
  try { if (pttStream) pttStream.getTracks().forEach(t => t.stop()); } catch(_) {}
  pttStream = null;
  try { if (pttCtx) await pttCtx.close(); } catch(_) {}
  pttCtx = null;
  analyser = null;
  meterBar.style.width = '0%';
}
