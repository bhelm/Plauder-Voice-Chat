/*
 * waifu_ui.js — UI-Integration des Waifu-Avatars in Plauder (Phase 3c)
 *
 * Klassisches Script (kein Modul), laeuft im globalen Scope wie vct/mic/playback.
 * Verantwortlich fuer:
 *   - Toggle (default AUS): laedt three-vrm erst beim Einschalten (dynamischer import)
 *   - Docked-Ansicht im Settings-Drawer
 *   - Pop-out in eigenes Fenster (Canvas wird VERSCHOBEN, nicht neu gebaut)
 *   - Lip-Sync + State: haengt sich nicht-invasiv an das globale setMicUi()
 *
 * ISOLATION: faellt hier irgendwas aus, laeuft der Voice-Chat unveraendert weiter.
 * Alles in try/catch, kein Wurf nach aussen.
 */
(function () {
  'use strict';

  var toggle, stageCard, stage, canvas, poppedNote, popoutBtn;
  var waifu = null;            // window.Waifu instance nach Load
  var loading = false;
  var popWin = null;           // Pop-out-Fenster
  var speakingPulse = null;    // Fallback-Mundanimation wenn keine Audio-Amplitude
  var LS_KEY = 'waifuEnabled';

  function $(id) { return document.getElementById(id); }

  function ready() {
    toggle = $('waifuToggle');
    stageCard = $('waifuStageCard');
    stage = $('waifuStage');
    canvas = $('waifuCanvas');
    poppedNote = $('waifuPoppedNote');
    popoutBtn = $('waifuPopoutBtn');
    if (!toggle) return; // Sektion nicht vorhanden -> nichts tun

    toggle.addEventListener('change', onToggle);
    if (popoutBtn) popoutBtn.addEventListener('click', onPopout);

    // Startzustand bestimmen:
    //   - Server-Default via window.__WAIFU_MODE__ (aus WAIFU_MODE env)
    //   - User-Override via localStorage gewinnt, falls der User selbst getoggelt hat
    var serverDefault = (window.__WAIFU_MODE__ === true);
    var stored = null;
    try { stored = localStorage.getItem(LS_KEY); } catch (_) {}

    var startOn;
    if (stored === '1') startOn = true;         // User hat explizit an
    else if (stored === '0') startOn = false;    // User hat explizit aus
    else startOn = serverDefault;                // kein User-Override -> Server entscheidet

    if (startOn) {
      toggle.checked = true;
      onToggle();
    }

    hookMicUi();
    hookReplyTags();
  }

  async function onToggle() {
    try {
      if (toggle.checked) {
        stageCard.style.display = '';
        try { localStorage.setItem(LS_KEY, '1'); } catch (_) {}
        await ensureLoaded();
      } else {
        stageCard.style.display = 'none';
        try { localStorage.setItem(LS_KEY, '0'); } catch (_) {}
        if (popWin && !popWin.closed) popWin.close();
        stopTalkDriver();
        if (waifu) { try { waifu.unmount(); } catch (_) {} waifu = null; }
      }
    } catch (e) {
      console.error('[waifu-ui] toggle failed:', e);
      if (toggle) toggle.checked = false;
    }
  }

  async function ensureLoaded() {
    if (waifu || loading) return;
    loading = true;
    try {
      // dynamischer Import erst JETZT -> kein three.js-Overhead solange aus
      var mod = await import((window.__BASE_PATH__ || '') + '/static/js/waifu.js');
      var W = (mod && mod.default) || window.Waifu;
      await W.mount(canvas);
      waifu = W;
      startTalkDriver();
    } finally {
      loading = false;
    }
  }

  /* ---- Pop-out: Canvas in eigenes Fenster verschieben ------------------ */
  function onPopout() {
    if (!waifu) return;
    if (popWin && !popWin.closed) { popWin.focus(); return; }

    popWin = window.open('', 'JoyAvatar', 'width=360,height=560,menubar=no,toolbar=no,location=no,status=no,resizable=yes');
    if (!popWin) { alert('Pop-out wurde blockiert (Popup-Blocker?).'); return; }

    var doc = popWin.document;
    doc.title = 'Joy';
    doc.body.style.cssText = 'margin:0;background:#14161c;overflow:hidden;';
    var host = doc.createElement('div');
    host.style.cssText = 'width:100vw;height:100vh;';
    host.title = 'Doppelklick: Vollbild';
    doc.body.appendChild(host);

    // Doppelklick -> Vollbild an/aus (zusaetzlich zum normalen Maximieren)
    host.addEventListener('dblclick', function () {
      try {
        if (doc.fullscreenElement) doc.exitFullscreen();
        else host.requestFullscreen();
      } catch (_) {}
    });
    doc.addEventListener('fullscreenchange', function () { try { waifu.resize(); } catch (_) {} });

    // Canvas physisch ins neue Dokument verschieben — Renderer/State bleiben erhalten
    host.appendChild(canvas);
    try { waifu.resize(); } catch (_) {}

    // Docked-Platzhalter zeigen
    poppedNote.style.display = 'flex';

    popWin.addEventListener('resize', function () { try { waifu.resize(); } catch (_) {} });
    popWin.addEventListener('beforeunload', dockBack);
  }

  function dockBack() {
    try {
      if (canvas && stage) {
        stage.insertBefore(canvas, poppedNote);
        poppedNote.style.display = 'none';
        if (waifu) { try { waifu.resize(); } catch (_) {} }
      }
    } catch (_) {}
    popWin = null;
  }

  /* ---- Lip-Sync + State: nicht-invasiv an globales setMicUi() ---------- */
  function hookMicUi() {
    if (typeof window.setMicUi !== 'function') {
      // setMicUi evtl. noch nicht definiert -> spaeter erneut versuchen
      return setTimeout(hookMicUi, 300);
    }
    if (window.__waifuMicHooked) return;
    window.__waifuMicHooked = true;

    var orig = window.setMicUi;
    window.setMicUi = function (state) {
      try { applyState(state); } catch (_) {}
      return orig.apply(this, arguments);
    };
  }

  function applyState(state) {
    if (!waifu) return;
    // Plauder-States -> Avatar (nur Koerpersprache; der Mund laeuft im
    // talkDriver direkt am echten Playback-Zustand, s.u.)
    switch (state) {
      case 'playing':          // Agent spricht (TTS laeuft)
        waifu.setState('speaking');
        break;
      case 'speaking':         // NUTZER spricht gerade
        waifu.setState('listening');
        break;
      case 'loading':
      case 'collecting':
        waifu.setState('thinking');
        break;
      default:                 // listening / idle / recording / error
        waifu.setState('idle');
    }
  }

  // Lip-Sync-Treiber: laeuft dauerhaft solange der Avatar gemountet ist und
  // prueft pro Frame den ECHTEN Playback-Zustand (anyAudioPlaying aus
  // playback.js). Damit ist der Mund unabhaengig von setMicUi-Events —
  // frueher stoppte ein VAD/Transcript-Event den Lip-Sync mitten in der
  // Wiedergabe, und ohne aktives Mic startete er gar nicht erst.
  function startTalkDriver() {
    if (speakingPulse) return;
    var t0 = performance.now();
    var step = function (now) {
      if (!waifu) { speakingPulse = null; return; }
      var playing = false;
      try {
        playing = (typeof anyAudioPlaying === 'function' && anyAudioPlaying())
               || (typeof replayPlaying !== 'undefined' && replayPlaying);
      } catch (_) {}
      if (playing) {
        var t = (now - t0) / 1000;
        var v = Math.abs(Math.sin(t * 9) * 0.5 + Math.sin(t * 14) * 0.3);
        waifu.setMouth(Math.min(1, v));
      } else {
        waifu.setMouth(0);
      }
      pumpEmotes(playing, now);
      speakingPulse = requestAnimationFrame(step);
    };
    speakingPulse = requestAnimationFrame(step);
  }

  function stopTalkDriver() {
    if (speakingPulse) cancelAnimationFrame(speakingPulse);
    speakingPulse = null;
    if (waifu) waifu.setMouth(0);
  }

  /* ---- Nonverbal-Tags -> Emotes ----------------------------------------- */
  // Tags stehen im Reply-Text ([laughter], [sigh], *lacht*, ...). Der Text
  // laeuft der Stimme voraus, deshalb werden erkannte Emotes gepuffert und
  // erst abgespielt, wenn wirklich Audio laeuft (talkDriver zieht sie ab).
  var TAG_EMOTES = [
    [/\[laughter\]|\*lacht\*|\bhaha+h*a*\b/i, 'laugh'],
    [/\[sigh\]|\*seufzt\*/i, 'sigh'],
    [/\[surprise-wa\]|\*staunt\*/i, 'surprise'],
    [/\[confirmation-en\]|\*nickt\*/i, 'nod'],
    [/\[dissatisfaction-hnn\]|\*brummt\*/i, 'grumble'],
    [/\*laechelt\*|\*lächelt\*|\*grinst\*/i, 'smile'],
  ];
  var emoteQueue = [];
  var emoteTail = '';        // Delta-Grenzen: Tag kann ueber 2 Deltas splitten
  var lastEmoteAt = 0;
  var lastPlayingAt = 0;     // letzter Zeitpunkt mit aktivem Audio (Stille-Timeout)
  var thinkingTurn = null;   // Turn, fuer den bereits 'thinking' gesetzt wurde

  function scanForEmotes(delta) {
    var text = emoteTail + (delta || '');
    for (var i = 0; i < TAG_EMOTES.length; i++) {
      if (TAG_EMOTES[i][0].test(text)) {
        if (emoteQueue.length < 4) emoteQueue.push(TAG_EMOTES[i][1]);
        text = text.replace(TAG_EMOTES[i][0], '');
      }
    }
    emoteTail = text.slice(-24);
  }

  function pumpEmotes(playing, now) {
    if (!playing) {
      // Erst nach laengerer Stille verwerfen (kurze Luecken zwischen
      // Audio-Chunks sollen einen gepufferten Emote nicht wegwerfen).
      if (emoteQueue.length && now - lastPlayingAt > 4000) emoteQueue.length = 0;
      return;
    }
    lastPlayingAt = now;
    if (!emoteQueue.length) return;
    // Mindestabstand nur ZWISCHEN Emotes; das erste nach Audio-Start sofort.
    if (lastEmoteAt && now - lastEmoteAt < 1600) return;
    lastEmoteAt = now;
    try { waifu.emote(emoteQueue.shift()); } catch (_) {}
  }

  function hookReplyTags() {
    if (typeof window.appendAgentBubbleDelta !== 'function') {
      return setTimeout(hookReplyTags, 300);
    }
    if (window.__waifuTagHooked) return;
    window.__waifuTagHooked = true;
    var orig = window.appendAgentBubbleDelta;
    window.appendAgentBubbleDelta = function (turnId, delta) {
      try {
        if (waifu) {
          // Erstes Delta dieses Turns => das LLM formuliert gerade => nachdenken.
          // (Bleibt bis der talkDriver bei echtem Audio auf 'speaking' schaltet.)
          if (turnId !== thinkingTurn) {
            thinkingTurn = turnId;
            waifu.setState('thinking');
          }
          scanForEmotes(delta);
        }
      } catch (_) {}
      return orig.apply(this, arguments);
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else {
    ready();
  }
})();
