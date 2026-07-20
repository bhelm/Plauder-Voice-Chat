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
  var animSelect, animBtn;
  var waifu = null;            // window.Waifu instance nach Load
  var loading = false;
  var popWin = null;           // Pop-out-Fenster
  var speakingPulse = null;    // Fallback-Mundanimation wenn keine Audio-Amplitude
  var speakingPulseIsTimeout = false; // true, wenn speakingPulse gerade per setTimeout laeuft (Tab hidden)
  var animTest = null;         // { kind:'emote'|'state', name, timer } — manueller Anim-Test
  var LS_KEY = 'waifuEnabled';

  function $(id) { return document.getElementById(id); }

  function ready() {
    toggle = $('waifuToggle');
    stageCard = $('waifuStageCard');
    stage = $('waifuStage');
    canvas = $('waifuCanvas');
    poppedNote = $('waifuPoppedNote');
    popoutBtn = $('waifuPopoutBtn');
    animSelect = $('waifuAnimSelect');
    animBtn = $('waifuAnimBtn');
    if (!toggle) return; // Sektion nicht vorhanden -> nichts tun

    toggle.addEventListener('change', onToggle);
    if (popoutBtn) popoutBtn.addEventListener('click', onPopout);
    if (animBtn) animBtn.addEventListener('click', onAnimToggle);
    if (animSelect) animSelect.addEventListener('change', function () { if (animTest) stopAnimTest(); });

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
    hookAgentBusy();
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
        stopAnimTest();
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
      fillAnimList();
      startTalkDriver();
    } finally {
      loading = false;
    }
  }

  /* ---- Anim-Tester: alle Animationen manuell ausloesen (ohne LLM) --------
     Emotes laufen in Schleife (One-Shots werden nach jedem Durchlauf neu
     getriggert), Koerpersprache-Zustaende werden gehalten. Solange ein Test
     laeuft, blenden alle automatischen Hooks (Mic/Busy/TalkDriver/Emote-
     Queue) ihre setAvatar-Aufrufe aus, damit nichts dazwischenfunkt. */
  var TEST_STATES = ['listening', 'speaking', 'thinking'];

  function tr(key, fb) {
    try {
      if (typeof t === 'function') {
        var s = t(key);
        if (s && s !== key) return s;
      }
    } catch (_) {}
    return fb;
  }

  function fillAnimList() {
    if (!animSelect || animSelect.options.length) return;
    var addGroup = function (label, kind, names) {
      var g = document.createElement('optgroup');
      g.label = label;
      names.forEach(function (n) {
        var o = document.createElement('option');
        o.value = kind + ':' + n;
        o.textContent = n;
        g.appendChild(o);
      });
      animSelect.appendChild(g);
    };
    var emotes = (waifu && waifu.emoteNames) ? waifu.emoteNames() : [];
    addGroup(tr('waifu.anim_emotes', 'Emotes'), 'emote', emotes);
    addGroup(tr('waifu.anim_states', 'States'), 'state', TEST_STATES);
  }

  function onAnimToggle() {
    try {
      if (animTest) { stopAnimTest(); return; }
      if (!waifu || !animSelect || !animSelect.value) return;
      var sep = animSelect.value.indexOf(':');
      var kind = animSelect.value.slice(0, sep), name = animSelect.value.slice(sep + 1);
      animTest = { kind: kind, name: name, timer: null };
      if (kind === 'state') {
        setAvatar(name);
      } else {
        waifu.emote(name);
        // Schleife: neu ausloesen, sobald der vorige Durchlauf (inkl.
        // VRMA-Clip-Ausblenden) wirklich fertig ist.
        animTest.timer = setInterval(function () {
          try { if (waifu && !waifu.emoteActive()) waifu.emote(name); } catch (_) {}
        }, 300);
      }
      if (animBtn) { animBtn.textContent = '⏹'; animBtn.title = tr('waifu.anim_stop', 'Stop'); }
    } catch (e) {
      console.warn('[waifu-ui] anim test failed:', e);
      stopAnimTest();
    }
  }

  function stopAnimTest() {
    var wasState = !!(animTest && animTest.kind === 'state');
    if (animTest && animTest.timer) clearInterval(animTest.timer);
    animTest = null;
    if (waifu) {
      try { if (waifu.stopEmote) waifu.stopEmote(); } catch (_) {}
      if (wasState) setAvatar('idle');   // ab hier uebernimmt wieder die Automatik
    }
    if (animBtn) { animBtn.textContent = '▶'; animBtn.title = tr('waifu.anim_play', 'Play'); }
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

  // Eigener Turn-Zustand, unabhaengig von der Mic-Zustandsmaschine: die feuert
  // nach turn.commit noch asynchrone 'listening'-Events (STT-done-Timeouts) und
  // wuerde 'thinking' sofort wieder wegkippen. agentThinking wird deshalb am
  // ECHTEN Turn-Lebenszyklus gefuehrt: an bei setAgentBusy(true) (= reply.start,
  // die "denkt nach"-Bubble), aus beim ersten echten Audio (talkDriver) oder
  // wenn der Turn ohne Audio endet (reply.silent/error, Abbruch, Barge-in).
  var agentThinking = false;
  var avatarMode = 'idle';   // zuletzt gesetzter Avatar-State (waifu hat keinen Getter)

  function setAvatar(s) {
    avatarMode = s;
    if (waifu) waifu.setState(s);
  }

  function applyState(state) {
    if (!waifu) return;
    if (animTest) return;    // manueller Anim-Test hat Vorrang
    // Plauder-States -> Avatar (nur Koerpersprache; der Mund laeuft im
    // talkDriver direkt am echten Playback-Zustand, s.u.)
    if (state === 'playing') { setAvatar('speaking'); return; }
    if (state === 'thinking') { agentThinking = true; setAvatar('thinking'); return; }
    if (agentThinking) { setAvatar('thinking'); return; }   // sticky bis Audio/Turn-Ende
    switch (state) {
      case 'speaking':         // NUTZER spricht gerade
      case 'collecting':       // Nutzereingabe laeuft noch (Partial/Debounce)
      case 'transcribing':     // STT verarbeitet die Nutzereingabe
      case 'transcribing_slow':
        setAvatar('listening');
        break;
      default:                 // listening / idle / loading / recording / error
        setAvatar('idle');
    }
  }

  // setAgentBusy(true) faellt exakt mit reply.start / "Joy denkt nach" zusammen;
  // false kommt nur auf Nicht-Audio-Enden (silent/error/discard/abort).
  function hookAgentBusy() {
    if (typeof window.setAgentBusy !== 'function') {
      return setTimeout(hookAgentBusy, 300);
    }
    if (window.__waifuBusyHooked) return;
    window.__waifuBusyHooked = true;
    var orig = window.setAgentBusy;
    window.setAgentBusy = function (v) {
      try {
        if (v) {
          agentThinking = true;
          if (waifu && !animTest) setAvatar('thinking');
        } else {
          agentThinking = false;
          if (waifu && !animTest && avatarMode === 'thinking') setAvatar('idle');
        }
      } catch (_) {}
      return orig.apply(this, arguments);
    };
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
      // Anim-Test 'speaking': Fake-Mund, damit der Zustand komplett zu sehen ist
      var testTalk = !!(animTest && animTest.kind === 'state' && animTest.name === 'speaking');
      if (playing || testTalk) {
        if (!animTest) {
          // TTS laeuft wirklich -> Nachdenken ist vorbei, Sprech-Koerpersprache.
          agentThinking = false;
          if (avatarMode !== 'speaking') setAvatar('speaking');
        }
        var t = (now - t0) / 1000;
        var v = Math.abs(Math.sin(t * 9) * 0.5 + Math.sin(t * 14) * 0.3);
        waifu.setMouth(Math.min(1, v));
      } else {
        if (!animTest && avatarMode === 'speaking') setAvatar('idle');
        waifu.setMouth(0);
      }
      pumpEmotes(playing, now);
      scheduleTalkPulse(step);
    };
    scheduleTalkPulse(step);
  }

  // requestAnimationFrame pausiert/drosselt in Hintergrund-Tabs (siehe waifu.js
  // scheduleFrame) -> ohne Fallback wuerde Lip-Sync/Emote-Pumping komplett
  // stehen bleiben, sobald der Tab in den Hintergrund geht.
  function scheduleTalkPulse(step) {
    if (typeof document !== 'undefined' && document.hidden) {
      speakingPulseIsTimeout = true;
      speakingPulse = setTimeout(function () { step(performance.now()); }, 200);
    } else {
      speakingPulseIsTimeout = false;
      speakingPulse = requestAnimationFrame(step);
    }
  }

  function stopTalkDriver() {
    if (speakingPulse) {
      if (speakingPulseIsTimeout) clearTimeout(speakingPulse); else cancelAnimationFrame(speakingPulse);
    }
    speakingPulse = null;
    if (waifu) waifu.setMouth(0);
  }

  /* ---- Nonverbal-Tags -> Emotes ----------------------------------------- */
  // Tags stehen im Reply-Text ([laughter], [sigh], *lacht*, ...). Der Text
  // laeuft der Stimme voraus, deshalb werden erkannte Emotes gepuffert und
  // erst abgespielt, wenn wirklich Audio laeuft (talkDriver zieht sie ab).
  var TAG_EMOTES = [
    [/\[laugh(?:s|ter)?\]|\*lacht\*|\b(?:ha){2,}h?\b|\b(?:he){2,}h?\b|\b(?:hi){2,}h?\b/i, 'lachen'],
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
    if (animTest) { emoteQueue.length = 0; return; }   // Test laeuft -> nichts dazwischenfunken
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
          // Fallback: erstes Delta dieses Turns => LLM arbeitet. Normalerweise
          // steht agentThinking schon seit reply.start (hookAgentBusy); das
          // hier greift nur, falls der Busy-Hook nicht installiert werden konnte.
          if (turnId !== thinkingTurn) {
            thinkingTurn = turnId;
            if (!animTest && avatarMode !== 'speaking') { agentThinking = true; setAvatar('thinking'); }
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
