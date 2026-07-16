# Projekt Waifu — Plan

> Status: **Entwurf / Planung** (kein Code). Erstellt 2026-07-15.
> Ziel: Modulare Erweiterung des Plauder-VC um (1) einen steuerbaren **Screen-Vision-Loop**
> und (2) ein lokales **VTuber-Frontend** im Web-UI.
> **Ziel-Profil: Joy** (Single-Mode, Englisch, brit. Akzent) — passt zum Single-Chat-Gating.
> VTuber-Figur wird also für Joy gebaut, nicht Xena.
> Inspiration: `0Xiaohei0/LocalAIVtuber2` (Windows-only, keine Lizenz → **nur Konzepte, kein Code-Copy**).

---

## 0. Leitplanken (nicht verhandelbar)

- **Modularität:** Beide Features sind eigenständige, abschaltbare Module. Kein Eingriff in
  bestehende STT/TTS/Speaker-Gate-Pfade außer über klar definierte Hooks.
- **Haus-Mode = OFF:** Waifu-Features laufen **nur im Single-Chat** (Joy-artig).
  Bei `HOUSE_MODE=1` sind Vision-Loop und VTuber-Frontend hart deaktiviert.
  → Neuer Umbrella-Flag `WAIFU_MODE`, der bei `HOUSE_MODE=1` zwangs-OFF ist.
- **Linux-nativ:** Läuft unter WSL/Linux (kein `pyautogui`, kein Windows-`mss`-Pfad blind
  übernehmen). Screen-Capture + Audio unter WSL sind der bekannte Reibungspunkt → früh testen.
- **Kein Lizenz-Risiko:** LAV2 hat KEINE Lizenz. Wir übernehmen **Architektur-Ideen**,
  schreiben Code selbst. Kein Copy-Paste aus dem Repo.
- **VRAM-Budget:** RTX 3090 ist schon mit LM Studio (Gemma-12B) + Honcho belegt.
  Vision-Stack muss klein bleiben (OCR + kleines Caption-Modell, CPU-fähig).

---

## 1. Wie LAV2 es macht (Referenz-Analyse)

Aus `backend/services/Input/VisionInput.py`:

- **Capture:** `mss` grabt einen Monitor (`monitor_index`), → PIL-Image.
- **OCR:** `easyocr.Reader(languages).readtext(..., decoder='beamsearch')`
  mit `confidence_threshold` (0.5) und `ocr_scale_factor` (0.5 = halbe Auflösung für Speed).
- **Caption:** BLIP (`Salesforce/blip-image-captioning-base`) erzeugt Bildbeschreibung.
- **Fusion:** OCR-Text + Caption → als Kontext-String ans LLM.
- **Flags:** `skip_ocr`, `save_screenshot` schon vorhanden → sauber trennbar.

**Wichtig:** Es ist **kein Vision-LLM**, sondern OCR + kleines Caption-Modell.
Klein, günstig, portierbar. Für uns die richtige Basis.

---

## 2. Feature A — Screen-Vision-Loop (steuerbar)

### 2.1 Backend-Modul

Neues Paket: `plauder/vision/` (analog zu den bestehenden `backends/`):

```
plauder/vision/
  __init__.py
  capture.py      # Screen-Grab (Linux: mss über X11/Wayland ODER XDG-portal; WSL-Weg klären)
  ocr.py          # easyocr-Wrapper (lazy-load, offline-cache wie STT/TTS)
  caption.py      # BLIP-base ODER Alternative (siehe 2.4)
  loop.py         # Vision-Loop: Intervall, Trigger, Dedup, Kontext-Fusion
  config.py       # Vision-spezifische Settings (oder in plauder/config.py integriert)
```

### 2.2 Steuerbarkeit (die Kern-Anforderung)

Der Loop muss **zur Laufzeit steuerbar** sein, nicht nur per Env:

| Kontrolle | Verhalten |
|---|---|
| **On/Off** | Vision-Loop komplett an/aus |
| **Intervall** | Sekunden zwischen Captures (z.B. 2–30 s) — oder „on demand" |
| **Trigger-Modus** | `interval` (periodisch) / `manual` (nur auf Knopf) / `on_ask` (nur wenn User nach Screen fragt) |
| **Monitor** | Welcher Monitor / welche Region |
| **OCR an/aus** | Nur Caption, nur OCR, oder beides |
| **Confidence / Scale** | OCR-Tuning (Threshold, Scale-Factor) |

State liegt im Server, wird per WS/HTTP vom Frontend gesetzt (siehe 4).

### 2.3 Integration in den Chat-Flow (Hook-Punkte)

- **Config:** `plauder/config.py` — neuer Block `# --- Waifu / Vision ---` mit
  `waifu_mode`, `vision_enabled`, `vision_interval`, `vision_trigger`, `vision_monitor`,
  `vision_ocr`, `vision_confidence`. Umbrella-Logik: `waifu_mode = env_flag("WAIFU_MODE", False) and not house_mode`.
- **App-Init:** `plauder/app.py` — bei `cfg.waifu_mode` das Vision-Modul instanziieren
  (analog zum bestehenden `if cfg.house_mode:` Log-Block, Zeile ~127).
- **Kontext-Einspeisung:** Vision-Ergebnis (OCR+Caption) wird als System-/Kontext-Notiz
  in den LLM-Prompt gehängt — **vor** dem User-Turn, klar markiert
  (`[Bildschirm: …]`), damit das LLM weiß, was es „sieht".
- **Kein Eingriff** in Speaker-Gate/STT: Vision ist ein **paralleler** Input-Kanal.

### 2.4 Offene technische Fragen (vor Implementierung klären)

1. **Screen-Capture unter WSL:** `mss` braucht X-Server/Display. Läuft der VC unter WSLg
   (WSL2 GUI) oder headless? → ggf. Capture auf Windows-Host via kleinem Helfer, oder
   WSLg-`:0`. **Muss zuerst als Spike getestet werden.**
2. **Caption-Modell:** BLIP-base ist ok, aber englisch-lastig. Für Xena (DE) ggf.
   OCR (multilingual via easyocr `['de','en']`) wichtiger als Caption. Evtl. Caption
   optional lassen.
3. **VRAM:** BLIP-base ~1–2 GB. Prüfen ob CPU-Inference reicht (langsamer, aber spart 3090).
4. **Datenschutz:** Screen-Inhalte gehen an LLM. Bei DeepSeek-Cloud-Profil = Screen-Inhalt
   verlässt das Haus. → Vision-Loop bei Cloud-Provider **default OFF** + deutlicher Hinweis.

---

## 3. Feature B — Lokales VTuber-Frontend

### 3.1 Bestand

Frontend vorhanden: `static/index.html` + `static/js/{vct,mic,playback,i18n,markdown}.js`.
Rechter Settings-Drawer: `<aside id="settingsDrawer" class="drawer">` mit `.drawer-body`,
darin `<details class="acc" data-sec="...">`-Accordion-Sektionen. Scripts sind klassisch
(kein Build-Step), eingebunden mit `?v=__ASSET_VER__`. Vendor-Libs lokal in `static/vendor/`.
VTuber-Rendering wird ein **neues, isoliertes** JS-Modul (`static/js/waifu.js`),
das bestehende Pfade (mic/playback/vct) nicht anfasst.

### 3.1a UI-Verhalten (Kern-Anforderung, ENTSCHIEDEN)

Der Avatar ist ein **optionales, dreistufig steuerbares** Element:

1. **Optional / default AUS.** Toggle „VTuber-Avatar anzeigen" in der Waifu-Settings-Rubrik.
   Ohne Aktivierung wird nichts geladen (kein three.js, kein VRM) — Null-Overhead.
2. **Docked (eingebettet).** Beim Einschalten erscheint der Avatar-Canvas **rechts im
   Settings-Drawer**, als eigene `<details data-sec="waifu">`-Sektion. Standard-Ansicht.
3. **Pop-out (herausgelöst).** Ein **Pop-out-Icon** (⧉) am Avatar-Canvas löst ihn in ein
   **eigenes freistehendes Fenster** (`window.open`, Detached-Window). Der Docked-Canvas
   zeigt dann einen „im Pop-out"-Platzhalter; Schließen des Pop-outs dockt zurück.

**Technik Pop-out:** `window.open('', 'waifu', 'width=..,height=..')`; three.js-Renderer +
Canvas werden in das neue Document verschoben (nicht neu instanziiert → State/Modell bleibt).
`beforeunload` des Pop-outs → zurück-docken. Lip-Sync/State-Events laufen über `postMessage`
oder geteiltes JS-Objekt weiter (gleicher Origin, daher direkter Zugriff möglich).

**Isolation:** Alles in `waifu.js` gekapselt. Bricht der Avatar, läuft der VC normal weiter.

### 3.2 Rendering-Ansatz — ENTSCHIEDEN: VRM (3D) + three.js

**Renderer:** `three.js` + `@pixiv/three-vrm`, rein clientseitig im Browser (WebGL, offline).
Alle Assets in `static/` — keine externen CDNs, kein Server-Call, keine Cloud.

**Modell-Workflow (zweistufig):**
1. **Grundfigur in VRoid Studio** (lokal, kostenlos, Windows) — Gesicht/Körper/Frisur/
   Kleidung per Slider. Ergebnis: sauberes `.vrm` mit VOLLSTÄNDIGEM Rig
   (Humanoid-Bones, Expressions, Spring Bones). Das ist der **Golden Master**.
2. **Feintuning via Blender-MCP** — nur additiv/nicht-destruktiv: Materialien, Farben,
   Accessoires anfügen, Objekte togglen. KEINE Edits an Shape-Keys oder Bone-Mapping.

Live2D / Sprite-Swap = verworfen (VRM erlaubt lokales Selbst-Bauen via VRoid + volle Animation).

### 3.3 Sync mit dem Audio-Pipeline

**Referenz-Muster aus LAV2** (`frontend/src/components/vrm-3d-renderer.tsx`, 381 Z.,
React/Vite — nur Muster übernommen, KEIN Code-Copy, LAV2 hat keine Lizenz):

- **Loader:** `GLTFLoader` + `VRMLoaderPlugin` (+ `VRMAnimationLoaderPlugin` für `.vrma`).
  Nach Load: `VRMUtils.removeUnnecessaryVertices`, `vrm.springBoneManager?.reset()`.
- **Lip-Sync (der Kern):** LAV2 baut aus der `aa`-Expression eine `NumberKeyframeTrack`
  → `AnimationClip` → `clipAction`. Getrieben über einen globalen State
  `ttsLiveVolume` (Audio-Amplitude); `> 0.1` ⇒ „speaking" ⇒ `aa`-Action läuft.
  **Für uns:** in `playback.js` die TTS-Audio-Amplitude (WebAudio `AnalyserNode`,
  RMS pro Frame) exposen → `waifu.js` mappt sie auf `expressionManager.setValue('aa', rms)`.
  Simpler & direkter als LAV2s Clip-Ansatz — reicht für Mund-Auf/Zu.
- **Blink:** eigene `NumberKeyframeTrack` auf `blink`, Intervall ~2 s, als Loop-Action.
- **Idle:** `.vrma`-Idle-Animation via `AnimationMixer` (LAV2 lädt `idle.vrma`).
  **Für uns Phase 3:** simpler Idle (leichtes Atmen/Sway per Bone-Rotation) reicht;
  `.vrma`-Clips optional später.
- **LookAt:** `VRMLookAtQuaternionProxy` + Target an der Kamera → Augen folgen.
- **Update-Loop:** `mixer.update(dt)` + `vrm.update(dt)` pro Frame (Spring Bones + Expr.).

- **State:** idle / listening / speaking / thinking → aus vorhandenen VC-Events (Mic aktiv,
  TTS läuft) ableiten. Keine neue Server-Logik nötig, nur Frontend-Event-Subscription.
- **Optional:** Emotions-Tags aus LLM-Antwort → VRM-Expression (`happy/angry/sad`, Phase 2).

**Unterschied zu LAV2:** Sie nutzen React 19 + Vite + shadcn/ui (Build-Step). Unser Plauder
ist Vanilla-JS ohne Build-Step → wir binden `three` + `three-vrm` als vorgebaute
ES-Module/UMD lokal in `static/vendor/` ein und schreiben `waifu.js` als klassisches Script.
Kein React, kein Vite. Muster gleich, Umsetzung schlanker.

### 3.4 Animations-Erhaltung (KRITISCH — Kern-Anforderung)

VRM animiert über vier Datenschichten, die den Blender-Roundtrip überleben MÜSSEN:

| Schicht | Wofür | Risiko |
|---|---|---|
| **Humanoid-Bones** | Body-Anim, Idle, Gesten | Bricht bei Namens-/Hierarchie-Änderung |
| **Blendshapes/Expressions** | Lip-Sync (`aa/ih/ou/ee/oh`) + Emotion/Blink | **Höchstes Risiko** — Mesh-Edits killen Shape-Keys |
| **Spring Bones** | Physik Haare/Kleidung | VRM-spezifisch, nur via Addon erhalten |
| **LookAt** | Augen folgen Blick | Selten angefasst |

**Nicht-verhandelbare Regeln:**
- **VRM-Addon Pflicht:** `VRM Addon for Blender` (saturday06) MUSS installiert sein, sonst
  exportiert Blender totes glTF ohne VRM-Extensions → Spring Bones + Expressions weg.
- **Golden Master unantastbar:** Original-VRoid-VRM nie überschreiben. Jeder Blender-Lauf
  schreibt eine NEUE Datei.
- **Nur additive Edits** in Blender: Material/Farbe/Accessoire. Kein Sculpting am
  Gesichts-Mesh (zerstört Viseme-Shape-Keys).
- **Verifikations-Gate nach JEDEM Export:**
  1. Humanoid-Bone-Count vorher == nachher
  2. Alle Expressions vorhanden (`aa/ih/ou/ee/oh/blink/happy/angry/sad`)
  3. Spring-Bone-Chains noch da
  4. Testrender in three-vrm: Mund-Test (`aa`) + Blink + Idle → visuell grün
  Erst wenn grün, gilt das bearbeitete Modell als gut.

### 3.5 Blender-MCP-Setup (Voraussetzung fürs Feintuning)

- Blender läuft **nativ unter Windows** (nicht WSL), mit MCP-Addon + VRM-Addon.
- MCP-Server im Hermes-Profil registrieren; Verbindung WSL↔Windows-Blender klären.
- Was via MCP zuverlässig geht: Transform, Material/Farbe, Objekt-Toggle, Import/Export,
  Blendshape-Prüfung. Was NICHT: blindes Feinmodellieren/Sculpting.

---

## 4. Settings-Rubrik (rechte Seite)

Neue eigene Rubrik im Settings-Panel (`static/index.html` + `vct.js`), **nur sichtbar wenn
`WAIFU_MODE` aktiv & `HOUSE_MODE=0`**:

```
▸ Waifu
    [ ] VTuber-Avatar anzeigen
        Modell:      [ VRM-Modell ▾ ]
    ── Screen Vision ──────────────
    [ ] Vision-Loop aktiv
        Trigger:     ( ) Intervall  ( ) Manuell  ( ) Auf Nachfrage
        Intervall:   [  5 ] s
        Monitor:     [ Monitor 1 ▾ ]
        [ ] OCR-Text lesen
        [ ] Bild beschreiben (Caption)
        Confidence:  [====|----] 0.5
        [ Jetzt einen Screenshot ansehen ]   ← manueller Trigger
    ⚠ Bei Cloud-LLM verlässt der Bildschirminhalt das Gerät.
```

- Settings werden per bestehendem Settings-Kanal an den Server gepusht (gleicher Weg wie
  aktuelle VC-Settings — in `server.py` mitschauen wie das läuft).
- Live-Update ohne Neustart (Loop liest State bei jedem Tick).

---

## 5. Phasenplan

| Phase | Inhalt | Ergebnis |
|---|---|---|
| **0 — Spike** | WSL-Screen-Capture testen (`mss`/WSLg). Klären ob überhaupt sinnvoll headless. | Go/No-Go für Vision |
| **1 — Vision-Backend** | `plauder/vision/` mit capture+ocr+loop, Config-Flags, `WAIFU_MODE`-Gating. Kontext-Fusion in LLM-Prompt. Tests. | Vision-Loop headless nutzbar |
| **2 — Settings-UI** | Waifu-Rubrik rechts, Loop-Steuerung, manueller Trigger, Cloud-Warnung. | Vision voll steuerbar im UI |
| **3 — VTuber-MVP** | ✅ **ERLEDIGT** VRoid→Joy.vrm (Golden Master, Verifikations-Gate grün). three-vrm-Renderer `static/js/waifu.js` + Standalone-Test `static/waifu_test.html`. Idle-Atmen, Auto-Blink, Lip-Sync-API (`setMouth`), Expressions. Joy rendert 3D, frontal, Emotionen ok. | ✅ Avatar sichtbar & animierbar |
| **3b — Blender-Feintuning** | Blender-MCP + VRM-Addon-Setup, additive Edits (Farbe/Accessoire), Verifikations-Gate. | Angepasstes VRM, Rig intakt |
| **3c — UI-Integration** | ✅ **ERLEDIGT** Docked-Sektion im Settings-Drawer (`data-sec="waifu"`) + Toggle + Pop-out-Icon (⧉). Lip-Sync-Fallback an globales `setMicUi()`. Server-Flag `WAIFU_MODE` (config.py + server.py `__WAIFU_MODE__`-Platzhalter) → `start-joy.sh` startet Avatar per Default an. | ✅ Avatar im echten UI, optional + pop-out, per Skript startbar |
| **4 — Politur** | Caption-Modell, Emotions-States (Emotion+Sprechen-Zusammenspiel klären: `happy` überlagert `aa`), Modell-Auswahl-Dropdown, `.vrma`-Idle. | Feature-komplett |

Jede Phase: `WAIFU_MODE=0` lässt VC **exakt wie heute** laufen (Regressionstest).

---

## 6. Risiken / Watch-outs

- **WSL-Capture** ist das größte Fragezeichen → Phase 0 zuerst, sonst kein Vision-Feature.
- **VRAM-Konkurrenz** mit LM Studio auf der 3090 → Vision-Modelle klein/CPU halten.
- **Cloud-Datenschutz:** Screen → DeepSeek. Default-OFF bei Cloud + UI-Warnung (Pflicht).
- **Haus-Mode-Kollision:** Sub-Flag-Logik in `config.py` sauber testen — Waifu darf bei
  `HOUSE_MODE=1` unter keinen Umständen anspringen (eigener Regressionstest).
- **Frontend-Isolation:** VTuber-JS darf bestehende Mic/Playback-Pfade nicht brechen.

---

## 7. Nächster Schritt

Wenn du grünes Licht gibst: **Phase 0 (WSL-Screen-Capture-Spike)** — 1 kleines Skript,
das unter deiner WSL einen Screenshot zieht und OCR drüberlaufen lässt. Ergebnis
entscheidet, ob Feature A überhaupt Sinn ergibt oder ob wir Capture anders lösen müssen.
