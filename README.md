# 🎙️ Plauder

Sprach-zu-Sprach-Chat im Browser: **Mikrofon → STT → LLM → TTS → Lautsprecher** —
mit Turn-Taking (Debounce/Coalescing), Barge-In, niedriger Latenz durch
End-to-End-Streaming, Wake-Word, Text-Eingabe und Bild-Uploads.

Der Server ist ein sauberes Python-Package (`plauder/`) mit **pluggable
Backends** für STT, TTS und LLM. Per `.env` lässt sich zwischen Cloud-APIs
(OpenAI, Fireworks) und lokalen Modellen (faster-whisper, OmniVoice, OpenClaw)
umschalten — ohne Codeänderung.

---

## Features

- 🎙️ **Voice-to-Voice** im Browser, keine App nötig (nur `static/index.html`).
- ⚡ **Streaming für niedrige Latenz** — LLM-Token werden satzweise sofort
  synthetisiert und als Audio-Chunks progressiv abgespielt; das Mikrofon streamt
  live mit und liefert Zwischen-Transkripte (siehe [Streaming](#streaming--latenz)).
- 🔔 **Wake-Word** — wählbarer Eingabe-Modus (neben VAD & Push-to-Talk): die KI
  reagiert nur auf „Antonia …", der Rest wird verworfen; mit Konversationsfenster
  für Folgefragen. Start-Default aus, in der UI umschaltbar.
- 🔀 **Pluggable Backends** — STT/TTS/LLM je unabhängig per `.env` wählbar,
  Cloud oder lokal/GPU, jede Kombination erlaubt.
- 🗣️ **Turn-Taking & Barge-In** — Debounce/Coalescing, Ins-Wort-Fallen stoppt
  die Wiedergabe sofort.
- 💬 Text-Eingabe und 🖼️ Bild-Uploads (multimodal) parallel zur Stimme.

---

## Getting Started

### Voraussetzungen

- **Python 3.11** und ein Mikrofon-fähiger Browser (Chrome/Edge/Firefox).
- Ein **OpenAI-API-Key** (STT/TTS im Cloud-Default) und ein **OpenAI-kompatibler
  LLM-Endpoint** (z. B. Fireworks, oder ein lokaler Server). Für den voll-lokalen
  Betrieb siehe [Lokal / GPU](#lokal--gpu).

### Schnellstart (Cloud-Default, keine GPU)

```bash
cp .env.example .env          # 1) Keys eintragen (mind. OPENAI_API_KEY + LLM_*)
./start.sh            # 2) legt venv an, installiert Deps, startet Server
```

Dann im Browser öffnen: **http://localhost:8319**, Mikrofon erlauben und sprechen.

Standardmäßig startest du im **VAD-Modus** (alles, was du sagst, wird gesendet).
Im Kasten **Eingabe-Modus** kannst du auf **Wake-Word** umschalten — dann beginnst
du mit **„Antonia, …"** (z. B. „Antonia, wie spät ist es?") und darfst direkt danach
~8 s ohne erneutes „Antonia" weiterreden. Soll die UI gleich im Wake-Modus starten:
`WAKE_WORD_ENABLED=1` in der `.env`.

`start.sh` ist idempotent (beliebig oft aufrufbar) und lauscht auf
`${HOST}:${PORT}` (Default `0.0.0.0:8319`).

### Minimal-`.env` (Cloud)

```bash
# STT + TTS über OpenAI
OPENAI_API_KEY=sk-...
# LLM über einen OpenAI-kompatiblen Endpoint (Fireworks, lokaler Server, …)
LLM_BACKEND=openai_compat
LLM_BASE_URL=https://api.fireworks.ai/inference/v1
LLM_API_KEY=fw_...
LLM_MODEL=accounts/fireworks/models/glm-5p2
```

Alle Optionen sind in [`.env.example`](.env.example) dokumentiert.

### Gesundheitscheck

```bash
curl http://localhost:8319/healthz   # 200 + aktive Backends
```

---

## Architektur

```
                         Browser (static/index.html)
                          │   ▲
   16kHz f32 PCM-Frames    │   │  PCM-Chunks (VCT2, gestreamt) / WAV (VCT1)
   (VAD live / Push-to-Talk)│  │  + JSON-Events
                          ▼   │
   ┌───────────────────────────────────────────────────────────────┐
   │                  plauder/server.py  (HTTP/WS-Layer)          │
   │   Routen: / , /healthz , /ws , /upload                          │
   │   ws_handler ─ Audio-Segmente/-Frames & Text                   │
   │       ├─► turn_state.TurnState   (Debounce + Coalescing)        │
   │       ├─► wake                   (Wake-Word-Gate)               │
   │       ├─► sanitizer              (Ghost-Filter, Merge, NO_REPLY)│
   │       ├─► audio                  (PCM/WAV/numpy, Satz-Splitter) │
   │       └─► session.ConversationManager (Verlauf pro Session-Key) │
   └───────────────────────────────────────────────────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
      ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
      │  STTBackend  │     │  LLMBackend  │     │  TTSBackend  │
      │ .transcribe  │     │ .chat        │     │ .synth       │
      │ .—           │     │ .chat_stream │     │ .synth_stream│
      ├──────────────┤     ├──────────────┤     ├──────────────┤
      │ openai_api   │     │ openai_compat│     │ openai_api   │   ← Cloud
      │ whisper_local│     │ openclaw     │     │ omnivoice_   │   ← lokal/GPU
      │ (faster-     │     │ (gateway)    │     │  local       │     (lazy)
      │  whisper)    │     │              │     │ (omnivoice)  │
      └──────────────┘     └──────────────┘     └──────────────┘
        STT_BACKEND          LLM_BACKEND          TTS_BACKEND
```

**Lazy Imports:** Schwere GPU-Deps (`faster_whisper`, `torch`, `omnivoice`) werden
ausschließlich in `load()` des jeweiligen Backends importiert — und nur, wenn es
aktiv ist. Im Cloud-Default wird nie GPU-Code geladen.

### Pipeline pro Turn

1. Browser sendet Sprach-Frames (VAD live / PTT) und/oder Text über den WebSocket.
2. `TurnState` sammelt Eingaben im Debounce-Fenster (`DEBOUNCE_MS`); neue Eingabe
   bricht laufende LLM-/TTS-Calls ab (Barge-In).
3. STT → Text; Whisper-Halluzinationen werden gefiltert.
4. **Wake-Word-Gate** (nur im Wake-Modus, per Connection): nur an die KI
   gerichtete Segmente lösen einen Turn aus.
5. `ConversationManager` hängt den Verlauf an und ruft das LLM
   (`chat_stream` im Streaming-Modus, sonst `chat`).
6. `sanitizer` entfernt Emojis/Markdown/Links; TTS synthetisiert satzweise.
7. Audio geht als turn-id-getaggte Frames an den Browser.

---

## Streaming & Latenz

Standardmäßig (`STREAMING=1`) ist der latenzkritische Pfad end-to-end gestreamt:

| Stufe | Was |
|------|-----|
| **A1** | LLM-Token werden gestreamt (SSE), Text erscheint live (`reply.delta`). |
| **A2** | Sobald ein Satz fertig ist, wird er sofort synthetisiert und als PCM-Chunks (`VCT2`) progressiv abgespielt — Satz 1 spielt, während Satz 2 generiert wird. |
| **B1** | Der Browser streamt Mikrofon-Frames live mit (`segment.stream.*`), statt am Ende einen Blob zu schicken. |
| **B2** | Der Server transkribiert den anwachsenden Puffer gedrosselt → Live-Zwischen-Transkripte (`transcript.partial`). |

`STREAMING=0` schaltet auf den klassischen Pfad zurück (erst komplett generieren,
dann ein WAV) — nützlich, falls ein LLM-Endpoint kein SSE-Streaming kann.
Stellschrauben: `TTS_CHUNK_MS`, sowie für B2 `STT_PARTIAL*` (Default an bei
`whisper_local`).

Die **Statistik-Karte** zeigt die gefühlte Antwortzeit: `audio.start` liefert
`e2eMs` (fertig gesprochen → erste Wiedergabe, inkl. der eingestellten
Debounce-Pause — getrennt ausgewiesen) sowie die „erste / gesamt"-Zeiten für
Agent und TTS, sodass sichtbar wird, dass die Wiedergabe lange vor Ende der
vollständigen Synthese startet.

---

## Wake-Word

Wake-Word ist ein **Eingabe-Modus** neben VAD und Push-to-Talk, wählbar im
Kasten **Eingabe-Modus** der UI (pro Browser-Verbindung). Im Wake-Modus lösen nur
Segmente, deren Transkript mit dem Wake-Word **beginnt** (Füllwörter wie „Hey/Ok"
davor erlaubt), einen Turn aus — alles andere wird verworfen. Fuzzy-Matching
toleriert Whisper-Verhörer („Antonja", „Anthonia", „An Tonia"). Nach einer Antwort
bleibt ein Konversationsfenster offen, sodass Folgefragen **ohne** erneutes
Wake-Word durchgehen. Getippte Eingaben umgehen das Gate immer.

`WAKE_WORD_ENABLED` ist nur der **Start-Default** (in welchem Modus die UI
hochfährt); umschalten geht jederzeit live in der UI. Die übrigen Variablen
konfigurieren das Matching:

| Variable | Default | Bedeutung |
|---|---|---|
| `WAKE_WORD_ENABLED` | `0` | Start-Default: `1` = UI startet im Wake-Modus |
| `WAKE_WORD` | = `AGENT_NAME` | Wake-Wort (leer = Agent-Name klein) |
| `WAKE_WORD_WINDOW_S` | `8` | Folgefragen-Fenster nach einer Antwort (s) |
| `WAKE_WORD_FUZZY` | `1` | Verhörer tolerieren |
| `WAKE_WORD_ANYWHERE` | `0` | `1` = Wake-Wort irgendwo statt nur am Anfang |
| `WAKE_WORD_RATIO` | `0.78` | Fuzzy-Schwelle (höher = strenger) |

---

## Backends umschalten

Drei unabhängige Schalter in der `.env`:

| Variable      | Werte                          | Default         |
|---------------|--------------------------------|-----------------|
| `STT_BACKEND` | `openai` · `whisper_local`     | `openai`        |
| `TTS_BACKEND` | `openai` · `omnivoice_local`   | `openai`        |
| `LLM_BACKEND` | `openai_compat` · `openclaw`   | `openai_compat` |

`cfg.validate()` prüft beim Start nur das jeweils *aktive* Backend (Keys/Pflicht-
felder). Fehlt eine lokale Dependency, liefert `load()` eine klare Fehlermeldung
statt eines Importfehlers.

### Lokal / GPU

```bash
pip install faster-whisper          # STT_BACKEND=whisper_local
pip install omnivoice torch         # TTS_BACKEND=omnivoice_local (s. k2-fsa/OmniVoice)
```

`.env` für lokales Whisper (GPU):

```bash
STT_BACKEND=whisper_local
WHISPER_DEVICE=cuda
WHISPER_MODEL=large-v3-turbo
WHISPER_LOCAL_FILES_ONLY=1
```

`faster-whisper` läuft auch auf der **CPU** (`WHISPER_DEVICE=cpu`, kleines Modell
wie `base`, `WHISPER_LOCAL_FILES_ONLY=0` zum Nachladen) — gut zum Testen ohne GPU.

---

## Tests

```bash
.venv/bin/python -m pytest -q                 # voll (alle Backends gemockt, keine API/GPU)
.venv/bin/python -m pytest tests/test_wake.py # ein Modul
```

Pro Modul eigene Tests; alle Backends werden gemockt — keine echten API-Calls,
keine GPU. Lazy-Imports und alle Backend-Kombinationen werden abgedeckt.

---

## Projektstruktur

```
plauder/
├── config.py              # .env-Loading, Config-Dataclass, Validierung
├── server.py              # aiohttp App, Routen, WS-Handler, Turn-/Streaming-Orchestrierung
├── audio.py               # PCM/WAV/numpy, Frame-Formate (VCT1/VCT2), Satz-Splitter
├── turn_state.py          # Debounce + Coalescing, VAD-Parameter
├── sanitizer.py           # Emoji/Markdown-Stripping, Ghost-Filter, NO_REPLY
├── wake.py                # Wake-Word-Matching (STT-Prefix, fuzzy)
├── session.py             # ConversationManager (Verlauf pro Session-Key)
├── telegram_bridge.py     # optionale Telegram-Spiegelung (Legacy, Default aus)
└── backends/
    ├── stt/{base,openai_api,whisper_local}.py
    ├── tts/{base,openai_api,omnivoice_local}.py
    └── llm/{base,openai_compat,openclaw}.py
server.py                  # Entrypoint-Shim → plauder.server.run()
static/index.html          # kompletter Browser-Client (Audio, WS, UI)
```

> **Secrets & Pfade:** API-Keys und maschinenspezifische Pfade gehören
> ausschließlich in die `.env` (steht in `.gitignore`), niemals in den Quellcode.
> Eine bestehende Minimal-`.env` (nur `OPENAI_API_KEY` + `FIREWORKS_API_KEY`)
> funktioniert dank Legacy-Fallback-Ketten unverändert weiter.

---

## Lizenz

Copyright (C) 2026 Robert Sachse / Bernd Helm

Dieses Programm ist freie Software: Du kannst es unter den Bedingungen der
**GNU General Public License v3** (oder einer späteren Version), wie von der
Free Software Foundation veröffentlicht, weitergeben und/oder modifizieren.
Es wird in der Hoffnung verteilt, dass es nützlich ist, jedoch **ohne jede
Gewährleistung**. Siehe [`LICENSE`](LICENSE) für den vollständigen Text.
