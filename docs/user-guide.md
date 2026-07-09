# Using the browser client

The whole client is one hand-written file, [`static/index.html`](../static/index.html)
— no build step, no install. Open the app URL, allow the microphone, and talk.
It is fully internationalized (English + German, chosen by `APP_LANGUAGE`), so
every label below appears in the language the server was started with.

> **Secure context required.** Browsers only expose the microphone over
> `http://localhost` or HTTPS. A bare `http://<ip>:8319` from another machine
> gets **no** mic — put the app behind HTTPS or tunnel it (see
> [INSTALL §1](../INSTALL.md)).

---

## The main bar

| Control | What it does |
|---|---|
| 🎤 **Mic** | Turns the audio channel on/off. In push-to-talk mode, hold it to talk. |
| ⏹ **Stop** | Interrupts the current turn (transcription / reply / speech). Always works, regardless of gates. |
| **Level meter** | Live input level. |
| 🍔 **New session** | Discards the conversation and starts a fresh LLM session — **on all connected devices** (see [Cross-device sync](#cross-device-sync)). |
| ☰ **Settings** | Opens the settings drawer (below). |

The composer at the bottom takes **text** (Enter = send, Shift+Enter = newline)
and **images** — attach with 📎, drag & drop, or Ctrl+V. Text does **not** open
the voice flow; the mic keeps running in parallel, and a turn can mix voice +
text + images.

---

## Input modes

Three modes, switchable live in **Settings → Input mode** (persisted per
browser). The server start-default is set by `WAKE_WORD_ENABLED`.

- **VAD (auto)** — everything you say is captured and sent after a pause.
  Default.
- **Wake word** — only utterances that start with the wake word ("Antonia, …")
  trigger a turn; after a reply a short conversation window lets follow-ups
  through without repeating it. See [Wake word](#wake-word).
- **Push-to-talk** — hold the mic button to talk; the VAD is off.

The header status pill and the mic-status line reflect the active mode. While a
turn is being recognized the mic status shows **"✍️ Transcribing …"** (and a
"speech model is starting up" variant when a cold self-hosted STT is warming up),
so there's visible feedback instead of a silent gap before the reply.

---

## Settings drawer

### 🎚️ Detection & timing

| Control | Range / default | What it does |
|---|---|---|
| **Thinking-pause tolerance** | 0.2–5.0 s (1.2 s) | The pause after speech before a turn is dispatched. Drives both the browser VAD and the server debounce (`DEBOUNCE_MS`). The slider range comes from `DEBOUNCE_MS_MIN`/`DEBOUNCE_MS_MAX`. |
| **VAD sensitivity** | 0.10–0.95 (0.60) | Speech-detection threshold, tuned live against the probability bar below it (green level must cross the red line to count as speech). |
| **Live VAD bar** | — | Real-time speech probability, for tuning the slider. |

### 🔊 Audio & microphone

| Control | Range / default | What it does |
|---|---|---|
| **Microphone** | System default | Pick the input device; the choice is remembered. |
| **Speaking speed** | 0.7–3.0× (0.95×) | TTS playback rate. Sent to the server as `speed`, or applied locally if `TTS_OPENAI_LOCAL_SPEED=1`. |
| **Volume** | 0–100 % (80 %) | TTS playback volume. |
| **Cue volume** | 0–100 % (70 %, 0 = off) | Volume of the wake-word tones (rising "pling" on open, falling "plong" on close). Only sound in wake mode; releasing the slider previews the tone. |

Settings persist in `localStorage` and survive reloads.

---

## Per-message actions

Every message bubble has a ⋮ overflow menu ("More"):

- **Play / Pause audio** — play the reply's synthesized audio.
- **Replay from start** — restart from the beginning (also: click the speaker
  icon to play/pause, hold it to replay).
- **Download audio** — save the reply as a WAV.

Audio is cached client-side per turn (~40 most recent). After a page reload or
for a previous session the cache is empty — the menu then shows **"No audio
available"**.

---

## Wake word

In wake mode the mic button shows the state: a calm **green** dot = armed,
waiting for the wake word; a pulsing **red** aura = the conversation window is
open (keep talking, no wake word needed). Cues play on open/close if cue volume
> 0.

- Filler words in front are allowed ("Hey Antonia, …", "Ok Antonia …").
- Fuzzy matching tolerates Whisper mishearings ("Antonja", "An Tonia").
- Utterances **without** the wake word are shown as collapsed "ignored (no wake
  word) · N×" notices and do **not** interrupt a running reply.
- Saying a stop word ("stop", "halt", …) as a whole utterance inside an open
  window cancels the reply and closes the window.

Tuning lives in the `.env` — see [Configuration → Wake word](configuration.md#wake-word).

---

## Voice lock

Appears as a **Voice lock** card only when the server has the speaker backend
loaded (`SPEAKER_LOCK_ENABLED=1` + model). It gates the mic to **one enrolled
voice**.

- **🎙️ Learn my voice** — records ~6 s; do this 3–5 times with your everyday mic
  and distance for a solid profile.
- A badge (🔓 open / 🔒 locked) and a chip (**OPEN / LOCKED / PAUSED**) show the
  state; a toggle enables/disables the gate.
- The **strictness slider** (0.20–0.90) sets the acceptance threshold live; the
  "Last voice match" meter shows where your latest segment scored relative to the
  marker.
- **Forget my voice** clears the profile.

Foreign voices are dropped ("… — ignored (not your voice)"); mixed segments have
the foreign passage struck through in red. Full behavior + setup:
[README → Voice lock](../README.md#voice-lock).

---

## Voices (cloning)

Appears as a **Voices** card only when the server has voice cloning wired
(`TTS_CLONE_ENABLED=1` + the OmniVoice wrapper behind TTS). It clones and manages
the voice the assistant speaks in.

- **🎙️ Record a voice** — records ~15 s, then asks for a name and adds it to the
  library. Speak naturally in a quiet spot for the best clone. The sample is
  cleaned automatically: words cut off at the start/end of the recording window
  are removed, so talking slightly over the edges no longer ruins the clone
  (still best to pause briefly before and after speaking — if *everything* was
  cut off you're asked to re-record).
- **⬆️ Upload** — pick an audio file (any format). The spoken words are detected
  automatically; if that fails you're asked to type them. The same edge cleanup
  is applied when the file can be decoded.
- The **dropdown picks the active voice** (changing it switches immediately);
  the icon buttons next to it act on the selected voice: **🔊 preview** (hear a
  test sentence), **✏️ rename**, **🗑 delete**. The built-in default voice can't
  be renamed or deleted (buttons grey out).
- The **active** voice is used for **every** connected device and is remembered
  across restarts and new sessions — pick once, it sticks.

Full behavior + setup: [README → Voices](../README.md#voices-cloning).

---

## Latency / stats footer

The footer shows the perceived response time for the last turn:

- **Reply** — end-to-end from "you stopped speaking" to first playback, split
  into `pause` (your configured debounce) + `system` time.
- **STT** — speech-recognition time.
- **Agent** — LLM time (first word / total).
- **TTS** — synthesis time (first chunk / total).
- **Seg** — speech segments detected this session.

Because the pipeline streams, playback starts long before synthesis finishes —
the "first" numbers are what you actually wait for.

---

## Cross-device sync

All open browsers share **one** session. A committed user input and each final
reply are mirrored to the other connections, so a phone and a laptop stay in the
same conversation. The **New session** button (🍔) rotates the session, cancels
any in-flight reply on the peers, and clears their transcripts — with a confirm
dialog first ("The conversation history is discarded on all connected devices").
Peers see "New session started on another device".

## Reconnect

The client reconnects automatically on a dropped WebSocket (exponential backoff,
and immediately when the tab regains focus or the network returns). You get one
"🔌 Connection lost – reconnecting …" notice per drop; the header pill shows
`connecting… / connected / disconnected`.
