# voice_chat — Hermes gateway platform plugin

Makes this browser voice chat a native Hermes messaging platform. The
gateway runs a localhost WebSocket bridge (`bridge.py`, default port 8321);
plauder connects to it with `LLM_BACKEND=hermes_gateway` and keeps the
connection open. That persistent link is what the stateless
`/v1/chat/completions` path could never provide: **asynchronous push
delivery** — `delegate_task(background=True)` results, cron deliveries and
`terminal notify_on_complete` land in the voice chat and are spoken.

```
Browser ⇄ plauder (STT/TTS, port 8319)
              ⇅ ws://127.0.0.1:8321/ws   (this bridge, token-authenticated)
          voice_chat adapter  ⇄  Hermes gateway (agent, tools, SessionDB)
```

- Inbound: transcripts become `MessageEvent`s with a per-turn voice-mode
  `channel_prompt` (terse, no markdown — override via
  `VOICE_CHAT_CHANNEL_PROMPT`, `-` disables). Image uploads travel as data
  URLs and are cached into real vision attachments (`media_urls`).
- Outbound: turn replies are tagged with the originating `turn_id`;
  everything else is a push and gets spoken. Frames sent while plauder is
  down are queued (bounded) and flushed on reconnect.
- Token streaming: with `display.platforms.voice_chat.streaming: true` in
  config.yaml the gateway's stream consumer drives `edit_message` →
  `agent.partial` frames; plauder yields suffix deltas so TTS starts on
  the first complete sentence (streaming cursor is stripped server-side).
- Out-of-process senders (`hermes send -t voice_chat`, cron running
  outside the gateway) use the bridge's token-authenticated
  `POST /push` (`standalone_sender_fn`). The home channel for bare
  targets comes from `VOICE_CHAT_HOME_CHANNEL` (default `default`).
- The adapter is text-only. STT, TTS, VAD, barge-in all stay in plauder.
- Wire protocol: header comment in `voice_chat/bridge.py`. Protocol tests:
  `tests/test_hermes_bridge.py` (runs in the plauder venv, no gateway needed).
- Known gap: plauder's "New Session" button rotates the legacy
  `.hermes_session_id` only — it does not reset the gateway session yet
  (send `/new` in the chat as a workaround).

## Install (once)

```bash
./install.sh    # symlinks voice_chat/ into ~/.hermes/plugins/platforms/
```

1. `~/.hermes/config.yaml`:
   - `plugins.enabled`: add `- platforms/voice_chat`
   - `platform_toolsets`: add a `voice_chat:` entry (copy the telegram list)
2. `~/.hermes/.env`:
   ```
   VOICE_CHAT_BRIDGE_TOKEN=<shared secret>
   VOICE_CHAT_ALLOW_ALL_USERS=true
   VOICE_CHAT_HOME_CHANNEL=default     # for bare deliver=voice_chat crons
   ```
3. Voice-chat `.env` (this repo):
   ```
   LLM_BACKEND=hermes_gateway
   HERMES_GATEWAY_WS_URL=ws://127.0.0.1:8321/ws
   HERMES_GATEWAY_TOKEN=<same secret>
   ```
4. `hermes gateway restart` — check `hermes gateway status` lists
   voice_chat and port 8321 is listening.
5. `systemctl restart voice-chat` — `journalctl -u voice-chat` should show
   `bridge connected` and `Gateway push delivery wired`.

Rollback: set `LLM_BACKEND=openai_compat` again and restart voice-chat —
the plugin can stay installed, it is inert without a connected client.

Since the plugin directory is a **symlink into this repo**, a `git pull`
updates the gateway-side code too; restart the gateway afterwards.
