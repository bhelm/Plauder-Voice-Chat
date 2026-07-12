#!/usr/bin/env bash
# Install (symlink) the voice_chat gateway plugin into the Hermes user
# plugin directory. Idempotent. The symlink keeps the repo as the single
# source of truth — a git pull updates the plugin; restart the gateway
# afterwards (`hermes gateway restart`).
#
# Remaining manual steps (once), see hermes_plugin/README.md:
#   1. Add `platforms/voice_chat` to plugins.enabled in ~/.hermes/config.yaml
#   2. Set VOICE_CHAT_BRIDGE_TOKEN (+ optional VOICE_CHAT_*) in ~/.hermes/.env
#   3. hermes gateway restart
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/voice_chat"
DEST_DIR="${HERMES_HOME:-$HOME/.hermes}/plugins/platforms"
DEST="$DEST_DIR/voice_chat"

mkdir -p "$DEST_DIR"
if [ -L "$DEST" ]; then
    CURRENT="$(readlink -f "$DEST")"
    if [ "$CURRENT" = "$(readlink -f "$SRC")" ]; then
        echo "already installed: $DEST -> $CURRENT"
        exit 0
    fi
    echo "replacing stale symlink ($CURRENT)"
    rm "$DEST"
elif [ -e "$DEST" ]; then
    echo "ERROR: $DEST exists and is not a symlink — refusing to overwrite" >&2
    exit 1
fi

ln -s "$SRC" "$DEST"
echo "installed: $DEST -> $SRC"
