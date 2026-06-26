#!/usr/bin/env bash
# Plauder — Setup + Start. Idempotent: legt bei Bedarf ein venv an, installiert
# die Requirements und startet den Server. Beliebig oft aufrufbar.
#
# Die GESAMTE Konfiguration (Agent-Name, STT/TTS/LLM-Backends, Keys, Sprache,
# Stimme, Wake-Word …) kommt aus der .env — siehe .env.example. Der Server liest
# die .env selbst ein; dieses Skript setzt keine agent-spezifischen Variablen.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="$HERE/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# --- venv anlegen, falls nicht vorhanden ---
if [[ ! -x "$PY" ]]; then
  echo "🐍 Erstelle venv in $VENV …"
  python3 -m venv "$VENV"
fi

# --- Abhängigkeiten installieren ---
echo "📦 Installiere Requirements …"
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r "$HERE/requirements.txt"

echo "🎙️  Starte Plauder … (Konfiguration aus .env)"
exec "$PY" server.py
