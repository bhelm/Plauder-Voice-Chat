#!/usr/bin/env bash
# Plauder Voice-Chat mit Joy (Hermes-Backend via API-Server)
# Ein Aufruf genuegt: laedt Joy-Config inkl. VTuber-Avatar und startet den Server.
set -euo pipefail
cd "$(dirname "$0")"

# faster-whisper/ctranslate2 needs CUDA 12 libs from the pip nvidia packages
NV=$(.venv/bin/python -c "import nvidia, pathlib; print(pathlib.Path(list(nvidia.__path__)[0]))")
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Joy-Umgebung laden (.env.joy enthaelt u.a. WAIFU_MODE=1 -> Avatar beim Start an)
if [ ".env.joy" != ".env" ]; then
  set -a; source .env.joy; set +a
fi

# Sicherer Default: Avatar an, falls .env.joy es mal nicht setzt.
export WAIFU_MODE="${WAIFU_MODE:-1}"

echo "▶ Plauder startet — Joy${WAIFU_MODE:+ · VTuber-Avatar: $([ "$WAIFU_MODE" = "1" ] && echo AN || echo aus)}"
exec .venv/bin/python server.py
