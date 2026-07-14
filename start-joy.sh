#!/usr/bin/env bash
# Plauder Voice-Chat mit Joy (Hermes-Backend via API-Server)
set -euo pipefail
cd "$(dirname "$0")"
# faster-whisper/ctranslate2 needs CUDA 12 libs from the pip nvidia packages
NV=$(.venv/bin/python -c "import nvidia, pathlib; print(pathlib.Path(list(nvidia.__path__)[0]))")
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ ".env.joy" != ".env" ]; then
  set -a; source .env.joy; set +a
fi
exec .venv/bin/python server.py
