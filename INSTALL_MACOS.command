#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "KIRA Superapp requires macOS."
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This MLX release requires an Apple Silicon Mac (M1 or newer)."
  exit 1
fi

PYTHON=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "Install Python 3.11 or 3.12, then run this installer again."
  exit 1
fi

echo "[1/4] Creating the local Python environment"
"$PYTHON" -m venv .venv
echo "[2/4] Installing KIRA dependencies"
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r requirements.txt
echo "[3/4] Downloading Orchestrator V1 and Whisper"
.venv/bin/python scripts/download_models.py
echo "[4/4] Verifying the installation"
.venv/bin/python scripts/verify_install.py

echo
echo "KIRA Superapp is ready. Launch it with:"
echo "  ./superapp"
