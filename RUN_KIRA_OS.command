#!/bin/zsh
set -u
cd "$(dirname "$0")"
PYTHON="$PWD/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "[KIRA SUPERAPP] Missing runtime: $PYTHON"
  echo "[KIRA SUPERAPP] Create .venv and install requirements before launching."
  exit 1
fi

if [[ ! -f "$PWD/interface.py" || ! -f "$PWD/ui.html" ]]; then
  echo "[KIRA SUPERAPP] interface.py or ui.html is missing from $PWD"
  exit 1
fi

exec "$PYTHON" "$PWD/interface.py"
