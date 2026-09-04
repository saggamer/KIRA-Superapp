#!/bin/zsh
set -u
cd "$(dirname "$0")"
PYTHON="$PWD/.venv/bin/python"

echo "[KIRA Superapp] Launching..."
echo "[KIRA Superapp] Runtime: $PYTHON"

if [[ ! -x "$PYTHON" ]]; then
  echo "[KIRA Superapp] Missing runtime: $PYTHON"
  exit 1
fi

if [[ ! -f "$PWD/interface.py" || ! -f "$PWD/model_worker.py" || ! -f "$PWD/ui.html" ]]; then
  echo "[KIRA Superapp] interface.py, model_worker.py, or ui.html is missing from $PWD"
  exit 1
fi

exec "$PYTHON" "$PWD/interface.py"
