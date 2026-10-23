#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
source "$PWD/scripts/kira_runtime.sh"
PYTHON="$(kira_runtime_python)"

if [[ ! -x "$PYTHON" ]]; then
  echo "Run INSTALL_MACOS.command first, or create .venv and install requirements.txt."
  exit 1
fi

exec "$PYTHON" scripts/download_models.py "$@"
