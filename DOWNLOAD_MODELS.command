#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Run INSTALL_MACOS.command first, or create .venv and install requirements.txt."
  exit 1
fi

exec .venv/bin/python scripts/download_models.py "$@"
