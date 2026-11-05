#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "$(uname -s)" == Linux ]] || { echo 'Use ./superapp on macOS.'; exit 1; }
runtime="${KIRA_LINUX_RUNTIME:-$HOME/.local/share/kira-superapp/runtime}"
export KIRA_PYTHON="${KIRA_PYTHON:-$runtime/bin/python}"
export PATH="$runtime/bin:$PATH"
export PYWEBVIEW_GUI=qt
export QT_API=pyqt6
export DIAGONAL_IMAGE_PYTHON="${DIAGONAL_IMAGE_PYTHON:-$KIRA_PYTHON}"
export DIAGONAL_DDGS_BIN="${DIAGONAL_DDGS_BIN:-$runtime/bin/ddgs}"
"$KIRA_PYTHON" -c 'from linux_model_worker import server_config; server_config()'
exec ./superapp "$@"
