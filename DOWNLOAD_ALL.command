#!/bin/zsh
# Complete first-run setup for GitHub clones and ZIP downloads.
set -euo pipefail
cd "${0:A:h}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: ./DOWNLOAD_ALL.command [--models-only]"
  echo "Install Kira, Live, voice and browser dependencies, then download all app models."
  echo "Requires Apple Silicon macOS, Python 3.11/3.12, Node.js 20+, npm and Git."
  exit 0
fi
if (( $# > 1 )) || [[ $# == 1 && "$1" != "--models-only" ]]; then
  echo "Unknown argument. Run ./DOWNLOAD_ALL.command --help" >&2
  exit 2
fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "Kira requires macOS on Apple Silicon (M1 or newer)." >&2
  exit 1
fi

source "$PWD/scripts/kira_runtime.sh"
PYTHON="$(kira_runtime_python)"
if [[ "${1:-}" != "--models-only" ]]; then
  for dependency in node npm git; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
      echo "Install $dependency first (Node.js 20+ includes npm), then retry." >&2
      exit 1
    fi
  done
  if [[ "$(node -p 'Number(process.versions.node.split(".")[0])')" -lt 20 ]]; then
    echo "Install Node.js 20 or newer, then retry." >&2
    exit 1
  fi
  if [[ ! -x "$PYTHON" ]]; then
    BOOTSTRAP=""
    for candidate in python3.12 python3.11 python3; do
      if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] not in ((3, 11), (3, 12)))'; then
        BOOTSTRAP="$(command -v "$candidate")"
        break
      fi
    done
    if [[ -z "$BOOTSTRAP" ]]; then
      echo "Install Python 3.11 or 3.12, then retry." >&2
      exit 1
    fi
    PYTHON="${KIRA_PYTHON:-$HOME/.local/share/kira-superapp/runtime/bin/python}"
    "$BOOTSTRAP" -m venv "${PYTHON:h:h}"
  fi
  "$PYTHON" -c 'import sys; sys.exit(sys.version_info[:2] not in ((3, 11), (3, 12)))' || {
    echo "The Kira runtime must use Python 3.11 or 3.12." >&2
    exit 1
  }
  echo "[1/4] Installing Python dependencies"
  "$PYTHON" -m ensurepip --upgrade
  "$PYTHON" -m pip install --upgrade pip wheel
  "$PYTHON" -m pip install -r requirements-all.txt
  "$PYTHON" -m pip check
  echo "[2/4] Installing Diagonal Browser and its media tools"
  (cd diagonal_browser && npm ci --include=dev --no-audit --no-fund && npm run setup)
  # The browser discovers its own DDG environment at this path.
  "$PYTHON" -m venv diagonal_browser/.venv
  diagonal_browser/.venv/bin/python -m pip install ddgs==9.16.0
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Runtime missing. Run ./DOWNLOAD_ALL.command without --models-only first." >&2
  exit 1
fi
echo "[3/4] Downloading app models (reruns reuse cached files)"
"$PYTHON" scripts/download_models.py --all
echo "[4/4] Verifying model discovery"
"$PYTHON" scripts/verify_install.py
echo "Setup complete. Launch Kira with ./superapp"
echo "Diagonal Browser also requires Google Chrome (or DIAGONAL_CHROME_BIN)."
