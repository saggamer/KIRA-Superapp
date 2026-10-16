#!/bin/zsh
set -euo pipefail

cd "${0:A:h}/diagonal_browser"

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Node.js 20+ and npm are required for Diagonal Browser."
  exit 1
fi
if [[ "$(node -p 'Number(process.versions.node.split(".")[0])')" -lt 20 ]]; then
  echo "Diagonal Browser requires Node.js 20 or newer."
  exit 1
fi
if [[ ! -x "${DIAGONAL_CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}" ]]; then
  echo "Google Chrome was not found. Set DIAGONAL_CHROME_BIN to its executable path."
  exit 1
fi
DEFAULT_DDGS_BIN="$HOME/Downloads/chatgpt_antigravity_bridge/bin/ddgs"
if [[ -n "${DIAGONAL_DDGS_BIN:-}" ]]; then
  DDGS_BIN="$DIAGONAL_DDGS_BIN"
elif [[ -x "$DEFAULT_DDGS_BIN" ]]; then
  DDGS_BIN="$DEFAULT_DDGS_BIN"
else
  DDGS_BIN="$(command -v ddgs || true)"
fi
if [[ -z "$DDGS_BIN" || ! -x "$DDGS_BIN" ]]; then
  echo "Antigravity DDG tool was not found. Set DIAGONAL_DDGS_BIN to its executable path."
  exit 1
fi

npm ci --no-audit --no-fund
npm run setup
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check ddgs==9.16.0
echo "Diagonal Browser is ready. Restart Kira to use BROWSER_TOOL."
