#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "$(uname -s)" == Linux ]] || { echo 'Run this installer on Linux.'; exit 1; }
for tool in python3 node npm; do
  command -v "$tool" >/dev/null || { echo "Install $tool first; see docs/LINUX.md."; exit 1; }
done
python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Python 3.11 or 3.12 required"'
node -e 'if(Number(process.versions.node.split(".")[0])<20)process.exit(1)'
runtime="${KIRA_LINUX_RUNTIME:-$HOME/.local/share/kira-superapp/runtime}"
python3 -m venv "$runtime"
"$runtime/bin/python" -m pip install --upgrade pip wheel
"$runtime/bin/python" -m pip install -r requirements-linux.txt
"$runtime/bin/python" -m pip check
(cd diagonal_browser && npm ci --include=dev --no-audit --no-fund && npx playwright install chromium ffmpeg)
echo "Installed Linux preview. Configure KIRA_MODEL_BASE_URL and KIRA_MODEL_NAME, then run ./RUN_LINUX.sh."
