#!/bin/zsh
set -euo pipefail

APP_ROOT="${0:A:h}"
cd "$APP_ROOT"

if [[ ! -x "$APP_ROOT/.venv/bin/python" ]]; then
  echo "KIRA OS virtual environment was not found at $APP_ROOT/.venv"
  echo "Run the KIRA OS setup first, then try again."
  read -r "?Press Return to close..."
  exit 1
fi

echo "KIRA OS secure Vibe Coding test"
echo "Project: ${APP_ROOT:h}/kira_windsurf_demo"
echo "The Gemini key is requested invisibly and exists only for this process."
echo

"$APP_ROOT/.venv/bin/python" "$APP_ROOT/scripts/run_gemini_windsurf_build.py"

echo
echo "Finished."
read -r "?Press Return to close..."
