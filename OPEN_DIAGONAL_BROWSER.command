#!/bin/zsh
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR/diagonal_browser"
if [[ ! -x node_modules/.bin/electron ]]; then
  npm ci
fi
if [[ ! -f node_modules/electron/path.txt ]]; then
  node node_modules/electron/install.js
fi
npm run desktop
