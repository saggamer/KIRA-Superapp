# KIRA Superapp for Linux — preview

This source distribution includes the KIRA desktop UI and the current Diagonal
Browser: compact two-row toolbar, image search, shared local research history,
closed-tab restoration, page inspection, screenshots, recordings and Ghostery
ad blocking with the bundled offline filter snapshot.

Linux chat and workflow generation use a **separately running OpenAI-compatible
chat-completions server**, configured explicitly by you. No model weights are
included or automatically converted. The Orchestrator selection connects to the
model named in `KIRA_MODEL_NAME`; selecting it does not mean that the server is
running KIRA's original Orchestrator weights. You can serve a compatible model
locally on CPU/GPU or use a hosted HTTPS endpoint. Prompts and tool observations
are sent to that configured server. Local histories and profiles stay on disk.

Native KIRA Live, MLX Horizon image generation, custom Orchestrator v1.5 adapters,
Touch ID and macOS app/AppleScript/Seatbelt integrations are unavailable. Other
macOS-specific desktop actions may report an error. Existing permission prompts
remain active; this preview does not provide the macOS Seatbelt sandbox on Linux.
Portable document, research and coding features depend on the configured model
and installed external tools. This is an early preview, not full macOS parity.

## Install

Target: Ubuntu 24.04 x86_64 with a graphical desktop, Python 3.11/3.12 and Node.js
20+ with npm. Other distributions and ARM64 have not been validated.
Install system libraries (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install python3-venv python3-dev build-essential portaudio19-dev \
  libportaudio2 libasound2t64 libnss3 libatk-bridge2.0-0 libgtk-3-0t64 \
  libgbm1 libxss1 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-image0 libxcb-render-util0 libxcb-xinerama0 libxkbcommon-x11-0 libegl1 libopengl0 \
  libpulse0 ffmpeg xdg-utils gnome-keyring
```

Install Node.js 20+ and npm using your distribution's supported method, then:

```bash
tar -xzf kira-superapp-linux-preview-2026.11.05.tar.gz
cd kira-superapp-linux-preview-2026.11.05
bash INSTALL_LINUX.sh
```

The installer creates `~/.local/share/kira-superapp/runtime`, installs portable
Python dependencies, Electron, and Playwright Chromium/FFmpeg. It does not use
sudo or install system libraries. The extracted app directory must be writable
because the current app stores its local working data beside the source.
If Playwright reports missing browser libraries, run
`cd diagonal_browser && npx playwright install-deps chromium` and review the
system package changes it requests.

## Start

Start your model server separately, then set its exact model identifier:

```bash
export KIRA_MODEL_BASE_URL=http://127.0.0.1:8080/v1
export KIRA_MODEL_NAME=your-served-model
# If required: export KIRA_MODEL_API_KEY=...  (do not commit credentials)
bash RUN_LINUX.sh
```

Use the **Orchestrator V1** selection for this connection. A remote endpoint must
use HTTPS. The server must support `POST /v1/chat/completions`, nonstreaming text,
`max_tokens` and `temperature`. Selecting other native models reports their
unsupported status. Missing server configuration fails before opening the UI;
connection errors are reported when generating. API keys can be stored through
the desktop secret-service keyring; a locked or missing keyring reports an error
and never silently falls back to plaintext storage.

Open Diagonal with KIRA's browser button or `cd diagonal_browser && npm run desktop`.
The Linux launcher configures the shared Python search runtime. Standalone desktop
launches need `DIAGONAL_IMAGE_PYTHON` set to that runtime's Python and
`DIAGONAL_DDGS_BIN` to its `bin/ddgs`.
Model browser tools use Playwright's bundled Chromium by default. Set
`DIAGONAL_CHROME_BIN` to a Google Chrome executable if a login provider requires
Chrome. Embedded Electron login restrictions still apply.

## Validation and provenance

`python3 -m unittest discover -s tests -p test_linux_runtime.py` checks worker IPC,
model-server requests, errors and output sanitization against a local fixture.
`cd diagonal_browser && npm test` runs the browser suite.
On Linux under Xvfb, `npm run test:desktop` verifies the current desktop UI.
The release workflow installs dependencies, opens the Qt KIRA window, runs these
checks and publishes artifacts only after they pass. Fixture tests do not prove
quality or compatibility with every real model server, OAuth provider or GPU.

`RELEASE_MANIFEST.json` records a SHA-256 digest for every packaged source file.
Profiles, histories, credentials, generated outputs, local environments and model
weights are excluded. Third-party browser filter attribution is retained in
`diagonal_browser/filters/`.
