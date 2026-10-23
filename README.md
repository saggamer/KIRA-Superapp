# KIRA Superapp

KIRA Superapp is a local macOS agent runtime built around
[Orchestrator V1](https://huggingface.co/saggamer/Orchestrator_V1). It combines chat,
tool routing, web research, document and slide generation, coding workflows, local
speech-to-text, scheduling, plugins, and MCP connector management behind explicit
permission gates.

## Requirements

- macOS on Apple Silicon (M1 or newer)
- Python 3.11 or 3.12
- At least 16 GB unified memory; 24 GB is recommended
- Approximately 12 GB of free disk space for the environment and models
- Internet access for first-time installation

The current model runtime uses MLX and therefore does not support Intel Macs.

## Install

Clone the repository and run the installer:

```bash
git clone https://github.com/saggamer/KIRA-Superapp.git
cd KIRA-Superapp
chmod +x INSTALL_MACOS.command DOWNLOAD_MODELS.command superapp
./INSTALL_MACOS.command
```

The installer creates `~/.local/share/kira-superapp/runtime`, installs dependencies, downloads Orchestrator V1
into `models/orchestrator_v1_fused`, downloads the Apple-optimized Whisper model,
and verifies model discovery.

Launch KIRA:

```bash
./superapp
```

KIRA Live 1 is available in the model selector for chat, and the **KIRA Live**
button starts its native spoken conversation. Orchestrator V1 remains a separate
choice for computer workflows. The old `--kira-live-experimental` launch flag
remains as a compatibility alias that selects KIRA by default; it is not needed.

Live loads one revision-pinned logical checkpoint bundle: Listener, Thinker,
Talker, codec, and accepted KIRA additions. These are coordinated weights,
not a monolithic or jointly trained tensor file. The app releases its idle
text-model worker before loading Live to avoid duplicate model residency.
Repeated text inference keeps a bounded MLX cache warm. Audit the actual
deployed parameter count with `python scripts/audit_kira_live_checkpoint.py`.

## Download Models Separately

After dependencies are installed:

```bash
./DOWNLOAD_MODELS.command
```

Download only one model:

```bash
./DOWNLOAD_MODELS.command --orchestrator-only
./DOWNLOAD_MODELS.command --whisper-only
```

The defaults are:

- `saggamer/Orchestrator_V1`
- `mlx-community/whisper-small.en-mlx`

Override them with `KIRA_ORCHESTRATOR_REPO` or `KIRA_WHISPER_REPO`. If Hugging
Face requires authentication or license acceptance, run `~/.local/share/kira-superapp/runtime/bin/hf auth login`
and retry.

## Existing Orchestrator Model

KIRA automatically finds a complete local folder named any of the following:

- `orchestrator_v1_fused`
- `orchestrator v1 fused`
- `Orchestrator V1 fused`
- `Orchestrator_V1`

It checks the app folder, `models`, Desktop, Downloads, Documents, and LM Studio
model directories. You can always provide an exact path:

```bash
export KIRA_ORCHESTRATOR_PATH="/absolute/path/to/orchestrator_v1_fused"
./superapp
```

Cloud-only iCloud placeholders are rejected so model loading cannot silently hang.
Download the folder locally in Finder before launching KIRA.

## API Keys

API keys are optional and are not included in this repository. Keys entered through
KIRA are stored in the user's macOS Keychain. Provider settings, usage records,
chat history, memories, logs, generated artifacts, and local models are excluded
from Git.

Never commit `.env` files or exported credentials.

## Privacy And Output Safety

- Model reasoning is filtered before UI display and before chat persistence.
- Private reasoning contents are not written to release logs.
- Mutating file, command, IDE, MCP, and automation actions remain permission-gated.
- Generated artifacts and personal runtime data stay local and untracked.

## Development

Orchestrator v1.5 includes automatic browser research, page interaction, saved
history and MCP workflow skills. See [Orchestrator skills](docs/ORCHESTRATOR_V15_SKILLS.md)
for explicit invocation, loading limits, audit metadata and the skills-off comparison.

For interactive web research, screenshots, and browser recordings, see [Diagonal Browser](diagonal_browser/README.md). It adds a persistent Playwright MCP service and a Kira `[BROWSER_TOOL]` action backed by the Antigravity bridge's DDG search utility.

Run the focused regression suite:

```bash
~/.local/share/kira-superapp/runtime/bin/python -m unittest discover -s tests -q
```

All launchers honor `KIRA_PYTHON` and otherwise share the installed runtime,
with `.venv` and `.venv-kira-live` as local fallbacks. Run a focused subset with:

```bash
~/.local/share/kira-superapp/runtime/bin/python -m unittest -q \
  tests.test_stabilization \
  tests.test_model_storage_selection \
  tests.test_orchestrator_runtime_stability
```

See [docs/APP_ARCHITECTURE.md](docs/APP_ARCHITECTURE.md) for the runtime map,
request flow, safety boundaries, local-data layout, current validation status,
and the RED v4 model-development handoff.

The realtime model architecture is documented in
[docs/KIRA_LIVE_1.md](docs/KIRA_LIVE_1.md). Orchestrator v1.5 is an opt-in
research-preview profile; its weights are distributed separately from this
source repository.

See [HARNESS_PROVENANCE.md](HARNESS_PROVENANCE.md) for the optional Codex and
DeepSeek harness integrations and their upstream licenses.

## Model And License Notes

Orchestrator V1 is downloaded separately from Hugging Face. Review its model card
and applicable Gemma terms before use. Third-party harness components retain their
upstream license files. KIRA Superapp source is provided under the Apache License 2.0.
