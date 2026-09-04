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

The installer creates `.venv`, installs dependencies, downloads Orchestrator V1
into `models/orchestrator_v1_fused`, downloads the Apple-optimized Whisper model,
and verifies model discovery.

Launch KIRA:

```bash
./superapp
```

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
Face requires authentication or license acceptance, run `.venv/bin/hf auth login`
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

Run the focused regression suite:

```bash
.venv/bin/python -m unittest -q \
  tests.test_stabilization \
  tests.test_model_storage_selection \
  tests.test_orchestrator_runtime_stability
```

See [HARNESS_PROVENANCE.md](HARNESS_PROVENANCE.md) for the optional Codex and
DeepSeek harness integrations and their upstream licenses.

## Model And License Notes

Orchestrator V1 is downloaded separately from Hugging Face. Review its model card
and applicable Gemma terms before use. Third-party harness components retain their
upstream license files. KIRA Superapp source is provided under the Apache License 2.0.
