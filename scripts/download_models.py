#!/usr/bin/env python3
"""Download KIRA's public Orchestrator and local Whisper models."""

import argparse
import os
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_REPO = os.environ.get(
    "KIRA_ORCHESTRATOR_REPO", "saggamer/Orchestrator_V1"
).strip()
WHISPER_REPO = os.environ.get(
    "KIRA_WHISPER_REPO", "mlx-community/whisper-small.en-mlx"
).strip()


def download(repo_id, local_dir=None):
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "token": os.environ.get("HF_TOKEN") or None,
    }
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    return Path(snapshot_download(**kwargs)).resolve()


def validate_orchestrator(path):
    required = [path / "config.json", path / "tokenizer.json"]
    weights = list(path.glob("*.safetensors"))
    missing = [item.name for item in required if not item.is_file()]
    if missing or not weights:
        detail = ", ".join(missing) or "model weights"
        raise RuntimeError(f"Orchestrator download is incomplete: missing {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrator-only", action="store_true")
    parser.add_argument("--whisper-only", action="store_true")
    args = parser.parse_args()
    if args.orchestrator_only and args.whisper_only:
        parser.error("Choose at most one of --orchestrator-only and --whisper-only")

    try:
        if not args.whisper_only:
            target = APP_ROOT / "models" / "orchestrator_v1_fused"
            target.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {ORCHESTRATOR_REPO} to {target}", flush=True)
            path = download(ORCHESTRATOR_REPO, target)
            validate_orchestrator(path)
            print(f"Orchestrator V1 ready: {path}", flush=True)

        if not args.orchestrator_only:
            print(f"Downloading {WHISPER_REPO} to the Hugging Face cache", flush=True)
            path = download(WHISPER_REPO)
            print(f"Whisper ready: {path}", flush=True)
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        print(
            "If Hugging Face requests authentication or license acceptance, run "
            "`.venv/bin/hf auth login`, accept the model terms in your browser, and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
