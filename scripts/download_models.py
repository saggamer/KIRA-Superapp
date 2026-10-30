#!/usr/bin/env python3
"""Download KIRA models; --all includes Live and the complete local voice stack."""

import argparse
import os
from pathlib import Path
import sys
import json
import shutil
import urllib.request


APP_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_REPO = os.environ.get(
    "KIRA_ORCHESTRATOR_REPO", "saggamer/Orchestrator_V1"
).strip()
WHISPER_REPO = os.environ.get(
    "KIRA_WHISPER_REPO", "mlx-community/whisper-small.en-mlx"
).strip()


def download(repo_id, local_dir=None, revision=None):
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "token": os.environ.get("HF_TOKEN") or None,
    }
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    if revision is not None:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs)).resolve()


def validate_orchestrator(path):
    required = [path / "config.json", path / "tokenizer.json"]
    weights = list(path.glob("*.safetensors"))
    missing = [item.name for item in required if not item.is_file()]
    if missing or not weights:
        detail = ", ".join(missing) or "model weights"
        raise RuntimeError(f"Orchestrator download is incomplete: missing {detail}")


def download_live():
    # Use the runtime's exact revisions, rather than drifting upstream defaults.
    sys.path.insert(0, str(APP_ROOT))
    from kira_live.weights import KIRA_LIVE_WEIGHT_SPECS, resolve_weight
    from huggingface_hub import hf_hub_download

    for spec in KIRA_LIVE_WEIGHT_SPECS:
        print(f"Downloading Live {spec.role}: {spec.model_id}@{spec.revision}", flush=True)
        download(spec.model_id, revision=spec.revision)
        resolve_weight(spec)

    manifest = json.loads((APP_ROOT / "kira_live/model_manifest.json").read_text())
    paths = [manifest["training_status"][key]["checkpoint"] for key in (
        "thinker_ple_moe", "live_emotion_stage1", "live_semantic_bridge"
    )]
    paths.append("training_runs/kira_live_emotion_v1/feature_normalization.npz")
    for relative in paths:
        target = (APP_ROOT / relative).resolve()
        target.relative_to(APP_ROOT.resolve())
        if target.is_file() and target.stat().st_size:
            print(f"Keeping existing Live checkpoint: {relative}", flush=True)
            continue
        cached = hf_hub_download(
            repo_id="saggamer/KIRA-LIVE-1",
            revision="9b7d5c2f39daafd0bd9949414f06223d0d1b4f6f",
            filename=relative, token=os.environ.get("HF_TOKEN") or None,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        shutil.copyfile(cached, temporary)
        temporary.replace(target)
    from kira_live.checkpoint_bundle import resolve_live_checkpoint
    resolve_live_checkpoint()
    print("KIRA Live checkpoint bundle ready", flush=True)


def download_voice():
    for repo in ("Qwen/Qwen3-ASR-0.6B", "Systran/faster-distil-whisper-small.en"):
        print(f"Downloading voice model: {repo}", flush=True)
        download(repo)
    target_dir = APP_ROOT / "models" / "kokoro"
    target_dir.mkdir(parents=True, exist_ok=True)
    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
    for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        target = target_dir / name
        if target.is_file() and target.stat().st_size:
            continue
        temporary = target.with_name(name + ".download")
        try:
            print(f"Downloading Kokoro asset: {name}", flush=True)
            with urllib.request.urlopen(base + name, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if not temporary.stat().st_size:
                raise RuntimeError(f"Empty Kokoro download: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrator-only", action="store_true")
    parser.add_argument("--whisper-only", action="store_true")
    parser.add_argument("--all", action="store_true", help="Also fetch Live checkpoints, Qwen ASR, portable Whisper and Kokoro voice assets.")
    args = parser.parse_args()
    if args.orchestrator_only and args.whisper_only:
        parser.error("Choose at most one of --orchestrator-only and --whisper-only")
    if args.all and (args.orchestrator_only or args.whisper_only):
        parser.error("--all cannot be combined with a single-model option")

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
        if args.all:
            download_live()
            download_voice()
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        print(
            "If Hugging Face requests authentication or license acceptance, run "
            f"`{Path(sys.executable).parent / 'hf'} auth login`, accept the model terms in your browser, and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
