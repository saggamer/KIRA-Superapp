#!/usr/bin/env python3
"""Test and upload an allowlisted KIRA Live research-preview add-on package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Your Hugging Face namespace/repository")
    parser.add_argument("--upload", action="store_true", help="Upload publicly; otherwise only validate/package")
    args = parser.parse_args()
    if len(args.repo.split("/")) != 2:
        parser.error("Use namespace/repository")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT, check=True)
    manifest = json.loads((ROOT / "kira_live/model_manifest.json").read_text())
    stage = Path(tempfile.mkdtemp(prefix="kira-live-preview-"))
    checksums = {}
    # No chat databases, raw recordings, credentials, training examples,
    # rejected experiments, or third-party donor weight copies are uploaded.
    for name in ("thinker_ple_moe", "live_emotion_stage1", "live_semantic_bridge"):
        entry = manifest["training_status"][name]
        if entry.get("decision") not in ("accepted", "accepted_by_speaker_disjoint_emotion_slice_gate"):
            raise RuntimeError(f"Unaccepted component: {name}")
        source = (ROOT / entry["checkpoint"]).resolve()
        source.relative_to(ROOT)
        destination = stage / entry["checkpoint"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        with source.open("rb") as checkpoint:
            checksums[entry["checkpoint"]] = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    for source in sorted((ROOT / "kira_live").glob("*.py")):
        destination = stage / "kira_live" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(ROOT / "kira_live/model_manifest.json", stage / "kira_live/model_manifest.json")
    normalization = Path("training_runs/kira_live_emotion_v1/feature_normalization.npz")
    source = ROOT / normalization
    if not source.is_file():
        raise RuntimeError("Required emotion normalization is missing")
    destination = stage / normalization
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with source.open("rb") as data:
        checksums[str(normalization)] = hashlib.file_digest(data, "sha256").hexdigest()
    shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
    (stage / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    shutil.copy2(ROOT / "docs/KIRA_LIVE_1_MODEL_CARD.md", stage / "README.md")
    print(f"Validated preview package: {stage}")
    if args.upload:
        hf = Path(sys.executable).parent / "hf"
        subprocess.run([str(hf), "auth", "whoami"], check=True)
        subprocess.run([str(hf), "upload", args.repo, str(stage), ".", "--type", "model", "--commit-message", "KIRA Live 1 research preview"], check=True)


if __name__ == "__main__":
    main()
