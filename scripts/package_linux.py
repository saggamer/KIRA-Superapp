#!/usr/bin/env python3
"""Freeze public runtime source using an allowlist; never walk runtime data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
NAME = "kira-superapp-linux-preview-2026.11.05"
ROOT_FILES = {"LICENSE", "README.md", "HARNESS_PROVENANCE.md", "commands.json", "superapp",
              "INSTALL_LINUX.sh", "RUN_LINUX.sh", "requirements-linux.txt"}
CODE_DIRS = {"kira_live", "orchestrator_v15", "diagonal_browser", "horizon_mlx"}
SOURCE_EXTENSIONS = {".py", ".mjs", ".js", ".cjs", ".html", ".css", ".svg", ".md"}


def public_files():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    selected = {ROOT / name for name in tracked if name and (
        Path(name).parts[0] in CODE_DIRS | {"assets", "docs", "tests", "scripts", "external_harnesses"}
        or len(Path(name).parts) == 1 and (Path(name).suffix == ".py" or name in ROOT_FILES))}
    selected.update(ROOT / name for name in ROOT_FILES)
    selected.update(ROOT.glob("*.py"))
    selected.update([ROOT / "docs/LINUX.md", ROOT / "scripts/package_linux.py", ROOT / "tests/test_linux_runtime.py", ROOT / "scripts/smoke_linux_ui.py"])
    for directory in CODE_DIRS:
        # Restrict each walk before descending into node_modules, profiles or assets.
        import os
        for parent, dirs, files in os.walk(ROOT / directory):
            dirs[:] = [d for d in dirs if d not in {"node_modules", "artifacts", "__pycache__", "filters"} and not d.startswith(".")]
            selected.update(Path(parent) / name for name in files if Path(name).suffix in SOURCE_EXTENSIONS)
    selected.update((ROOT / "diagonal_browser/filters").glob("*"))
    for file in sorted(selected):
        if not file.is_file() or file.is_symlink():
            continue
        relative = file.relative_to(ROOT)
        if any(part.startswith(".venv") or part in {"node_modules", ".profile", "artifacts", "__pycache__"} for part in relative.parts):
            continue
        yield file, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release/linux")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp) / NAME
        for source, relative in public_files():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
        (stage / "RELEASE_MANIFEST.json").write_text(json.dumps({"name": NAME, "files": manifest}, indent=2) + "\n")
        archive = args.output / (NAME + ".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(stage, arcname=NAME)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.output / "SHA256SUMS").write_text(f"{checksum}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "files": len(manifest), "sha256": checksum}))


if __name__ == "__main__":
    main()
