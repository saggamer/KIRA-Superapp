#!/usr/bin/env python3
"""Verify KIRA's runtime and local model discovery without loading weights."""

import platform
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from interface import KiraBrain
from listen import LocalSTT


def main():
    if platform.system() != "Darwin":
        raise SystemExit("KIRA Superapp requires macOS.")
    if platform.machine() != "arm64":
        raise SystemExit("This MLX release requires Apple Silicon.")

    brain = object.__new__(KiraBrain)
    brain.app_root = str(APP_ROOT)
    model_path = brain._first_existing_path(brain._orchestrator_model_candidates())
    if not brain._is_loadable_model_path(model_path):
        raise SystemExit(f"Orchestrator V1 was not found or is incomplete: {model_path}")
    if not brain._is_model_storage_resident(model_path):
        raise SystemExit(f"Orchestrator V1 is a cloud placeholder, not a local model: {model_path}")

    stt = LocalSTT()
    if not stt.ready:
        raise SystemExit("A local speech-to-text runtime is not ready: " + stt.error)

    print("KIRA runtime verification: PASS")
    print("Orchestrator:", model_path)
    print("Speech-to-text:", stt.status())


if __name__ == "__main__":
    main()
