#!/usr/bin/env python3
"""Install and prefetch KIRA Live's local speech stack without slowing app boot."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
QWEN_MODEL = "Qwen/Qwen3-ASR-0.6B"


def run(command):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def install_stack(light=False):
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "sounddevice",
        "kokoro-onnx",
        "onnxruntime",
        "mlx-whisper",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "--force-reinstall",
        "git+https://github.com/TEN-framework/ten-vad.git",
    ])
    if not light:
        run([sys.executable, "-m", "pip", "install", "-U", "qwen-asr", "huggingface-hub"])


def prefetch_qwen():
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        QWEN_MODEL,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.tiktoken",
        ],
    )
    print(f"Qwen3-ASR cached at: {path}")


def report():
    sys.path.insert(0, str(ROOT))
    from listen import get_local_stt_status, create_vad
    from voice import get_voice_engine_status

    vad = create_vad(prefer_ten=True)
    try:
        vad_status = vad.status()
    finally:
        vad.close()
    print("STT:", get_local_stt_status())
    print("VAD:", vad_status)
    print("TTS:", get_voice_engine_status())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--light",
        action="store_true",
        help="Install the smaller MLX Whisper fallback instead of Qwen3-ASR 0.6B.",
    )
    parser.add_argument("--check", action="store_true", help="Only print current readiness.")
    args = parser.parse_args()

    if not args.check:
        install_stack(light=args.light)
        if not args.light:
            prefetch_qwen()
    report()


if __name__ == "__main__":
    main()
