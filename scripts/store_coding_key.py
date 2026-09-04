#!/usr/bin/env python3
"""Store a coding-provider credential in macOS Keychain without terminal echo."""

import argparse
import getpass
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("openai", "anthropic", "gemini", "openrouter", "custom"))
    args = parser.parse_args()
    key = getpass.getpass(f"{args.provider} key: ").strip()
    if not key:
        raise SystemExit("No key supplied.")
    result = subprocess.run(
        [
            "/usr/bin/security",
            "add-generic-password",
            "-U",
            "-a",
            "default",
            "-s",
            f"ai.kiraos.coding.{args.provider}",
            "-w",
            key,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    key = ""
    if result.returncode != 0:
        print("Keychain rejected the credential.", file=sys.stderr)
        raise SystemExit(1)
    print(f"{args.provider} credential stored in macOS Keychain.")


if __name__ == "__main__":
    main()
