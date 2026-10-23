import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def test_superapp_uses_runtime_override_from_another_directory(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ['zsh', str(root / 'superapp')], cwd=directory,
                env={**os.environ, 'KIRA_PYTHON': '/bin/echo'},
                capture_output=True, text=True, timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[-1], str(root / 'interface.py'))

    def test_model_downloader_uses_runtime_override_and_forwards_arguments(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ['zsh', str(root / 'DOWNLOAD_MODELS.command'), '--whisper-only'],
            env={**os.environ, 'KIRA_PYTHON': '/bin/echo'},
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'scripts/download_models.py --whisper-only')
