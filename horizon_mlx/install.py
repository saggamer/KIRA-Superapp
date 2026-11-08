"""Provision the image runtime separately from Kira's chat/voice dependencies."""
import os
from pathlib import Path
import subprocess
import sys
import venv


def runtime_python(app_root):
    return Path(os.environ.get('KIRA_HORIZON_PYTHON',Path(app_root)/'.venv-horizon-mlx/bin/python')).expanduser()


def ensure_runtime(app_root):
    executable=runtime_python(app_root)
    if not executable.is_file():
        if os.environ.get('KIRA_HORIZON_PYTHON'):raise FileNotFoundError('Configured Horizon Python is missing')
        venv.EnvBuilder(with_pip=True).create(executable.parent.parent)
    check="from importlib.metadata import version; assert version('mflux')=='0.24.0'; assert version('mlx').startswith('0.32.')"
    if subprocess.run([str(executable),'-c',check],capture_output=True).returncode:
        subprocess.run([str(executable),'-m','pip','install','-r',str(Path(app_root)/'requirements-horizon-mlx.txt')],check=True)
    return executable
