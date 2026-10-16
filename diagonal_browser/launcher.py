"""Launch the Diagonal desktop browser from Kira's native UI."""

from pathlib import Path
import subprocess

from .kira_client import _node_binary


ROOT = Path(__file__).resolve().parent


def launch_browser():
    electron_cli = ROOT / "node_modules" / "electron" / "cli.js"
    if not electron_cli.is_file():
        raise RuntimeError("Diagonal Browser is not installed. Run SETUP_DIAGONAL_BROWSER.command first.")
    process = subprocess.Popen(
        [_node_binary(), str(electron_cli), str(ROOT / "desktop.mjs")],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"ok": True, "pid": process.pid}
