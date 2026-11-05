"""Open the real Qt UI under Xvfb with isolated HOME/profile state in CI."""
from pathlib import Path
import os
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import webview
from interface import LazyKiraAPI

root = Path(__file__).resolve().parents[1]
os.chdir(root)
window = webview.create_window("KIRA Linux smoke", url=str(root / "ui.html"), js_api=LazyKiraAPI(), width=1200, height=800)
passed = []

def loaded():
    try:
        assert window.evaluate_js("document.title")
        assert window.evaluate_js("document.body.innerText.length") > 100
        passed.append(True)
    finally:
        window.destroy()

window.events.loaded += loaded
timer = threading.Timer(45, lambda: os._exit(2))
timer.daemon = True
timer.start()
webview.start(gui="qt")
timer.cancel()
assert passed, "KIRA desktop did not finish loading"
print("KIRA Linux Qt UI loaded successfully")
