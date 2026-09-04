"""Serve the real UI and Python API with disposable data, never production keys.

This is a test-only HTTP transport for the pywebview API, not a simulated backend.
OS dialogs/GPU/voice still require a native desktop verification run.
"""
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
ROOT = Path(tempfile.mkdtemp(prefix="kira-ui-check-"))
os.environ["HOME"] = str(ROOT / "home")
Path(os.environ["HOME"]).mkdir()
for key in list(os.environ):
    if key.endswith("API_KEY"):
        del os.environ[key]

import interface

interface.BASE_DIR = str(ROOT)


class IsolatedBrain(interface.KiraBrain):
    def _coding_key_service(self, provider):
        return "ai.kiraos.isolated." + ROOT.name + "." + provider


brain = IsolatedBrain()
api = interface.LazyKiraAPI()
api._brain = brain
TOKEN = secrets.token_urlsafe(24)
ALLOWED = {
    "warm_after_ui_ready", "new_chat", "get_chats", "load_chat", "delete_chat",
    "get_user_profile", "set_user_display_name", "get_chat_sources", "poll_updates",
    "get_slash_suggestions", "get_voice_status", "get_coding_session", "get_coding_settings",
    "get_coding_budget", "get_model_library", "get_integration_settings", "get_plugin_settings",
    "get_mcp_settings", "get_kira_menu", "import_attachment", "get_memory_status",
    "create_mcp_draft", "save_coding_session",
    "create_scheduled_item", "manage_scheduled_task", "get_scheduled_tasks", "test_mcp_draft",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/":
            return self.respond({"error": "not found"}, 404)
        bridge = """<script>
window.pywebview = {api: new Proxy({}, {get: (_, method) => async (...args) => {
 const response = await fetch('/api', {method:'POST', headers:{'Content-Type':'application/json','X-Kira-Test':%s},body:JSON.stringify({method,args})});
 const result = await response.json(); if (!response.ok) throw new Error(result.error); return result;
}})};
window.addEventListener('load', () => window.dispatchEvent(new Event('pywebviewready')));
</script>""" % json.dumps(TOKEN)
        html = (APP / "ui.html").read_text().replace("</head>", bridge + "</head>")
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api" or self.headers.get("X-Kira-Test") != TOKEN:
            return self.respond({"error": "unauthorized"}, 403)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size > 46 * 1024 * 1024:
                return self.respond({"error": "too large"}, 413)
            payload = json.loads(self.rfile.read(size))
            method = payload["method"]
            if method not in ALLOWED:
                return self.respond({"error": f"{method} requires native or separately authorized testing"}, 403)
            if method == "create_scheduled_item" and (not payload.get("args") or payload["args"][0] != "reminder"):
                return self.respond({"error": "Only reminders may be scheduled in this UI test."}, 403)
            target = api if hasattr(api, method) else brain
            result = getattr(target, method)(*payload.get("args", []))
            self.respond(result)
        except Exception as exc:
            self.respond({"error": str(exc)}, 500)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}/", "test_data": str(ROOT)}), flush=True)
    try:
        server.serve_forever()
    finally:
        brain.is_running = False
        server.server_close()
