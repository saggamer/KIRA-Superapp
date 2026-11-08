"""Opt-in upstream-runtime smoke test against a local fixture API, never a paid model."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from test_coding_mode import CodingHarness

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "external_harnesses/deepseek-runtime/node_modules/.bin/dsh"


@unittest.skipUnless(os.environ.get("KIRA_TEST_DSH_RUNTIME") == "1" and BINARY.exists(),
                     "Set KIRA_TEST_DSH_RUNTIME=1 with dsh installed to run the local runtime smoke")
class DeepSeekRuntimeTests(unittest.TestCase):
    def test_real_upstream_planner_with_local_compatible_endpoint(self):
        plan = {"summary": "Fixture plan", "files": [], "commands": []}
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(request)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in [
                    {"id": "fixture", "object": "chat.completion.chunk", "model": "fixture-model",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": json.dumps(plan)}, "finish_reason": None}]},
                    {"id": "fixture", "object": "chat.completion.chunk", "model": "fixture-model",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 20}},
                ]:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                host = CodingHarness(root)
                host._coding_harness_path = lambda _name: str(BINARY)
                project = Path(root) / "project"
                project.mkdir()
                profile = {"provider": "nvidia_nim", "model": host.CODING_NEMOTRON_MODEL, "api_key": "fixture-key",
                           "base_url": f"http://127.0.0.1:{server.server_port}/v1"}
                raw = host._coding_external_plan("deepseek", profile, "Return exactly one JSON object.",
                                                 "Return an empty file plan.", str(project))
                self.assertEqual(host._coding_parse_plan(raw)["summary"], "Fixture plan")
                self.assertEqual(list(project.iterdir()), [])
                self.assertTrue(requests)
                self.assertEqual(len(requests), 1)
                self.assertFalse(requests[0].get("tools"))
                self.assertTrue(all(r["model"] == host.CODING_NEMOTRON_MODEL for r in requests))
                self.assertNotIn("thinking_token_budget", requests[0])
                self.assertTrue(requests[0]["chat_template_kwargs"]["enable_thinking"])
                self.assertNotIn("reasoning_effort", requests[0])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
