import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from linux_model_worker import server_config, prompt_messages


class LinuxRuntimeTests(unittest.TestCase):
    def test_configuration_requires_explicit_model_and_secure_remote_url(self):
        for base, model in [("", ""), ("http://example.com/v1", "m"), ("https://user:secret@example.com/v1", "m")]:
            with patch.dict(os.environ, {"KIRA_MODEL_BASE_URL": base, "KIRA_MODEL_NAME": model}):
                with self.assertRaises(ValueError):
                    server_config()
        with patch.dict(os.environ, {"KIRA_MODEL_BASE_URL": "https://example.com/v1/", "KIRA_MODEL_NAME": "test"}):
            self.assertEqual(server_config(), ("https://example.com/v1", "test"))

    def test_prompt_conversion_preserves_history(self):
        prompt = "<bos><|turn>user\nhello<turn|>\n<|turn>model\nhi<turn|>\n<|turn>user\nnext<turn|>\n<|turn>model\n"
        self.assertEqual(prompt_messages(prompt), [
            {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "next"}])

    def test_worker_ipc_generation_privacy_and_failure(self):
        observed = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed.append((self.path, body, self.headers.get("Authorization")))
                if body["messages"][0]["content"] == "fail":
                    self.send_error(500, "private-server-detail")
                    return
                payload = json.dumps({"choices": [{"message": {"content": "<think>private reasoning</think>Public answer"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            env = {**os.environ, "KIRA_MODEL_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "KIRA_MODEL_NAME": "fixture", "KIRA_MODEL_API_KEY": "fixture-secret"}
            requests = [
                {"id": "early", "type": "generate"},
                {"id": "load", "type": "load", "brain": "orchestrator"},
                {"id": "ok", "type": "generate", "prompt": "<|turn>user\nhello<turn|>\n<|turn>model\n", "max_tokens": 100},
                {"id": "error", "type": "generate", "prompt": "fail"},
                {"id": "stop", "type": "shutdown"}]
            result = subprocess.run([sys.executable, str(ROOT / "linux_model_worker.py")], input="\n".join(map(json.dumps, requests)) + "\n", env=env, text=True, capture_output=True, timeout=15, check=True)
            responses = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual([r["id"] for r in responses], [r["id"] for r in requests])
            self.assertFalse(responses[0]["ok"])
            self.assertTrue(responses[1]["loaded"])
            self.assertEqual(responses[2]["content"], "Public answer")
            self.assertFalse(responses[3]["ok"])
            self.assertTrue(responses[4]["shutdown"])
            self.assertNotIn("fixture-secret", result.stdout + result.stderr)
            self.assertNotIn("private-server-detail", result.stdout + result.stderr)
            self.assertEqual(observed[0][0], "/v1/chat/completions")
            self.assertEqual(observed[0][1]["model"], "fixture")
            self.assertEqual(observed[0][2], "Bearer fixture-secret")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
