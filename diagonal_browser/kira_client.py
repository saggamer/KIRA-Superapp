"""Persistent stdio MCP client used by Kira's browser action bridge."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parent


def _node_binary():
    candidates = (
        os.environ.get("DIAGONAL_NODE_BIN"),
        shutil.which("node"),
        "/opt/homebrew/bin/node",
        "/usr/local/bin/node",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("Node.js 20+ was not found. Run SETUP_DIAGONAL_BROWSER.command first.")


class BrowserMCPClient:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.serial = 0

    def _send(self, method, params=None, expect=True):
        self.serial += 1
        request = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if expect:
            request["id"] = self.serial
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        if not expect:
            return None
        deadline = time.monotonic() + 60
        invalid_lines = 0
        while True:
            ready, _, _ = select.select([self.process.stdout], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                raise TimeoutError(f"Diagonal Browser did not answer {method} within 60 seconds")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Diagonal Browser MCP server exited. Run `npm ci` in diagonal_browser and check Node 20+.")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                if invalid_lines >= 10:
                    raise RuntimeError("Diagonal Browser MCP server sent too many non-JSON lines")
                continue
            if not isinstance(reply, dict):
                continue
            if reply.get("id") != self.serial:
                continue
            if "error" in reply:
                raise RuntimeError(reply["error"].get("message", "MCP error"))
            return reply.get("result", {})

    def _start(self):
        if self.process and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            [_node_binary(), str(ROOT / "server.mjs")],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        try:
            self._send("initialize", {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "kira-os", "version": "0.1.0"},
            })
            self._send("notifications/initialized", expect=False)
        except Exception:
            self.close()
            raise

    def close(self):
        process, self.process = self.process, None
        if not process:
            return
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=2)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()

    def call(self, name, arguments=None):
        with self.lock:
            self._start()
            result = self._send("tools/call", {"name": name, "arguments": arguments or {}})
            content = result.get("content", [])
            text = "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
            if result.get("isError"):
                raise RuntimeError(text or f"{name} failed")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}


_client = BrowserMCPClient()
atexit.register(_client.close)


def call_browser_tool(name, arguments=None):
    return _client.call(name, arguments)
