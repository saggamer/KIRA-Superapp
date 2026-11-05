"""Linux preview model-server adapter using the existing isolated JSON-line IPC."""
import json
import os
import re
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from kira_live.privacy import sanitize_public_output


def server_config():
    base = os.environ.get("KIRA_MODEL_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("KIRA_MODEL_NAME", "").strip()
    url = urlsplit(base)
    if not model or url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Set KIRA_MODEL_BASE_URL (e.g. http://127.0.0.1:8080/v1) and KIRA_MODEL_NAME to your running OpenAI-compatible model server.")
    if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote model servers require HTTPS.")
    return base, model


def prompt_messages(prompt):
    turns = re.findall(r"<\|turn>(user|model)\n(.*?)<turn\|>", prompt, re.S)
    if not turns:
        return [{"role": "user", "content": prompt}]
    return [{"role": "assistant" if role == "model" else "user", "content": content}
            for role, content in turns]


def complete(request):
    base, model = server_config()
    body = {"model": model, "messages": prompt_messages(str(request.get("prompt", ""))),
            "max_tokens": max(32, min(1200, int(request.get("max_tokens", 900)))),
            "temperature": max(0, min(2, float(request.get("temperature", 0.35)))),
            "stream": False}
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("KIRA_MODEL_API_KEY", "").strip()
    if key:
        headers["Authorization"] = "Bearer " + key
    # Do not follow redirects carrying an Authorization header to another host.
    from urllib.request import HTTPRedirectHandler, build_opener
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    opener = build_opener(NoRedirect())
    with opener.open(Request(base + "/chat/completions", data=json.dumps(body).encode(), headers=headers), timeout=50) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Model server returned no text content.")
    return sanitize_public_output(content)


def main():
    loaded = False
    for line in sys.stdin:
        request_id = ""
        try:
            request = json.loads(line)
            request_id = request.get("id", "")
            kind = request.get("type")
            if kind == "shutdown":
                print(json.dumps({"id": request_id, "ok": True, "shutdown": True}), flush=True)
                break
            if kind == "load":
                if request.get("brain") != "orchestrator":
                    raise ValueError("Only the Orchestrator connection is supported on Linux.")
                server_config()
                loaded = True
                result = {"loaded": True, "backend": "model_server"}
            elif kind == "generate" and loaded:
                result = {"content": complete(request)}
            else:
                raise ValueError("Load the Orchestrator connection before generating.")
            print(json.dumps({"id": request_id, "ok": True, **result}), flush=True)
        except Exception as exc:
            # HTTP exception bodies and URLs can contain credentials or prompt data.
            error = str(exc) if isinstance(exc, ValueError) else "Model server request failed (" + type(exc).__name__ + "). Check the server and its configured model."
            print(json.dumps({"id": request_id, "ok": False, "error": error}), flush=True)


if __name__ == "__main__":
    main()
