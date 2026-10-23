"""Write Kira's WEB_SEARCH/WEB_RESEARCH evidence to Diagonal's shared history."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import uuid


_SECRET = re.compile(r"^(code|state|nonce|password|passwd|secret|token|access_token|refresh_token|id_token|api_key|apikey|authorization|session|sessionid)$", re.I)


def _url(value):
    try:
        parsed = urlsplit(str(value or ""))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        query = urlencode([(key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
                           if not _SECRET.match(key)])
        fragment = "" if re.search(r"access_token|id_token|code=|token=", parsed.fragment, re.I) else parsed.fragment
        return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, query, fragment))
    except ValueError:
        return ""


def save_history(*, kind, query="", title="", url="", text="", results=None, profile_dir=None):
    """Save an independent record atomically; Node's history tools read this schema."""
    directory = Path(profile_dir or os.environ.get("DIAGONAL_PROFILE_DIR") or Path(__file__).parent / ".profile") / "search_history"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    entry = {"version": 1, "id": str(uuid.uuid4()), "kind": kind, "source": "llm",
             "createdAt": now, "updatedAt": now, "title": str(title)[:500], "url": _url(url)}
    if kind == "search":
        entry.update(query=str(query)[:2000], mode="web", results=[
            {"title": str(item.get("title", ""))[:500], "url": _url(item.get("url") or item.get("href")),
             "snippet": str(item.get("snippet") or item.get("body") or "")[:3000]}
            for item in (results or [])[:100]
        ])
    elif kind == "page":
        entry.update(text=str(text)[:25000], textTruncated=len(str(text)) > 25000)
    else:
        raise ValueError("Invalid history kind")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(entry, handle, ensure_ascii=False)
        os.replace(temporary, directory / f"{entry['id']}.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return entry["id"]
