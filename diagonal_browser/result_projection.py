"""Bounded, valid JSON observations for Kira's small and large models."""

from __future__ import annotations

import json


def _compact(value, *, text_limit: int, element_limit: int):
    if isinstance(value, list):
        return [_compact(item, text_limit=text_limit, element_limit=element_limit)
                for item in value[:element_limit]]
    if not isinstance(value, dict):
        return value[:text_limit] if isinstance(value, str) else value
    result = {}
    for key, item in value.items():
        if key == "elements" and isinstance(item, list):
            result[key] = [
                {field: element[field] for field in (
                    "index", "role", "name", "x", "y", "gridCell", "href", "value",
                    "hasValue", "valueLength", "sensitive", "options",
                ) if field in element}
                for element in item[:element_limit] if isinstance(element, dict)
            ]
            if len(item) > element_limit:
                result["omittedElements"] = len(item) - element_limit
        elif key in {"text", "body", "content"} and isinstance(item, str):
            result[key] = item[:text_limit]
            if len(item) > text_limit:
                result[f"{key}OmittedChars"] = len(item) - text_limit
        else:
            result[key] = _compact(item, text_limit=text_limit, element_limit=element_limit)
    return result


def render_browser_result(value: object, *, max_chars: int = 16000) -> str:
    """Never cut a serialized result mid-JSON or silently lose its controls."""
    if not isinstance(value, dict):
        value = {"result": value}
    # Keep history pagination aligned with the actual entries shown to the model.
    # Generic list clipping would otherwise skip records at the next offset.
    entries = value.get("entries")
    if isinstance(entries, list) and len(entries) > 10 and all(
            isinstance(entry, dict) and "historyId" in entry for entry in entries):
        value = {**value, "entries": entries[:10], "nextOffset": value.get("offset", 0) + 10,
                 "omittedEntries": len(entries) - 10}
    for text_limit, element_limit in ((5500, 80), (3500, 60), (1800, 40), (800, 20)):
        projected = _compact(value, text_limit=text_limit, element_limit=element_limit)
        encoded = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return encoded
    fallback = {key: value[key] for key in (
        "tabId", "url", "title", "status", "path", "screenshot", "observationId",
        "historyId", "nextOffset", "nextTextOffset", "nextResultOffset",
        "enabled", "site", "siteEnabled", "activeOnSite", "ready", "engine", "blocked",
        "networkRules", "cosmeticRules", "updatedAt",
    ) if key in value}
    fallback["resultTruncated"] = True
    fallback["message"] = "Browser result exceeded the model budget. Inspect this page in smaller steps."
    return json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))
