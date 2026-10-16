"""JSON-only image search worker for the Diagonal desktop browser."""

import argparse
import json
import sys
from urllib.parse import urlparse

from ddgs import DDGS


def web_url(value):
    candidate = str(value or "").strip()
    parsed = urlparse(candidate)
    return candidate if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def dimension(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def search_images(query, limit=36, size=None, color=None, type_image=None, layout=None):
    raw = DDGS(timeout=15).images(
        query=query,
        max_results=max(1, min(limit, 48)),
        safesearch="moderate",
        size=size,
        color=color,
        type_image=type_image,
        layout=layout,
    )
    results = []
    for item in raw:
        image = web_url(item.get("image"))
        thumbnail = web_url(item.get("thumbnail"))
        source_url = web_url(item.get("url"))
        if not image or not thumbnail or not source_url:
            continue
        source = str(item.get("source") or "").strip() or urlparse(source_url).hostname or "Website"
        results.append({
            "title": str(item.get("title") or "Untitled image").strip()[:300],
            "image": image,
            "thumbnail": thumbnail,
            "url": source_url,
            "source": source[:120],
            "width": dimension(item.get("width")),
            "height": dimension(item.get("height")),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Search images for Diagonal Browser")
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=36)
    parser.add_argument("--size", choices=["Large", "Wallpaper"])
    parser.add_argument("--color", choices=["Monochrome"])
    parser.add_argument("--type-image", choices=["gif", "transparent"])
    parser.add_argument("--layout", choices=["Wide", "Tall", "Square"])
    args = parser.parse_args()
    try:
        print(json.dumps(search_images(
            args.query, args.max_results, args.size, args.color, args.type_image, args.layout
        )))
    except Exception as error:
        print(f"Image search failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
