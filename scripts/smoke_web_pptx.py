#!/usr/bin/env python3
"""Exercise KIRA's web reader and evidence-backed PPTX pipeline end to end."""

import argparse
import json
import os
import queue
import shutil
import sys
from pathlib import Path
from urllib.parse import quote_plus
from urllib.parse import urlparse


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from actions import KiraActionsMixin


class SmokeActions(KiraActionsMixin):
    def __init__(self, root):
        self.app_root = str(APP_DIR)
        self.agentic_workspace = str(root / "workspace")
        self.agentic_logs_path = str(root / "logs")
        self.architect_workspace = str(root / "architect")
        self.home_path = str(root)
        self.active_chat_id = "web-pptx-smoke"
        self.response_queue = queue.Queue()
        self.last_artifacts = []
        self.last_artifact_path = ""
        self.artifacts_index_path = str(root / "artifacts.json")
        self.web_evidence_by_chat = {}
        self.computer_roots = {
            "home": str(root),
            "agentic_workspace": self.agentic_workspace,
        }
        os.makedirs(self.agentic_workspace, exist_ok=True)
        os.makedirs(self.agentic_logs_path, exist_ok=True)
        os.makedirs(self.architect_workspace, exist_ok=True)

    def _log_agentic_event(self, *_args, **_kwargs):
        return None

    def _load_chat_messages(self, _chat_id):
        return []


def write_fixture(site_root):
    site_root.mkdir(parents=True, exist_ok=True)
    visual_sources = {
        "kira-product.png": APP_DIR / "demo" / "KIRA_OS_Product_Demo_Poster.png",
        "kira-tree.png": APP_DIR / "demo" / "KIRA_OS_Specialist_TREE_Demo_Poster.png",
        "kira-logo.png": APP_DIR / "website" / "assets" / "kira_logo_sample4.png",
        "kira-icon.png": APP_DIR / "assets" / "kira_logo_icon.png",
        "kira-blackhole.jpg": APP_DIR / "website" / "assets" / "interstellar_blackhole.jpg",
    }
    for name, source in visual_sources.items():
        if not source.is_file():
            raise RuntimeError(f"missing smoke visual: {source}")
        shutil.copy2(source, site_root / name)

    (site_root / "search.html").write_text(
        """<!doctype html><html><head><title>KIRA Search Fixture</title></head><body>
        <main>
          <a href="product.html">KIRA OS local agentic environment</a>
          <a href="architecture.html">KIRA OS specialist TREE architecture</a>
        </main></body></html>""",
        encoding="utf-8",
    )
    (site_root / "product.html").write_text(
        """<!doctype html><html><head><title>KIRA OS Product Overview</title>
        <meta property="og:image" content="kira-product.png"></head><body>
        <article><h1>KIRA OS Product Overview</h1>
        <img src="kira-logo.png" alt="KIRA OS identity and local agentic compute">
        <img src="kira-blackhole.jpg" alt="KIRA OS immersive local compute background">
        <p>KIRA OS is a local agentic environment that connects an orchestration model to verified
        research, media, artifact, application, and computer-control branches.</p>
        <p>The product keeps model judgment separate from execution evidence: a task is complete only
        after a branch returns a real result that can be inspected.</p></article></body></html>""",
        encoding="utf-8",
    )
    (site_root / "architecture.html").write_text(
        """<!doctype html><html><head><title>KIRA OS Specialist TREE</title>
        <meta property="og:image" content="kira-tree.png"></head><body>
        <article><h1>KIRA OS Specialist TREE</h1>
        <img src="kira-tree.png" alt="KIRA OS specialist TREE architecture and branch verification">
        <img src="kira-icon.png" alt="KIRA OS local orchestration icon">
        <p>The TREE routes independent research and media work in parallel, then joins their evidence
        before an artifact specialist builds and verifies the requested output.</p>
        <p>This design lets Orchestrator V1 choose a route while deterministic gates reject missing,
        irrelevant, or low-quality results before completion is reported.</p></article></body></html>""",
        encoding="utf-8",
    )


def build_spec(images):
    repo = "https://github.com/saggamer/KIRA-OS"
    model = "https://huggingface.co/saggamer/Orchestrator_V1"
    image_a = images[0]
    image_b = images[1] if len(images) > 1 else images[0]
    image_c = images[2] if len(images) > 2 else image_a
    image_d = images[3] if len(images) > 3 else image_b
    image_e = images[4] if len(images) > 4 else image_c
    slides = [
        (
            "A local agentic environment",
            "KIRA OS gives a local orchestration model access to research, media, artifacts, applications, "
            "and computer-control tools through explicit branch contracts. The model decides what work is "
            "needed while the runtime records what actually happened.",
            image_a,
        ),
        (
            "THE TREE: route, join, verify",
            "Independent research and media branches can run together. Their outputs join into a structured "
            "artifact request, and the specialist verifier blocks completion when sources, images, links, "
            "content depth, or a real output path are missing.",
            image_b,
        ),
        (
            "Orchestrator V1 as the decision layer",
            "Orchestrator V1 interprets goals, selects branches, examines returned evidence, and writes a "
            "human-facing answer. The execution layer remains responsible for file existence, tool results, "
            "permission boundaries, and verifiable completion.",
            image_c,
        ),
        (
            "Artifacts that carry evidence",
            "Slides and documents are assembled from structured specifications, downloaded media, and direct "
            "source links. Pixel-level media checks and post-generation inspection prevent blank placeholders "
            "or unsupported image files from being passed off as finished visuals.",
            image_d,
        ),
        (
            "Evidence before completion",
            "KIRA OS does not treat a model sentence as proof that an action happened. A generated file must "
            "exist and pass quality checks; research must include fetched page text and direct URLs; risky "
            "changes remain visible to the permission layer.",
            image_e,
        ),
    ]
    lines = [
        "TITLE: KIRA OS - Evidence Before Completion",
        "SUBTITLE: A real-media web research and artifact verification smoke test",
        "THEME: dark cyan",
    ]
    for title, body, image in slides:
        lines.extend(
            [
                f"SLIDE: {title}",
                f"BODY: {body}",
                f"IMAGE: {image}",
                f"LINK: KIRA OS repository | {repo}",
                f"LINK: Orchestrator V1 model | {model}",
            ]
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-query",
        default="",
        help="Also run an external search through KIRA. Network access must be available to this process.",
    )
    args = parser.parse_args()

    root = APP_DIR / "stress_lab" / "web_pptx_smoke"
    if root.exists():
        shutil.rmtree(root)
    site_root = root / "site"
    write_fixture(site_root)
    actions = SmokeActions(root)

    base_url = "https://kira-smoke.invalid"

    original_fetch = actions._web_fetch_html_source
    original_download = actions._download_web_image

    def fixture_fetch(target, timeout=20, max_bytes=2_000_000):
        parsed = urlparse(str(target))
        if parsed.netloc != "kira-smoke.invalid":
            return original_fetch(target, timeout=timeout, max_bytes=max_bytes)
        name = Path(parsed.path).name or "search.html"
        path = site_root / name
        if not path.is_file():
            raise RuntimeError(f"fixture page not found: {name}")
        return path.read_text(encoding="utf-8")[:max_bytes], "fixture-http", "text/html; charset=utf-8"

    def fixture_download(source, title="web_image"):
        parsed = urlparse(str(source))
        if parsed.netloc != "kira-smoke.invalid":
            return original_download(source, title=title)
        source_path = site_root / Path(parsed.path).name
        if not source_path.is_file():
            return "", f"fixture image not found: {source_path.name}"
        target_dir = Path(actions.agentic_workspace) / "web_images"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        return str(target_path), ""

    def local_search_results(query, limit=6):
        search_url = f"{base_url}/search.html?q={quote_plus(str(query))}"
        html, _source, _content_type = actions._web_fetch_html_source(search_url)
        return actions._extract_web_links(html, search_url, limit=limit), search_url

    live_result = ""
    if args.live_query:
        live_result = actions._web_search_tool(args.live_query)

    actions._web_fetch_html_source = fixture_fetch
    actions._download_web_image = fixture_download
    actions._web_search_results = local_search_results
    query = "KIRA OS local agentic environment specialist TREE verification"
    normal_search = actions._web_search_tool(query)
    if "WEB_SEARCH ERROR" in normal_search or "Fetched top pages:" not in normal_search:
        raise RuntimeError("normal KIRA search/reader smoke failed")

    hub = actions._specialist_hub()
    specialist_results = hub.execute_many(
        [
            {"agent": "research", "query": query, "max_pages": 2},
            {"agent": "media", "query": "KIRA OS local agentic environment TREE", "limit": 5},
        ],
        chat_id=actions.active_chat_id,
    )
    research, media = specialist_results
    if research.get("status") != "verified":
        raise RuntimeError(f"research specialist failed: {research}")
    if media.get("status") != "verified" or len(media.get("assets", [])) < 2:
        raise RuntimeError(f"media specialist failed: {media}")

    evidence = research.get("evidence", "")
    actions.web_evidence_by_chat[actions.active_chat_id] = evidence
    slides = hub.execute(
        {
            "agent": "slides",
            "request": "Create an evidence-backed KIRA OS PPTX with web research, presentation-quality images, and source hyperlinks.",
            "evidence": evidence,
            "spec": build_spec(media["assets"]),
        },
        chat_id=actions.active_chat_id,
    )
    if slides.get("status") != "verified":
        raise RuntimeError(f"slides specialist failed: {slides}")

    report = {
        "passed": True,
        "normal_search": {
            "query": query,
            "has_real_links": "Real result links:" in normal_search,
            "has_readable_pages": "Readable text:" in normal_search,
        },
        "research": {
            "status": research.get("status"),
            "elapsed_seconds": research.get("elapsed_seconds"),
        },
        "media": {
            "status": media.get("status"),
            "assets": media.get("assets", []),
            "elapsed_seconds": media.get("elapsed_seconds"),
        },
        "pptx": {
            "status": slides.get("status"),
            "path": slides.get("path"),
            "quality": slides.get("quality", {}),
            "elapsed_seconds": slides.get("elapsed_seconds"),
        },
        "live_search": {
            "requested": bool(args.live_query),
            "passed": bool(live_result) and "WEB_SEARCH ERROR" not in live_result,
            "preview": live_result[:1600],
        },
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
