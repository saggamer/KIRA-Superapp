#!/usr/bin/env python3
"""Run one end-to-end KIRA Vibe Coding build with an ephemeral Gemini key."""

import getpass
import os
import sys


APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.realpath(os.path.join(APP_ROOT, "..", "kira_windsurf_demo"))


def main():
    key = getpass.getpass("Gemini key: ").strip()
    if not key:
        raise SystemExit("No key supplied.")
    os.environ["GEMINI_API_KEY"] = key
    key = ""

    sys.path.insert(0, APP_ROOT)
    from interface import KiraBrain

    brain = KiraBrain()
    saved = brain.save_coding_provider(
        "gemini",
        "gemini-3.5-flash",
        "",
        "https://generativelanguage.googleapis.com",
    )
    if not saved.get("ok"):
        raise SystemExit("Provider setup failed: " + str(saved.get("error", "unknown error")))

    print("Testing Gemini connectivity through KIRA...")
    connection = brain.test_coding_provider("gemini", "gemini-3.5-flash")
    if not connection.get("ok"):
        raise SystemExit("Connectivity failed: " + str(connection.get("error") or connection.get("response")))
    print("Gemini connectivity: PASS")

    chat = brain.new_chat("Build Diagonal Flow in Windsurf")
    chat_id = chat["id"]
    session = brain.configure_coding_session(
        chat_id,
        PROJECT,
        "Windsurf",
        "gemini",
        "gemini-3.5-flash",
    )
    if not session.get("ok"):
        raise SystemExit("Coding session failed: " + str(session.get("error", "unknown error")))

    task = """
Build a complete, polished local web app named Diagonal Flow for managing software projects and agent tasks.
This is an end-to-end Vibe Coding bridge test, so create the working application rather than an explanation.

Requirements:
- Use plain HTML, CSS, and JavaScript with no external runtime dependencies.
- Create index.html, styles.css, app.js, and README.md.
- Provide a compact professional dark interface with a sidebar, project summary, searchable task board,
  Backlog/In Progress/Done columns, task creation and editing, priority labels, due dates, and progress metrics.
- Support drag-and-drop between columns, filtering, responsive mobile/desktop layouts, keyboard accessibility,
  empty states, and localStorage persistence.
- Seed a few realistic example tasks on first launch, but make all data editable.
- Include a clear reset-demo-data action with confirmation.
- Avoid marketing copy, gradients, decorative blobs, nested cards, and oversized headings.
- Do not add API keys, analytics, network calls, package managers, or long-running server commands.
- Return complete file contents and only bounded verification commands if a command is genuinely needed.
""".strip()

    print("Running Orchestrator V1 planning and Gemini implementation...")
    result = brain._handle_vibe_coding_prompt(task, chat_id, [])
    pending = brain.get_pending_permissions()
    coding_request = next((item for item in reversed(pending) if item.get("action") == "coding_apply"), None)
    if coding_request is None:
        raise SystemExit("KIRA produced no applicable coding plan: " + result)
    if brain.current_brain != "orchestrator":
        raise SystemExit("Orchestrator V1 did not load; refusing to claim an Orchestrator-driven build.")

    print("Orchestrator V1 loaded: PASS")
    print("Validated coding plan: PASS")
    applied = brain._execute_permissioned_action(coding_request)
    if "applied and verified" not in applied.lower():
        raise SystemExit("KIRA apply failed: " + applied)
    print(applied)
    print("KIRA Vibe Coding build: PASS")


if __name__ == "__main__":
    main()
