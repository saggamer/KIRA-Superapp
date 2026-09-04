#!/usr/bin/env python3
"""
Non-invasive KIRA OS stress harness.

This creates a fake macOS workspace, a fake pywebview bridge, and a browser-ready
UI wrapper so the real ui.html can be booted and tested without touching Finder,
Terminal, user files, microphone hardware, model weights, or system apps.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
STRESS_DIR = APP_DIR / "stress_lab"
SIM_ROOT = STRESS_DIR / "mac_os_sim"
UI_HTML = APP_DIR / "ui.html"
SIM_HTML = STRESS_DIR / "ui_sim.html"
REPORT_JSON = STRESS_DIR / "stress_report.json"


REQUIRED_PYTHON_FILES = [
    "interface.py",
    "actions.py",
    "specialists.py",
    "voice.py",
    "listen.py",
    "memory.py",
    "vision.py",
    "kira_technical_architect/subagents/subagent_runtime.py",
]

REQUIRED_UI_IDS = [
    "model-selector",
    "kira-menu-toggle",
    "voice-toggle",
    "live-monitor-panel",
    "kira-menu-panel",
    "slash-panel",
    "user-input",
    "voice-live",
    "voice-listen-btn",
    "voice-live-input",
]

REQUIRED_API_METHODS = [
    "warm_after_ui_ready",
    "get_chats",
    "new_chat",
    "load_chat",
    "delete_chat",
    "send_prompt",
    "poll_updates",
    "toggle_voice_mode",
    "configure_voice_stack",
    "get_voice_status",
    "listen_once",
    "start_voice_turn",
    "record_voice_partial",
    "complete_voice_turn",
    "stop_voice",
    "set_screen_overlay_state",
    "get_live_snapshot",
    "get_kira_menu",
    "open_menu_target",
    "get_slash_suggestions",
    "approve_pending_tool",
    "reject_pending_tool",
]

REQUIRED_PACKAGES = {
    "webview": "pywebview",
    "mlx": "mlx",
    "mlx_lm": "mlx-lm",
    "fpdf": "fpdf2",
    "pptx": "python-pptx",
    "docx": "python-docx",
    "fitz": "PyMuPDF",
    "PIL": "Pillow",
    "speech_recognition": "SpeechRecognition",
}


FAKE_BRIDGE = r"""
<script>
(function () {
    const bootStart = performance.now();
    const state = {
        bootStart,
        loadAt: 0,
        readyAt: 0,
        apiCalls: [],
        prompts: [],
        overlayStates: [],
        updates: [],
        pending: {},
        chats: [],
        activeChatId: "",
        nextId: 1
    };

    function nowSeconds() {
        return Date.now() / 1000;
    }

    function api(name, payload) {
        state.apiCalls.push({ name, payload: payload || null, at: performance.now() });
    }

    function makeChat(title) {
        const id = "sim_chat_" + state.nextId++;
        const chat = {
            id,
            title: title || "Mac simulation stress",
            created_at: nowSeconds(),
            updated_at: nowSeconds(),
            messages: []
        };
        state.chats.unshift(chat);
        state.activeChatId = id;
        return chat;
    }

    function activeChat() {
        if (!state.activeChatId) return makeChat("Mac simulation stress");
        return state.chats.find(chat => chat.id === state.activeChatId) || makeChat("Mac simulation stress");
    }

    function assistantMessage(text, model) {
        const chat = activeChat();
        chat.messages.push({
            role: "assistant",
            content: text,
            model_used: model || "Orchestrator V1",
            created_at: nowSeconds()
        });
        chat.updated_at = nowSeconds();
        state.updates.push({
            type: "message",
            content: text,
            model_used: model || "Orchestrator V1",
            chat_id: chat.id
        });
    }

    function pushProgress(stage, percent, status) {
        state.updates.push({
            type: "progress",
            stage,
            label: stage,
            percent,
            branches: [
                { name: "plan", status: percent > 20 ? "done" : "running" },
                { name: "read-only macOS simulation scan", status: percent > 65 ? "done" : "running" },
                { name: "artifact/answer synthesis", status: status || (percent >= 100 ? "done" : "running") }
            ],
            chat_id: activeChat().id
        });
    }

    function queueNormalResponse(prompt, mode) {
        state.updates.push({ type: "status", content: "Simulated Orchestrator V1 is thinking...", chat_id: activeChat().id });
        setTimeout(() => pushProgress("Planning branch", 30), 60);
        setTimeout(() => pushProgress("Inspecting fake macOS state", 70), 120);
        setTimeout(() => {
            pushProgress("Grounding answer", 100, "done");
            assistantMessage(
                "Simulation complete. I inspected the isolated Mac environment, used the agentic branch for the request, and returned a grounded answer without touching the real laptop. Mode: " + mode + ". Prompt: " + prompt,
                "Orchestrator V1"
            );
        }, 190);
    }

    function queuePermission(prompt) {
        const id = "sim_perm_" + state.nextId++;
        state.pending[id] = prompt;
        state.updates.push({
            type: "permission_request",
            id,
            action: "simulated_delete_path",
            preview: "/stress_lab/mac_os_sim/Users/test/Downloads/old-installer.dmg",
            chat_id: activeChat().id
        });
    }

    makeChat("Mac simulation stress");

    window.__kiraStress = state;
    window.pywebview = {
        api: {
            warm_after_ui_ready: function () {
                api("warm_after_ui_ready");
                return Promise.resolve({ ok: true, simulated: true });
            },
            get_chats: function () {
                api("get_chats");
                return Promise.resolve(state.chats.map(chat => ({
                    id: chat.id,
                    title: chat.title,
                    created_at: chat.created_at,
                    updated_at: chat.updated_at
                })));
            },
            new_chat: function (title) {
                api("new_chat", title);
                return Promise.resolve(makeChat(title || "New Chat"));
            },
            load_chat: function (id) {
                api("load_chat", id);
                const chat = state.chats.find(item => item.id === id);
                if (!chat) return Promise.resolve({ error: "missing chat" });
                state.activeChatId = id;
                return Promise.resolve(chat);
            },
            delete_chat: function (id) {
                api("delete_chat", id);
                state.chats = state.chats.filter(item => item.id !== id);
                if (state.activeChatId === id) state.activeChatId = state.chats[0] ? state.chats[0].id : "";
                return Promise.resolve({ ok: true });
            },
            send_prompt: function (prompt, mode, selectedModel, chatId) {
                api("send_prompt", { prompt, mode, selectedModel, chatId });
                const chat = chatId ? state.chats.find(item => item.id === chatId) || activeChat() : activeChat();
                state.activeChatId = chat.id;
                chat.messages.push({ role: "user", content: String(prompt || ""), created_at: nowSeconds() });
                chat.title = chat.title === "New Chat" ? String(prompt || "New Chat").slice(0, 60) : chat.title;
                chat.updated_at = nowSeconds();
                state.prompts.push({ prompt, mode, selectedModel, chatId: chat.id, at: performance.now() });
                const lower = String(prompt || "").toLowerCase();
                if (lower.includes("delete") || lower.includes("move risky")) {
                    queuePermission(prompt);
                } else {
                    queueNormalResponse(String(prompt || ""), mode || "chat");
                }
                return Promise.resolve({ status: "queued", chat_id: chat.id });
            },
            poll_updates: function () {
                api("poll_updates");
                const updates = state.updates.splice(0);
                return Promise.resolve(updates);
            },
            toggle_voice_mode: function (enabled) {
                api("toggle_voice_mode", enabled);
                state.voiceEnabled = Boolean(enabled);
                return Promise.resolve("Voice mode " + (enabled ? "activated" : "deactivated"));
            },
            configure_voice_stack: function (stt, turn, tts, live) {
                api("configure_voice_stack", { stt, turn, tts, live });
                state.voiceStack = { stt, turn, tts, live };
                return Promise.resolve({ ok: true });
            },
            get_voice_status: function () {
                api("get_voice_status");
                return Promise.resolve({
                    voice_ready: false,
                    listen_ready: false,
                    stts_silent_test: true,
                    stt_provider: "browser",
                    turn_detector: "endpoint",
                    tts_provider: "kokoro_onnx",
                    realtime_stack: { stt_provider: "browser", turn_detector: "endpoint", tts_provider: "kokoro_onnx" }
                });
            },
            listen_once: function () {
                api("listen_once");
                return Promise.resolve({ ok: true, text: "Simulated voice prompt" });
            },
            start_voice_turn: function (source, live) {
                api("start_voice_turn", { source, live });
                return Promise.resolve({ ok: true });
            },
            record_voice_partial: function (text, source, confidence) {
                api("record_voice_partial", { text, source, confidence });
                return Promise.resolve({ ok: true });
            },
            complete_voice_turn: function (text, source, confidence, live) {
                api("complete_voice_turn", { text, source, confidence, live });
                return Promise.resolve({ ok: true });
            },
            stop_voice: function () {
                api("stop_voice");
                return Promise.resolve({ ok: true });
            },
            set_screen_overlay_state: function (stateName) {
                api("set_screen_overlay_state", stateName);
                state.overlayStates.push({ state: stateName, at: performance.now() });
                return Promise.resolve({ ok: true });
            },
            get_live_snapshot: function () {
                api("get_live_snapshot");
                return Promise.resolve({
                    time_label: "Simulated live snapshot",
                    front_app: "Finder Simulation",
                    current_brain: "orchestrator",
                    last_agentic_intent: state.prompts[state.prompts.length - 1] ? state.prompts[state.prompts.length - 1].mode : "idle",
                    queue_depth: state.updates.length,
                    pending_permissions: Object.keys(state.pending).length,
                    top_process: { name: "FakeRenderer", rss_mb: 312, mem_percent: "1.3" },
                    disk: { free: 512 * 1024 * 1024 * 1024, percent_used: 38 }
                });
            },
            get_kira_menu: function (query, force) {
                api("get_kira_menu", { query, force });
                return Promise.resolve({
                    updated_label: "from simulation",
                    sections: [
                        {
                            title: "Apps",
                            items: [
                                { title: "Finder Simulation", kind: "app", subtitle: "safe fake app", icon: "apps", open_target: "app:Finder Simulation" },
                                { title: "Terminal Simulation", kind: "app", subtitle: "safe fake terminal", icon: "terminal", open_target: "app:Terminal Simulation" }
                            ]
                        },
                        {
                            title: "Artifacts",
                            items: [
                                { title: "Stress Report PPTX", kind: "pptx", subtitle: "simulated generated file", icon: "slideshow", open_target: "file:stress-report.pptx" },
                                { title: "System Summary DOCX", kind: "docx", subtitle: "simulated generated file", icon: "description", open_target: "file:system-summary.docx" }
                            ]
                        },
                        {
                            title: "Commands",
                            items: [
                                { title: "Inspect fake Downloads", kind: "agentic", subtitle: "read-only", icon: "search", prompt: "Inspect the simulated Downloads folder read-only and summarize risks." },
                                { title: "Generate simulated PPTX", kind: "artifact", subtitle: "safe artifact", icon: "slideshow", prompt: "Generate a PPTX report using the simulated Mac state." }
                            ]
                        }
                    ]
                });
            },
            open_menu_target: function (target) {
                api("open_menu_target", target);
                return Promise.resolve("Opened simulated target: " + target);
            },
            get_slash_suggestions: function () {
                api("get_slash_suggestions");
                return Promise.resolve([
                    { command: "/open", title: "Open app/file/url", hint: "Ask what to open when unclear.", category: "system" },
                    { command: "/web", title: "Browse/read a web page", hint: "Fetch page content before answering.", category: "research" },
                    { command: "/pptx", title: "Generate a PPTX", hint: "Create a deck with images when tools are available.", category: "artifact" },
                    { command: "/optimize", title: "Read-only system optimization scan", hint: "Recommend before changing anything.", category: "system" }
                ]);
            },
            approve_pending_tool: function (id) {
                api("approve_pending_tool", id);
                delete state.pending[id];
                state.updates.push({ type: "status", content: "Running approved simulated action", chat_id: activeChat().id });
                setTimeout(() => assistantMessage("Approved action finished in the simulation. No real file was deleted.", "Agentic Tool"), 100);
                return Promise.resolve({ ok: true });
            },
            reject_pending_tool: function (id) {
                api("reject_pending_tool", id);
                delete state.pending[id];
                assistantMessage("Permission rejected. The simulated action was not run.", "Agentic Tool");
                return Promise.resolve({ ok: true });
            },
            shutdown_app: function () {
                api("shutdown_app");
                return Promise.resolve({ ok: true });
            }
        }
    };

    window.addEventListener("load", function () {
        state.loadAt = performance.now();
    });

    const observer = new MutationObserver(function () {
        if (!state.readyAt && document.documentElement.classList.contains("kira-boot-ready")) {
            state.readyAt = performance.now();
        }
    });

    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

    setTimeout(function () {
        window.dispatchEvent(new Event("pywebviewready"));
    }, 35);
})();
</script>
"""


def write_fake_macos_tree() -> None:
    files = {
        "Applications/Finder.app/Contents/Info.plist": "<plist><dict><key>CFBundleName</key><string>Finder Simulation</string></dict></plist>\n",
        "Applications/Terminal.app/Contents/Info.plist": "<plist><dict><key>CFBundleName</key><string>Terminal Simulation</string></dict></plist>\n",
        "Users/test/Downloads/notes.txt": "safe notes\n",
        "Users/test/Downloads/old-installer.dmg": "fake binary installer\n",
        "Users/test/Documents/project/README.md": "# Fake Project\n\nThis is safe simulation data.\n",
        "Users/test/Desktop/report-draft.pptx.txt": "placeholder deck metadata\n",
        "System/Library/CoreServices/SystemVersion.plist": "ProductName macOS Simulation\n",
    }
    for rel, content in files.items():
        path = SIM_ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def build_ui_sim() -> None:
    html = UI_HTML.read_text(encoding="utf-8")
    if "window.pywebview" not in FAKE_BRIDGE:
        raise RuntimeError("fake bridge missing pywebview shim")
    if "</head>" in html:
        html = html.replace("</head>", FAKE_BRIDGE + "\n</head>", 1)
    else:
        html = FAKE_BRIDGE + "\n" + html
    SIM_HTML.write_text(html, encoding="utf-8")

    assets_src = APP_DIR / "assets"
    assets_dst = STRESS_DIR / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, assets_dst, dirs_exist_ok=True)


def check_python_compile() -> list[dict]:
    results = []
    for name in REQUIRED_PYTHON_FILES:
        path = APP_DIR / name
        item = {"file": name, "exists": path.exists(), "ok": False, "error": ""}
        if not path.exists():
            item["error"] = "missing"
        else:
            try:
                ast.parse(path.read_text(encoding="utf-8"))
                item["ok"] = True
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
    return results


def extract_script_blocks(html: str) -> list[str]:
    scripts = []
    for match in re.finditer(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.IGNORECASE | re.DOTALL):
        code = match.group(1).strip()
        if code:
            scripts.append(code)
    return scripts


def check_javascript_syntax() -> dict:
    node = shutil.which("node")
    if not node:
        return {"ok": False, "error": "node not found", "scripts": 0}

    html = SIM_HTML.read_text(encoding="utf-8")
    scripts = extract_script_blocks(html)
    combined = "\n;\n".join(scripts)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as handle:
        handle.write(combined)
        temp_name = handle.name
    try:
        proc = subprocess.run([node, "--check", temp_name], capture_output=True, text=True, timeout=10)
        return {
            "ok": proc.returncode == 0,
            "scripts": len(scripts),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def check_ui_contract() -> dict:
    html = UI_HTML.read_text(encoding="utf-8")
    missing_ids = [item for item in REQUIRED_UI_IDS if f'id="{item}"' not in html]
    missing_modes = [mode for mode in ["agent", "vibe_coding"] if f'data-mode="{mode}"' not in html]
    return {
        "ok": not missing_ids and not missing_modes,
        "missing_ids": missing_ids,
        "missing_modes": missing_modes,
    }


def check_backend_contract() -> dict:
    interface_text = (APP_DIR / "interface.py").read_text(encoding="utf-8")
    actions_text = (APP_DIR / "actions.py").read_text(encoding="utf-8")
    combined = interface_text + "\n" + actions_text
    missing = [name for name in REQUIRED_API_METHODS if f"def {name}" not in combined]
    return {
        "ok": not missing,
        "missing_api_methods": missing,
    }


def check_dependencies() -> list[dict]:
    results = []
    for module, package in REQUIRED_PACKAGES.items():
        results.append({
            "module": module,
            "package": package,
            "available": importlib.util.find_spec(module) is not None,
        })
    return results


def check_model_paths() -> dict:
    orch_candidates = [
        APP_DIR / "orchestrator_v1_fused",
        APP_DIR / "models" / "orchestrator_v1_fused",
        Path.home() / "Desktop" / "Kira_OS" / "orchestrator_v1_fused",
    ]
    kira = APP_DIR / "kira_v1_fused"
    kira_models = APP_DIR / "models" / "kira_v1_fused"
    legacy_kira = APP_DIR / "kira_os_fused"
    legacy_kira_models = APP_DIR / "models" / "kira_os_fused"
    resolved_orch = next(
        (
            path for path in orch_candidates
            if path.exists()
            and (path / "config.json").exists()
            and (
                (path / "model.safetensors").exists()
                or (path / "model.safetensors.index.json").exists()
            )
        ),
        None
    )
    resolved_kira = kira if kira.exists() else (kira_models if kira_models.exists() else (legacy_kira if legacy_kira.exists() else legacy_kira_models))
    return {
        "ok": resolved_orch is not None,
        "searched_orchestrator_paths": [str(path) for path in orch_candidates],
        "resolved_orchestrator_path": str(resolved_orch) if resolved_orch else "",
        "orchestrator_files": sorted(path.name for path in resolved_orch.glob("*"))[:20] if resolved_orch else [],
        "kira_v1_fused": kira.exists(),
        "models_kira_v1_fused": kira_models.exists(),
        "kira_os_fused": legacy_kira.exists(),
        "models_kira_os_fused": legacy_kira_models.exists(),
        "resolved_kira_path": str(resolved_kira) if resolved_kira.exists() else "",
    }


def check_tree_and_council() -> dict:
    sys.path.insert(0, str(APP_DIR))
    from interface import KiraBrain

    brain = KiraBrain.__new__(KiraBrain)
    brain.active_chat_id = "stress-tree"
    brain.web_evidence_by_chat = {}
    brain.computer_roots = {}
    brain.council_parallel_enabled = True
    brain.council_role_timeout_seconds = 3
    brain._log_agentic_event = lambda *_args, **_kwargs: None

    real_file = STRESS_DIR / "tree-proof.pptx"
    real_file.write_bytes(b"KIRA TREE proof")
    real_artifact = f"PPTX generated: `{real_file}`"
    fake_artifact = "PPTX generated: `/tmp/kira-tree-missing-proof.pptx`"
    web_evidence = "WEB_RESEARCH:\nFetched pages: 2\nReal result links: https://example.com"
    opened = f"Opened: `{real_file}`"

    started = time.perf_counter()

    def simulated_role(role_name, _instruction):
        time.sleep(0.2)
        return {
            "role": role_name,
            "proposal": f"{role_name} evidence",
            "elapsed": 0.2,
            "status": "ok",
        }

    proposals = brain._run_council_roles_parallel(
        [("Pathfinder", "path"), ("Verifier", "verify")],
        simulated_role,
        "stress",
    )
    parallel_elapsed = time.perf_counter() - started

    checks = {
        "tree_registry_is_evidence": brain._has_raw_agentic_result("BRANCH_REGISTRY:\nversion: tree-3"),
        "tree_diagnostic_stays_direct": not brain._raw_agentic_result_needs_synthesis("/tree", "TREE_STATUS:\nready"),
        "natural_tree_result_needs_synthesis": brain._raw_agentic_result_needs_synthesis(
            "explain the tree", "BRANCH_REGISTRY:\nversion: tree-3"
        ),
        "real_artifact_is_proven": brain._agentic_goal_satisfied("create a pptx", real_artifact),
        "fake_artifact_is_rejected": not brain._agentic_goal_satisfied("create a pptx", fake_artifact),
        "cumulative_research_artifact_open": brain._agentic_goal_satisfied(
            "research anime, create a pptx, and open it",
            web_evidence + "\n\n" + real_artifact + "\n\n" + opened,
        ),
        "council_roles_returned": len(proposals) == 2 and all(item.get("status") == "ok" for item in proposals),
        "council_ran_in_parallel": parallel_elapsed < 0.36,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "parallel_elapsed_seconds": round(parallel_elapsed, 3),
        "proposals": proposals,
    }


def check_subagent_lifecycle() -> dict:
    sys.path.insert(0, str(APP_DIR))
    from actions import KiraActionsMixin

    class HarnessActions(KiraActionsMixin):
        def _log_agentic_event(self, *_args, **_kwargs):
            return None

    # Production deliberately rejects subagents discovered under stress/test
    # fixture paths. Keep this lifecycle fixture in an isolated temporary root
    # so the check exercises a valid production-shaped candidate.
    root = Path(tempfile.mkdtemp(prefix="kira-subagent-harness-"))
    model_dir = root / "harness_mlx_candidate"
    workspace = root / "workspace"
    logs = root / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text('{"model_type":"harness"}\n', encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    # Model discovery intentionally ignores tiny placeholder directories. A
    # sparse file keeps this fixture cheap while matching a real MLX layout.
    with open(model_dir / "model.safetensors", "wb") as model_file:
        model_file.seek((1024 * 1024) + 16)
        model_file.write(b"\0")

    runner = HarnessActions()
    runner.app_root = str(APP_DIR)
    runner.home_path = str(root)
    runner.agentic_workspace = str(workspace)
    runner.agentic_logs_path = str(logs)
    runner.architect_workspace = str(root / "architect")
    runner.subagent_workspace = str(root / "subagents")
    runner.computer_roots = {"harness": str(root)}
    runner.readable_roots = [str(root)]
    runner.sensitive_path_markers = []
    runner.last_agentic_intent = None
    runner.last_agentic_result = None

    scan_result = runner._scan_local_models_tool(
        f"ROOT: {root}\nMAX_GB: 1"
    )
    create_result = runner._create_subagent_tool(
        "NAME: harness_specialist\n"
        "PURPOSE: Verify isolated subagent lifecycle.\n"
        f"MODEL_PATH: {model_dir}\n"
        "BACKEND: mlx\n"
        "SYSTEM_PROMPT:\nReturn evidence only."
    )
    queue_result = runner._subagent_task_tool(
        "AGENT: harness_specialist\nTASK: Inspect this safe packet.\nRUN: false",
        "stress-subagent",
    )
    run_result = runner._subagent_task_tool(
        "AGENT: harness_specialist\nTASK: Inspect this safe packet.\nRUN: true\nTIMEOUT: 5",
        "stress-subagent",
    )
    registry = runner._load_subagent_registry()
    task_files = list((Path(runner.subagent_workspace) / "tasks").glob("*.json"))
    runtime_file = Path(runner.subagent_workspace) / "subagent_runtime.py"
    runtime_spec = importlib.util.spec_from_file_location("kira_subagent_runtime_harness", runtime_file)
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime_module)

    checks = {
        "model_scan_found_candidate": str(model_dir) in scan_result,
        "subagent_created": "Created: `harness_specialist`" in create_result,
        "registry_written": any(item.get("name") == "harness_specialist" for item in registry.get("subagents", [])),
        "task_packet_queued": "Queued packet" in queue_result and bool(task_files),
        "invalid_model_not_claimed_success": run_result.startswith("SUBAGENT_TASK ERROR:"),
        "runtime_present": runtime_file.exists(),
        "thought_channel_stripped": runtime_module.public_output("<|channel>thought private<channel|>PUBLIC") == "PUBLIC",
        "unfinished_thought_rejected": runtime_module.public_output("<|channel>thought private only") == "",
        "thought_process_stripped": runtime_module.public_output("<thought_process>private</thought_process>PUBLIC") == "PUBLIC",
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "scan_result": scan_result,
        "create_result": create_result,
        "queue_result": queue_result,
        "run_result": run_result,
    }


def check_specialist_hub() -> dict:
    sys.path.insert(0, str(APP_DIR))
    from actions import KiraActionsMixin

    class HarnessActions(KiraActionsMixin):
        def _log_agentic_event(self, *_args, **_kwargs):
            return None

    root = STRESS_DIR / "specialist_harness"
    root.mkdir(parents=True, exist_ok=True)
    runner = HarnessActions()
    runner.app_root = str(root)
    runner.home_path = str(root)
    runner.agentic_workspace = str(root / "workspace")
    runner.agentic_logs_path = str(root / "logs")
    runner.architect_workspace = str(root / "architect")
    runner.computer_roots = {"harness": str(root)}
    runner.readable_roots = [str(root)]
    runner.sensitive_path_markers = []
    runner.web_evidence_by_chat = {}
    runner.last_agentic_intent = None
    runner.last_agentic_result = None
    Path(runner.agentic_workspace).mkdir(parents=True, exist_ok=True)
    Path(runner.agentic_logs_path).mkdir(parents=True, exist_ok=True)

    media_a = root / "agentic-system-interface-product.png"
    media_b = root / "agentic-system-interface-tree.png"
    shutil.copy2(APP_DIR / "demo" / "KIRA_OS_Product_Demo_Poster.png", media_a)
    shutil.copy2(APP_DIR / "demo" / "KIRA_OS_Specialist_TREE_Demo_Poster.png", media_b)

    def delayed_research(_query, max_pages="5"):
        time.sleep(0.2)
        return (
            "WEB_RESEARCH:\nFetched pages: 2\n"
            "Source: https://example.com/agentic-systems\n"
            "Readable text:\nverified evidence"
        )

    def delayed_media(_query, limit="5"):
        time.sleep(0.2)
        return (
            "WEB_IMAGE_SEARCH:\nDownloaded images: 2\n"
            f"- Path: `{media_a}`\n"
            "  Source: https://example.com/agentic-system-interface-product.png\n"
            "  Page: https://example.com/agentic-systems\n"
            "  Alt: Agentic system interface product view\n"
            f"- Path: `{media_b}`\n"
            "  Source: https://example.com/agentic-system-interface-tree.png\n"
            "  Page: https://example.com/agentic-systems\n"
            "  Alt: Agentic system interface Tree view"
        )

    runner._web_research_tool = delayed_research
    runner._web_image_search_tool = delayed_media
    blocks = (
        "[SPECIALIST_TASK]\nAGENT: research\nQUERY: agentic systems\n[/SPECIALIST_TASK]\n"
        "[SPECIALIST_TASK]\nAGENT: media\nQUERY: agentic system interface\n[/SPECIALIST_TASK]"
    )
    started = time.perf_counter()
    result = runner._run_specialist_tools(blocks, "stress-specialists")
    elapsed = time.perf_counter() - started
    registry = runner._specialist_hub().registry()
    checks = {
        "fixed_agents_registered": len(registry.get("agents", [])) == 5,
        "slides_agent_registered": any(item.get("id") == "slides" for item in registry.get("agents", [])),
        "artifact_verifier_registered": any(
            item.get("id") == "artifact_verifier" for item in registry.get("agents", [])
        ),
        "parallel_execution": elapsed < 0.36,
        "both_results_verified": result.count("Status: verified") == 2,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "parallel_elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    started = time.perf_counter()
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    write_fake_macos_tree()
    build_ui_sim()

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app_dir": str(APP_DIR),
        "stress_dir": str(STRESS_DIR),
        "sim_html": str(SIM_HTML),
        "fake_macos_root": str(SIM_ROOT),
        "checks": {
            "python_compile": check_python_compile(),
            "javascript_syntax": check_javascript_syntax(),
            "ui_contract": check_ui_contract(),
            "backend_contract": check_backend_contract(),
            "dependencies": check_dependencies(),
            "model_paths": check_model_paths(),
            "tree_and_council": check_tree_and_council(),
            "subagent_lifecycle": check_subagent_lifecycle(),
            "specialist_hub": check_specialist_hub(),
        },
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    mandatory_ok = (
        all(item.get("ok") for item in report["checks"]["python_compile"])
        and report["checks"]["javascript_syntax"].get("ok", False)
        and report["checks"]["ui_contract"].get("ok", False)
        and report["checks"]["backend_contract"].get("ok", False)
        and all(item.get("available") for item in report["checks"]["dependencies"])
        and report["checks"]["model_paths"].get("ok", False)
        and report["checks"]["tree_and_council"].get("ok", False)
        and report["checks"]["subagent_lifecycle"].get("ok", False)
        and report["checks"]["specialist_hub"].get("ok", False)
    )
    report["ok"] = mandatory_ok
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "ok": mandatory_ok,
        "report": str(REPORT_JSON),
        "sim_html": str(SIM_HTML),
        "fake_macos_root": str(SIM_ROOT),
    }, indent=2))
    return 0 if mandatory_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
