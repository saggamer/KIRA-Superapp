import json
import base64
import ast
import os
import queue
import re
import tempfile
import unittest

from interface import KiraBrain
from memory import SmartMemoryStore


class StabilizationTests(unittest.TestCase):
    def test_kira_live_short_chat_is_not_replaced_by_fixed_greeting(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "interface.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('target_brain != "kira"', source)

    def test_attachment_import_is_scoped_and_extractable(self):
        brain = KiraBrain.__new__(KiraBrain)
        with tempfile.TemporaryDirectory() as root:
            brain.attachments_workspace = os.path.join(root, "attachments")
            os.makedirs(brain.attachments_workspace)
            result = brain.import_attachment(
                "../notes.md",
                base64.b64encode(b"Grounded attachment evidence.").decode("ascii"),
                "text/markdown",
                "chat-1",
            )
            normalized = brain._normalize_task_attachments([result], "chat-1")
            context = brain._build_attachment_context(normalized)

        self.assertTrue(result["ok"])
        self.assertEqual(result["name"], "notes.md")
        self.assertEqual(len(normalized), 1)
        self.assertIn("Grounded attachment evidence.", context)
        self.assertIn("trusted as user-provided evidence", context)

    def test_attachment_rejects_unsupported_executable(self):
        brain = KiraBrain.__new__(KiraBrain)
        with tempfile.TemporaryDirectory() as root:
            brain.attachments_workspace = os.path.join(root, "attachments")
            os.makedirs(brain.attachments_workspace)
            result = brain.import_attachment(
                "payload.app",
                base64.b64encode(b"not an app").decode("ascii"),
                "application/octet-stream",
                "chat-1",
            )

        self.assertFalse(result["ok"])
        self.assertIn("Unsupported", result["error"])

    def test_pdf_attachment_extracts_text_for_model_context(self):
        import fitz

        brain = KiraBrain.__new__(KiraBrain)
        with tempfile.TemporaryDirectory() as root:
            brain.attachments_workspace = os.path.join(root, "attachments")
            os.makedirs(brain.attachments_workspace)
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "KIRA attached PDF evidence")
            payload = document.tobytes()
            document.close()
            result = brain.import_attachment(
                "evidence.pdf",
                base64.b64encode(payload).decode("ascii"),
                "application/pdf",
                "chat-pdf",
            )
            normalized = brain._normalize_task_attachments([result], "chat-pdf")
            context = brain._build_attachment_context(normalized)

        self.assertTrue(result["ok"])
        self.assertIn("KIRA attached PDF evidence", context)

    def test_model_discovery_skips_placeholder_directory(self):
        brain = KiraBrain.__new__(KiraBrain)
        with tempfile.TemporaryDirectory() as root:
            placeholder = os.path.join(root, "placeholder")
            valid = os.path.join(root, "valid")
            os.makedirs(placeholder)
            os.makedirs(valid)
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                with open(os.path.join(valid, name), "wb") as handle:
                    handle.write(b"{}")

            selected = brain._first_existing_path([placeholder, valid])
            self.assertEqual(selected, valid)

    def test_worker_marks_failed_task_done(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.is_running = True
        brain.task_queue = queue.Queue()
        brain.response_queue = queue.Queue()

        def fail_once(_task):
            brain.is_running = False
            raise RuntimeError("expected failure")

        brain._route_and_generate = fail_once
        brain.task_queue.put({"prompt": "smoke"})
        brain._worker_loop()

        self.assertEqual(brain.task_queue.unfinished_tasks, 0)
        self.assertEqual(brain.response_queue.get_nowait()["type"], "error")

    def test_screen_overlay_is_forced_off(self):
        brain = KiraBrain.__new__(KiraBrain)
        with tempfile.TemporaryDirectory() as root:
            brain.screen_overlay_state_path = os.path.join(root, "overlay.json")
            brain.screen_overlay_process = None
            result = brain.set_screen_overlay_state("agentic")
            with open(brain.screen_overlay_state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)

        self.assertTrue(result["ok"])
        self.assertFalse(result["feature_enabled"])
        self.assertEqual(state["state"], "off")

    def test_kira_live_test_console_has_guarded_launch_control(self):
        ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui.html")
        with open(ui_path, "r", encoding="utf-8") as handle:
            markup = handle.read()

        self.assertIn('id="live-toggle"', markup)
        self.assertIn('id="live-readiness"', markup)
        self.assertIn('id="voice-listen-btn"', markup)
        self.assertIn("disabled", markup)
        self.assertIn("No fallback was started", markup)

    def test_atomic_json_write_replaces_complete_document(self):
        brain = KiraBrain.__new__(KiraBrain)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "state.json")
            brain._atomic_write_json(path, {"version": 1})
            brain._atomic_write_json(path, {"version": 2, "ready": True})
            with open(path, "r", encoding="utf-8") as handle:
                state = json.load(handle)

        self.assertEqual(state, {"version": 2, "ready": True})

    def test_untagged_private_reasoning_keeps_only_public_tail(self):
        brain = KiraBrain.__new__(KiraBrain)
        leaked = (
            'The user has input "bro" as the current task. '
            'My instructions are to maintain the persona of Orchestrator V1. '
            'I need to respond neutrally. '
            '1. Analyze input: informal address. '
            '2. Determine goal: keep the conversation flowing. '
            '3. Formulate response: ask for the task. '
            "I will maintain the calm persona.I'm here. What can I assist you with today?"
        )

        self.assertEqual(
            brain._strip_private_reasoning(leaked),
            "I'm here. What can I assist you with today?",
        )

    def test_identity_response_does_not_leak_private_reasoning(self):
        brain = KiraBrain.__new__(KiraBrain)
        leaked = (
            'The user is asking for my identity. I need to respond according to the persona instructions, '
            'which dictate my identity is "Orchestrator V1." I must maintain the calm, capable tone and '
            'provide a direct answer. I must not mention backend processes. The instruction set also includes '
            'a personalization element, but it is not relevant.You are chatting with Orchestrator V1.'
        )

        self.assertEqual(
            brain._strip_private_reasoning(leaked),
            "You are chatting with Orchestrator V1.",
        )

    def test_saved_chat_is_sanitized_when_loaded(self):
        brain = KiraBrain.__new__(KiraBrain)
        leaked = (
            "The user is asking for my identity. I need to follow persona instructions. "
            "I must not mention backend processes.You are chatting with Orchestrator V1."
        )
        with tempfile.TemporaryDirectory() as root:
            brain.chat_history_path = root
            path = os.path.join(root, "chat-1.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"id": "chat-1", "messages": [{"role": "assistant", "content": leaked}]}, handle)
            chat = brain._load_chat("chat-1")

        self.assertEqual(
            chat["messages"][0]["content"],
            "You are chatting with Orchestrator V1.",
        )

    def test_model_candidates_include_common_macos_locations(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.app_root = "/Applications/KIRA Superapp"
        candidates = brain._orchestrator_model_candidates()

        self.assertIn("/Applications/KIRA Superapp/models/orchestrator_v1_fused", candidates)
        self.assertTrue(any(path.endswith("Desktop/orchestrator v1 fused") for path in candidates))
        self.assertFalse(any("gemma-4-E4B" in path for path in candidates))

    def test_normal_explanation_is_not_trimmed_as_private_reasoning(self):
        brain = KiraBrain.__new__(KiraBrain)
        answer = (
            "Math is the study of quantities, structures, patterns, and change. "
            "It provides a language for describing relationships and solving problems."
        )

        self.assertEqual(brain._strip_private_reasoning(answer), answer)

    def test_display_name_is_local_editable_and_sanitized(self):
        brain = KiraBrain.__new__(KiraBrain)
        state = {"user": {"known_name": "", "display_name": ""}}
        brain._load_personalization_rag = lambda: state
        brain._save_personalization_rag = lambda data: state.update(data)

        saved = brain.set_user_display_name("  Ada   <Lovelace>  ")
        loaded = brain.get_user_profile()

        self.assertTrue(saved["ok"])
        self.assertEqual(saved["display_name"], "Ada Lovelace")
        self.assertEqual(loaded["greeting"], "Hello, Ada Lovelace.")

    def test_ui_uses_editable_greeting_empty_state(self):
        ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui.html")
        with open(ui_path, "r", encoding="utf-8") as handle:
            markup = handle.read()

        self.assertIn("Hello, apply your name.", markup)
        self.assertIn('id="profile-settings-modal"', markup)
        self.assertNotIn("Hello, Dhanvanth.", markup)

    def test_superapp_ui_exposes_only_chat_and_code_workspaces(self):
        ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui.html")
        with open(ui_path, "r", encoding="utf-8") as handle:
            markup = handle.read()

        self.assertIn("<title>KIRA Superapp</title>", markup)
        self.assertIn("KIRA SUPERAPP", markup)
        self.assertIn('data-mode="agent"', markup)
        self.assertIn('data-mode="vibe_coding"', markup)
        self.assertIn("let currentMode = 'agent';", markup)
        self.assertEqual(markup.count('class="mode-btn'), 2)
        self.assertNotIn('data-mode="chat"', markup)
        self.assertNotIn("Basic Chat", markup)
        self.assertNotIn("Agentic RAG", markup)

    def test_settings_are_owned_by_sidebar_profile_menu(self):
        ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui.html")
        with open(ui_path, "r", encoding="utf-8") as handle:
            markup = handle.read()

        self.assertIn('id="profile-menu-toggle"', markup)
        self.assertIn('id="profile-account-menu"', markup)
        self.assertIn("openSettingsFromProfile('providers')", markup)
        self.assertNotIn('id="settings-toggle"', markup)

    def test_coding_api_models_use_provider_dropdowns(self):
        ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui.html")
        with open(ui_path, "r", encoding="utf-8") as handle:
            markup = handle.read()

        self.assertIn('<select id="coding-model-input"', markup)
        self.assertIn('<select id="coding-settings-model"', markup)
        self.assertIn("list_coding_models", markup)
        self.assertNotIn('placeholder="Enter the exact provider model ID"', markup)

    def test_settings_open_before_backend_hydration(self):
        ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui.html")
        with open(ui_path, "r", encoding="utf-8") as handle:
            markup = handle.read()

        start = markup.index("function toggleSettings(open")
        end = markup.index("function setSettingsTab", start)
        settings_function = markup[start:end]
        show_index = settings_function.index("codingSettingsModal.classList.toggle('flex', visible)")
        hydrate_index = settings_function.index("hydrateSettingsModal")
        self.assertLess(show_index, hydrate_index)

    def test_every_ui_bridge_call_exists_on_lazy_api(self):
        app_root = os.path.dirname(os.path.dirname(__file__))
        ui_path = os.path.join(app_root, "ui.html")
        interface_path = os.path.join(app_root, "interface.py")
        with open(ui_path, "r", encoding="utf-8") as handle:
            markup = handle.read()
        with open(interface_path, "r", encoding="utf-8") as handle:
            module = ast.parse(handle.read())

        ui_calls = set(re.findall(r"window\.pywebview\.api\.([A-Za-z_]\w*)", markup))
        lazy_api = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "LazyKiraAPI"
        )
        backend_methods = {
            node.name for node in lazy_api.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertTrue(ui_calls)
        self.assertEqual(sorted(ui_calls - backend_methods), [])

    def test_superapp_glass_theme_is_neutral(self):
        ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui.html")
        with open(ui_path, "r", encoding="utf-8") as handle:
            markup = handle.read()

        self.assertIn("--glass-surface: rgba(20, 20, 21, 0.58)", markup)
        self.assertIn("backdrop-filter: blur(32px)", markup)
        self.assertNotIn("--accent: #9bdaf1", markup)
        self.assertNotIn("rgba(75, 132, 155, 0.14)", markup)

    def test_smart_memory_is_relevant_and_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            store = SmartMemoryStore(os.path.join(root, "memory.json"))
            store.remember_exchange(
                "chat-slides",
                "I prefer concise slide decks with sources.",
                "Understood. I will keep decks concise and sourced.",
            )
            store.remember_exchange(
                "chat-other",
                "I like dark editor themes.",
                "Understood.",
            )
            context = store.build_context(
                "Create a sourced slide deck",
                chat_id="chat-new",
                max_chars=700,
            )

        self.assertIn("concise slide decks with sources", context)
        self.assertLessEqual(len(context), 700)

    def test_smart_memory_never_persists_secrets(self):
        secret = "sk-" + "example-super-secret-token-123456789"
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "memory.json")
            store = SmartMemoryStore(path)
            store.remember_exchange(
                "chat-secret",
                f"My API key is {secret}",
                f"I received {secret}",
            )
            with open(path, "r", encoding="utf-8") as handle:
                stored = handle.read()
            context = store.build_context("What did I say before?", chat_id="chat-secret")

        self.assertNotIn(secret, stored)
        self.assertNotIn(secret, context)
        self.assertIn("sensitive content omitted", context)

    def test_chat_boundary_captures_exchange_and_redacts_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            brain = KiraBrain.__new__(KiraBrain)
            brain.chat_history_path = os.path.join(root, "chats")
            os.makedirs(brain.chat_history_path, exist_ok=True)
            brain.smart_memory = SmartMemoryStore(os.path.join(root, "memory.json"))
            brain.active_chat_id = "memory-chat"

            brain._append_chat_message(
                "memory-chat", "user", "Prefer concise release summaries.", "Orchestrator V1"
            )
            brain._append_chat_message(
                "memory-chat", "assistant", "Understood. I will keep them concise.", "Orchestrator V1"
            )
            context = brain._smart_memory_context("release summary", "memory-chat")
            self.assertIn("Prefer concise release summaries", context)

            brain.smart_memory = None
            brain._append_chat_message(
                "memory-chat", "user", "My API key is sk-test-secret", "Orchestrator V1"
            )
            fallback = brain._compact_chat_fallback("memory-chat")
            self.assertNotIn("sk-test-secret", fallback)
            self.assertIn("sensitive content omitted", fallback)

    def test_personalization_context_excludes_quick_pathway_database(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain._load_personalization_rag = lambda: {
            "user": {"display_name": "Ada"},
            "stable_preferences": ["Prefers concise sourced slide decks."],
            "project_interests": ["Local agentic workflows"],
            "recent_user_signals": [],
            "learned_quick_pathways": [{"example": "DO NOT INJECT " * 500}],
        }

        context = brain._compact_personalization_context("Make a sourced slide deck", max_chars=700)

        self.assertIn("Preferred name: Ada", context)
        self.assertIn("concise sourced slide decks", context)
        self.assertNotIn("DO NOT INJECT", context)
        self.assertLessEqual(len(context), 700)


if __name__ == "__main__":
    unittest.main()
