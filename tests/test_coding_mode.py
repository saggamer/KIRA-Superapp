import json
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from coding import KiraCodingMixin


class CodingHarness(KiraCodingMixin):
    def __init__(self, root):
        self.test_root = os.path.realpath(root)
        self.architect_workspace = os.path.join(root, "architect")
        self.response_queue = queue.Queue()
        self.pending_permissions = {}
        self._active_permission_context = None
        self.keys = {}
        self.queued = None
        self.command_calls = []
        self.write_count = 0
        self.fail_write_number = None
        self._init_coding_mode()

    def _coding_project_is_sensitive(self, path):
        real = os.path.realpath(path)
        if real == self.test_root or real.startswith(self.test_root + os.sep):
            return False
        return super()._coding_project_is_sensitive(path)

    def _coding_store_key(self, provider, api_key):
        self.keys[provider] = api_key
        return True

    def _coding_load_key(self, provider):
        key = self.keys.get(provider, "")
        return (key, "keychain") if key else ("", "missing")

    def _coding_delete_key(self, provider):
        self.keys.pop(provider, None)

    def discover_coding_ides(self):
        return [{"id": "Cursor", "label": "Cursor", "app_name": "Cursor", "path": "/Applications/Cursor.app"}]

    def _queue_permission(self, action, payload, preview, chat_id=None):
        self.queued = {"action": action, "payload": payload, "preview": preview, "chat_id": chat_id}
        return "Permission required: `perm_coding_test`"

    def _coding_orchestrator_brief(self, task, session, project_context):
        return "Implement the requested change and preserve the existing project style."

    def _write_file_with_seatbelt(self, path, content, append=False):
        self.write_count += 1
        if self.fail_write_number == self.write_count:
            return "WRITE FAILED: simulated transaction failure"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(content)
        return "WRITE OK"

    def _run_project_command(self, payload):
        self.command_calls.append(payload)
        return "COMMAND OK: " + payload["command"]

    def _coding_execute_project_command(self, payload):
        self.command_calls.append(payload)
        return {
            "ok": True,
            "execution_id": "exec_test",
            "status": "completed",
            "command": payload["command"],
            "cwd": payload["cwd"],
            "duration_ms": 4,
            "exit_code": 0,
            "stdout": "COMMAND OK",
            "stderr": "",
            "output_truncated": False,
            "seatbelt": "test",
        }


class CodingModeTests(unittest.TestCase):
    def test_all_builtin_providers_expose_default_base_urls(self):
        with tempfile.TemporaryDirectory() as root:
            settings = CodingHarness(root).get_coding_settings()

        urls = {item["id"]: item["base_url"] for item in settings["providers"]}
        self.assertEqual(urls["openai"], "https://api.openai.com/v1")
        self.assertEqual(urls["anthropic"], "https://api.anthropic.com")
        self.assertEqual(urls["gemini"], "https://generativelanguage.googleapis.com")
        self.assertEqual(urls["deepseek"], "https://api.deepseek.com")
        self.assertEqual(urls["nvidia_nim"], "https://integrate.api.nvidia.com/v1")
        self.assertEqual(urls["openrouter"], "https://openrouter.ai/api/v1")

    def test_nvidia_nim_lists_models_with_its_own_api_key(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"data": [
                    {"id": "nvidia/nemotron-3.5-lightning-30b-a3b"},
                    {"id": "z-ai/glm-5.2"},
                ]}).encode("utf-8")

        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            with patch("coding.urllib.request.urlopen", return_value=Response()) as urlopen:
                result = brain.list_coding_models("nvidia_nim", "nvapi-secret", "")
            request = urlopen.call_args.args[0]
            settings = brain.get_coding_settings()

        self.assertTrue(result["ok"])
        self.assertEqual(request.full_url, "https://integrate.api.nvidia.com/v1/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer nvapi-secret")
        self.assertEqual(len(result["models"]), 2)
        nim = next(item for item in settings["providers"] if item["id"] == "nvidia_nim")
        self.assertEqual(nim["label"], "NVIDIA NIM")
        self.assertIn("z-ai/glm-5.2", nim["models"])

    def test_provider_secret_is_never_written_to_settings(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            result = brain.save_coding_provider("openai", "gpt-test", "secret-value", "")
            with open(brain.coding_settings_path, "r", encoding="utf-8") as handle:
                settings_text = handle.read()

        self.assertTrue(result["ok"])
        self.assertNotIn("secret-value", settings_text)
        self.assertIn('"model": "gpt-test"', settings_text)

    def test_model_library_password_gates_key_reveal(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            brain.save_coding_provider("openai", "gpt-used", "secret-value", "")
            with brain.coding_lock:
                settings = brain._coding_read_json(brain.coding_settings_path, {})
                settings["providers"]["openai"]["models"] = ["gpt-used", "unused-catalog-model"]
                brain._coding_write_json(brain.coding_settings_path, settings)

            before = brain.get_model_library()
            blocked = brain.reveal_model_library_key("openai")
            password = brain.set_key_view_password("correct-horse")
            revealed = brain.reveal_model_library_key("openai")
            brain.lock_model_library()
            wrong = brain.unlock_model_library("wrong-password")
            with open(brain.coding_settings_path, "r", encoding="utf-8") as handle:
                settings_text = handle.read()

        self.assertFalse(before["password_set"])
        self.assertTrue(blocked["locked"])
        self.assertTrue(password["ok"])
        self.assertEqual(revealed["api_key"], "secret-value")
        self.assertFalse(wrong["ok"])
        self.assertEqual(before["models"][0]["models"], ["gpt-used"])
        self.assertNotIn("secret-value", settings_text)
        self.assertNotIn("correct-horse", settings_text)

    def test_model_library_rejects_short_password(self):
        with tempfile.TemporaryDirectory() as root:
            result = CodingHarness(root).set_key_view_password("short")
        self.assertFalse(result["ok"])
        self.assertIn("8 characters", result["error"])

    def test_codex_harness_runs_read_only_and_returns_validated_plan(self):
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            brain.app_root = os.path.dirname(os.path.dirname(__file__))
            brain.keys["openai"] = "openai-secret"
            brain.discover_coding_harnesses = lambda: [
                {"id": "codex", "label": "OpenAI Codex", "available": True, "path": "/usr/bin/codex"}
            ]
            project = os.path.join(root, "project")
            os.makedirs(project)
            calls = []

            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                output = command[command.index("--output-last-message") + 1]
                with open(output, "w", encoding="utf-8") as handle:
                    json.dump({
                        "summary": "Add file", "answer": "Prepared", "notes": [],
                        "files": [{"path": "index.html", "content": "<h1>KIRA</h1>", "reason": "Requested"}],
                        "commands": [],
                    }, handle)
                return Result()

            profile = brain._coding_provider_profile("openai", "gpt-test")
            with patch("coding.subprocess.run", side_effect=fake_run):
                plan = brain._coding_prepare_plan(profile, "system", "task", project, "run-test", "codex")

        command, kwargs = calls[0]
        self.assertIn("read-only", command)
        self.assertIn("--ephemeral", command)
        self.assertEqual(kwargs["env"]["OPENAI_API_KEY"], "openai-secret")
        self.assertEqual(plan["files"][0]["path"], "index.html")

    def test_provider_update_preserves_money_cap(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            brain.save_coding_provider("openai", "gpt-one", "secret-value", "")
            brain.save_coding_budget("openai", 25, 2, 3, 12, 1.5, "hard_stop")
            brain.save_coding_provider("openai", "gpt-two", "", "")
            budget = brain.get_coding_budget("openai")

        self.assertEqual(budget["monthly_cap_usd"], 25)
        self.assertEqual(budget["task_cap_usd"], 2)
        self.assertEqual(budget["starting_spend_usd"], 1.5)

    def test_usage_ledger_calculates_provider_cost(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            brain.save_coding_budget("openai", 10, 0, 2, 4, 0.25, "hard_stop")
            brain._coding_record_usage(
                {"provider": "openai", "model": "gpt-test"},
                input_tokens=1000,
                output_tokens=500,
            )
            budget = brain.get_coding_budget("openai")

        self.assertAlmostEqual(budget["tracked_api_spend_usd"], 0.004)
        self.assertAlmostEqual(budget["total_spend_usd"], 0.254)
        self.assertEqual(budget["request_count"], 1)

    def test_hard_stop_blocks_projected_task_overage(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            brain.save_coding_budget("openai", 10, 0.001, 100, 100, 0, "hard_stop")
            profile = {"provider": "openai", "model": "gpt-test"}

            with self.assertRaisesRegex(RuntimeError, "per-task cap"):
                brain._coding_budget_guard(profile, "system", "task", max_tokens=100)

    def test_finish_current_policy_blocks_the_next_call_after_crossing_cap(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            brain.save_coding_budget("openai", 0.001, 0, 10, 10, 0, "finish_current")
            profile = {"provider": "openai", "model": "gpt-test"}
            guard = brain._coding_budget_guard(profile, "system", "task", max_tokens=10000)
            brain._coding_record_usage(profile, input_tokens=100, output_tokens=100)

            self.assertEqual(guard["policy"], "finish_current")
            with self.assertRaisesRegex(RuntimeError, "reached"):
                brain._coding_budget_guard(profile, "system", "another task", max_tokens=10)

    def test_reset_usage_keeps_user_entered_starting_spend(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            brain.save_coding_budget("openai", 10, 0, 1, 1, 2.5, "hard_stop")
            profile = {"provider": "openai", "model": "gpt-test"}
            brain._coding_record_usage(profile, input_tokens=1000, output_tokens=1000)
            budget = brain.reset_coding_usage("openai")

        self.assertEqual(budget["request_count"], 0)
        self.assertEqual(budget["tracked_api_spend_usd"], 0)
        self.assertEqual(budget["total_spend_usd"], 2.5)

    def test_plan_rejects_traversal_secret_paths_and_destructive_commands(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = os.path.join(root, "project")
            os.makedirs(project)

            with self.assertRaisesRegex(ValueError, "unsafe coding path"):
                brain._coding_validate_plan(project, {"files": [{"path": "../outside.py", "content": "x"}], "commands": []})
            with self.assertRaisesRegex(ValueError, "unsafe coding path"):
                brain._coding_validate_plan(project, {"files": [{"path": ".env", "content": "TOKEN=x"}], "commands": []})
            with self.assertRaisesRegex(ValueError, "destructive coding command"):
                brain._coding_validate_plan(project, {"files": [], "commands": [{"command": "rm -rf build"}]})

    def test_project_context_excludes_secrets_and_dependencies(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            (project / "src").mkdir(parents=True)
            (project / "node_modules" / "pkg").mkdir(parents=True)
            (project / "src" / "app.py").write_text("print('visible')", encoding="utf-8")
            (project / ".env").write_text("TOP_SECRET=hidden", encoding="utf-8")
            (project / "node_modules" / "pkg" / "index.js").write_text("hidden dependency", encoding="utf-8")
            context = brain._coding_project_context(str(project))

        self.assertIn("visible", context)
        self.assertNotIn("TOP_SECRET", context)
        self.assertNotIn("hidden dependency", context)

    def test_project_context_prioritizes_repository_instructions_and_task_files(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            (project / "src" / "payments").mkdir(parents=True)
            (project / "AGENTS.md").write_text("Always run the focused payment test.", encoding="utf-8")
            (project / "src" / "payments" / "checkout.py").write_text("def checkout(): pass\n", encoding="utf-8")
            for index in range(110):
                (project / f"misc_{index}.txt").write_text("unrelated\n", encoding="utf-8")

            context = brain._coding_project_context(
                str(project), task="fix the payments checkout function"
            )

        self.assertIn("Always run the focused payment test", context)
        self.assertIn("src/payments/checkout.py", context)
        self.assertLess(context.index("AGENTS.md"), context.index("misc_0.txt"))

    def test_invalid_plan_gets_one_bounded_repair(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            invalid = json.dumps({
                "summary": "bad", "files": [{"path": "../escape.py", "content": "x"}], "commands": [],
            })
            repaired = json.dumps({
                "summary": "fixed", "files": [{"path": "app.py", "content": "VALUE = 1\n"}], "commands": [],
            })
            with patch.object(brain, "_coding_call_provider", side_effect=[invalid, repaired]) as provider:
                plan = brain._coding_prepare_plan(
                    {"provider": "openai", "model": "test"}, "system", "task", str(project), "code_test"
                )

            run_path = Path(brain.coding_runs_path) / "code_test.jsonl"
            events = [json.loads(line)["phase"] for line in run_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(provider.call_count, 2)
        self.assertEqual(plan["files"][0]["path"], "app.py")
        self.assertIn("plan_rejected", events)
        self.assertIn("planned", events)

    def test_missing_python_check_is_inferred(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            checks = brain._coding_infer_checks(
                str(project), [{"path": "src/app.py", "content": "print('ok')\n"}]
            )

        self.assertEqual(len(checks), 1)
        self.assertIn("python3 -m compileall -q src/app.py", checks[0]["command"])

    def test_plan_rejects_install_and_formatter_commands(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            for command in ("npm install left-pad", "black src"):
                with self.assertRaisesRegex(ValueError, "non-verification"):
                    brain._coding_validate_plan(
                        str(project), {"files": [], "commands": [{"command": command}]}
                    )

    def test_vibe_planning_queues_permission_without_writing(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            target = project / "app.py"
            target.write_text("print('old')\n", encoding="utf-8")
            brain.save_coding_provider("openai", "gpt-test", "secret-value", "")
            brain.configure_coding_session("chat-1", str(project), "Cursor", "openai", "gpt-test")
            provider_plan = json.dumps({
                "summary": "Update greeting",
                "answer": "Prepared the greeting update.",
                "files": [{"path": "app.py", "content": "print('new')\n", "reason": "requested"}],
                "commands": [{"command": "python -m compileall app.py", "reason": "syntax check"}],
            })

            with patch.object(brain, "_coding_call_provider", return_value=provider_plan):
                answer = brain._handle_vibe_coding_prompt("update the greeting", "chat-1")

            self.assertEqual(target.read_text(encoding="utf-8"), "print('old')\n")
            self.assertEqual(brain.queued["action"], "coding_apply")
            self.assertEqual(brain.queued["payload"]["files"][0]["path"], "app.py")
            self.assertIn("perm_coding_test", answer)
            self.assertIn("Nothing has changed yet", answer)

    def test_approved_plan_writes_verifies_runs_checks_and_opens_ide(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            payload = {
                "project_path": str(project),
                "ide": "Cursor",
                "summary": "Create module",
                "files": [{"path": "src/new.py", "content": "VALUE = 1\n", "reason": "requested"}],
                "commands": [{"command": "python -m compileall src", "reason": "verify"}],
                "chat_id": "chat-1",
            }
            opened = type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            with patch("coding.subprocess.run", return_value=opened) as run:
                result = brain._execute_coding_apply(payload)

            self.assertEqual((project / "src" / "new.py").read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertIn("sha256:", result)
            self.assertIn("Opened `src/new.py` in Cursor with the project", result)
            self.assertEqual(
                run.call_args.args[0],
                [
                    "open", "-a", "Cursor", os.path.realpath(project),
                    os.path.realpath(project / "src" / "new.py"),
                ],
            )
            self.assertEqual(brain.command_calls[0]["command"], "python -m compileall src")

    def test_failed_transaction_rolls_back_existing_and_new_files(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            existing = project / "existing.py"
            existing.write_text("ORIGINAL\n", encoding="utf-8")
            brain.fail_write_number = 2
            payload = {
                "project_path": str(project),
                "summary": "Two edits",
                "files": [
                    {"path": "existing.py", "content": "CHANGED\n", "reason": "first"},
                    {"path": "new.py", "content": "PARTIAL\n", "reason": "second"},
                ],
                "commands": [],
            }

            result = brain._execute_coding_apply(payload)

            self.assertIn("Rollback: complete", result)
            self.assertEqual(existing.read_text(encoding="utf-8"), "ORIGINAL\n")
            self.assertFalse((project / "new.py").exists())

    def test_failed_verification_rolls_back_and_does_not_open_ide(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            target = project / "app.py"
            target.write_text("ORIGINAL\n", encoding="utf-8")
            payload = {
                "project_path": str(project),
                "ide": "Cursor",
                "summary": "Change app",
                "files": [{"path": "app.py", "content": "BROKEN\n", "reason": "test"}],
                "commands": [{"command": "python3 -m compileall -q app.py", "reason": "verify"}],
                "run_id": "code_verify_failure",
            }
            failed = {
                "ok": False, "execution_id": "exec_failed", "status": "failed",
                "command": "python3 -m compileall -q app.py", "cwd": str(project),
                "duration_ms": 3, "exit_code": 1, "stdout": "", "stderr": "syntax error",
                "output_truncated": False, "seatbelt": "test",
            }

            with patch.object(brain, "_coding_execute_project_command", return_value=failed), patch(
                "coding.subprocess.run"
            ) as open_app:
                result = brain._execute_coding_apply(payload)

            self.assertIn("CODING VERIFICATION FAILED", result)
            self.assertIn("Rollback: complete", result)
            self.assertEqual(target.read_text(encoding="utf-8"), "ORIGINAL\n")
            open_app.assert_not_called()

    def test_ide_discovery_only_returns_installed_apps(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            applications = Path(root) / "Applications"
            (applications / "Cursor.app").mkdir(parents=True)
            brain.coding_application_roots = [str(applications)]

            ides = KiraCodingMixin.discover_coding_ides(brain)

        self.assertEqual([item["id"] for item in ides], ["Cursor"])

    def test_coding_command_returns_structured_lifecycle_and_caps_output(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            brain.CODING_COMMAND_OUTPUT_LIMIT_BYTES = 4096

            result = KiraCodingMixin._coding_execute_project_command(brain, {
                "cwd": str(project),
                "command": "python3 -c \"print('x' * 12000)\"",
                "timeout": 10,
            })

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["execution_id"].startswith("exec_"))
        self.assertTrue(result["output_truncated"])
        self.assertLessEqual(len(result["stdout"].encode("utf-8")), 4096)
        self.assertGreaterEqual(result["duration_ms"], 0)

    def test_coding_command_timeout_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            started = time.monotonic()

            result = KiraCodingMixin._coding_execute_project_command(brain, {
                "cwd": str(project),
                "command": "python3 -c \"import time; time.sleep(20)\"",
                "timeout": 1,
            })

        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(result["exit_code"], 124)
        self.assertLess(time.monotonic() - started, 5)

    def test_coding_command_can_be_cancelled_by_execution_id(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            result_box = {}

            def run_command():
                result_box["result"] = KiraCodingMixin._coding_execute_project_command(brain, {
                    "cwd": str(project),
                    "command": "python3 -c \"import time; time.sleep(20)\"",
                    "timeout": 30,
                })

            worker = threading.Thread(target=run_command)
            worker.start()
            deadline = time.monotonic() + 3
            execution_id = ""
            while time.monotonic() < deadline and not execution_id:
                with brain.coding_execution_lock:
                    execution_id = next(iter(brain.coding_active_executions), "")
                time.sleep(0.02)
            cancel = brain.cancel_coding_execution(execution_id)
            worker.join(timeout=5)

        self.assertTrue(cancel["ok"])
        self.assertFalse(worker.is_alive())
        self.assertEqual(result_box["result"]["status"], "cancelled")
        self.assertEqual(result_box["result"]["exit_code"], 130)

    def test_coding_command_labels_seatbelt_unavailable_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            brain = CodingHarness(root)
            project = Path(root) / "project"
            project.mkdir()
            brain._seatbelt_available = lambda: True
            brain._seatbelt_profile = lambda **_kwargs: "(version 1)"
            calls = []

            def fake_run(runner, cwd, timeout, execution_id, seatbelt_mode):
                calls.append({
                    "runner": runner,
                    "cwd": cwd,
                    "timeout": timeout,
                    "execution_id": execution_id,
                    "seatbelt": seatbelt_mode,
                })
                if len(calls) == 1:
                    return {
                        "ok": False,
                        "execution_id": execution_id,
                        "status": "failed",
                        "exit_code": 71,
                        "stdout": "",
                        "stderr": "sandbox-exec: sandbox_apply: Operation not permitted",
                        "seatbelt": seatbelt_mode,
                    }
                return {
                    "ok": True,
                    "execution_id": execution_id,
                    "status": "completed",
                    "exit_code": 0,
                    "stdout": "verified",
                    "stderr": "",
                    "seatbelt": seatbelt_mode,
                }

            with patch.object(brain, "_coding_run_bounded_process", side_effect=fake_run):
                result = KiraCodingMixin._coding_execute_project_command(brain, {
                    "cwd": str(project),
                    "command": "python3 -m compileall .",
                    "timeout": 10,
                })

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["seatbelt"], "project")
        self.assertEqual(calls[1]["seatbelt"], "fallback_unavailable")
        self.assertEqual(calls[0]["execution_id"], calls[1]["execution_id"])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["seatbelt_fallback"])
        self.assertIn("sandbox_apply", result["seatbelt_fallback_reason"])


if __name__ == "__main__":
    unittest.main()
