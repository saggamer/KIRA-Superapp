import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_coding_mode import CodingHarness


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.host = CodingHarness(self.tmp.name)
        self.project = Path(self.tmp.name) / "project"
        self.project.mkdir()
        self.profile = {"provider": "nvidia_nim", "model": self.host.CODING_NEMOTRON_MODEL,
                        "api_key": "test-key", "base_url": "https://integrate.api.nvidia.com/v1"}
        self.plan = {"summary": "Implement", "files": [{"path": "app.js", "content": "const x = 1;"}],
                     "commands": [{"command": "node --check app.js"}]}

    def prepare(self):
        return self.host._coding_prepare_plan(self.profile, "system", "task", str(self.project), "hybrid", "kira_hybrid")

    def events(self):
        path = Path(self.host.coding_runs_path) / "hybrid.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_single_agent_proposes_without_writing_and_records_file_evidence(self):
        with patch.object(self.host, "_coding_supervisor_decision", return_value={"mode": "single"}), \
             patch.object(self.host, "_coding_call_provider", return_value=json.dumps(self.plan)) as worker:
            plan = self.prepare()
        self.assertEqual(plan["files"][0]["path"], "app.js")
        self.assertFalse((self.project / "app.js").exists())
        self.assertEqual(worker.call_args.kwargs["max_tokens"], 16384)
        evidence = next(e["detail"]["evidence"] for e in self.events() if e["phase"] == "worker_observed")
        self.assertEqual(len(evidence["files"][0]["sha256"]), 64)
        saved = next(e["detail"]["path"] for e in self.events() if e["phase"] == "proposal_saved")
        self.assertEqual(json.loads(Path(saved).read_text())["files"][0]["content"], "const x = 1;")

    def test_reviewer_sees_actual_code_and_rejection_is_repaired_once(self):
        responses = [self.plan, {"approved": False, "findings": ["Fix undefined variable"]},
                     self.plan, {"approved": True, "findings": []}]
        with patch.object(self.host, "_coding_supervisor_decision", return_value={"mode": "coder_reviewer"}), \
             patch.object(self.host, "_coding_call_provider", side_effect=[json.dumps(r) for r in responses]) as worker:
            self.prepare()
        self.assertEqual(worker.call_count, 4)
        self.assertIn("const x = 1", worker.call_args_list[1].args[2])
        self.assertIn("undefined variable", worker.call_args_list[2].args[2])

    def test_review_budget_exhaustion_stops_before_apply(self):
        responses = [self.plan, {"approved": False, "findings": ["broken"]}] * 2
        with patch.object(self.host, "_coding_supervisor_decision", return_value={"mode": "coder_reviewer"}), \
             patch.object(self.host, "_coding_call_provider", side_effect=[json.dumps(r) for r in responses]), \
             self.assertRaisesRegex(ValueError, "bounded repair"):
            self.prepare()
        self.assertEqual(list(self.project.iterdir()), [])

    def test_missing_local_model_is_reported_as_fallback(self):
        with patch.object(self.host, "_coding_supervisor_decision", side_effect=RuntimeError("missing weights")), \
             patch.object(self.host, "_coding_call_provider", return_value=json.dumps(self.plan)):
            self.prepare()
        self.assertTrue(any(e["phase"] == "supervisor_unavailable" for e in self.events()))
        self.assertFalse(any(e["phase"] == "supervisor_decision" for e in self.events()))

    def test_bad_supervisor_route_never_runs_worker(self):
        with patch.object(self.host, "_coding_supervisor_decision", return_value={"mode": "unlimited"}), \
             patch.object(self.host, "_coding_call_provider") as worker, self.assertRaises(ValueError):
            self.prepare()
        worker.assert_not_called()

    def test_model_monitor_cannot_override_failed_check(self):
        payload = {**self.plan, "project_path": str(self.project), "harness": "kira_hybrid", "run_id": "hybrid"}
        with patch.object(self.host, "_coding_execute_project_command", return_value={"ok": False, "exit_code": 1}), \
             patch.object(self.host, "_coding_supervisor_decision", return_value={"assessment": "success"}):
            result = self.host._execute_coding_apply(payload)
        self.assertIn("VERIFICATION FAILED", result)
        self.assertFalse((self.project / "app.js").exists())
        self.assertFalse(any(e["phase"] == "completed" for e in self.events()))

    def test_supervision_loads_v15_explicitly(self):
        with patch.object(self.host, "_evict_and_load", create=True) as load, \
             patch.object(self.host, "_tokenizer_prompt", create=True, side_effect=lambda messages: str(messages)), \
             patch.object(self.host, "_generate_with_watchdog", create=True, return_value='{"mode":"single"}'), \
             patch.object(self.host, "_split_orchestrator_response", create=True, side_effect=lambda text: ("", text)):
            result = self.host._coding_supervisor_decision("Choose", {})
        load.assert_called_once_with("orchestrator_v15")
        self.assertEqual(result["mode"], "single")

    def test_stream_ignores_reasoning_and_retains_usage(self):
        chunks = [{"choices": []}, {"choices": [{"delta": {"reasoning_content": "private"}}]},
                  {"choices": [{"delta": {"content": '{"files":[]}'}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 100}}]
        response = io.BytesIO(("\n".join("data: " + json.dumps(c) for c in chunks) + "\ndata: [DONE]\n").encode())
        result = self.host._coding_collect_stream(response)
        self.assertEqual(result["choices"][0]["message"]["content"], '{"files":[]}')
        self.assertEqual(result["usage"]["completion_tokens"], 100)
        self.assertNotIn("private", json.dumps(result))

    def test_truncated_or_interrupted_stream_cannot_be_a_plan(self):
        for finish in (None, "length", "tool_calls"):
            response = io.BytesIO(('data: ' + json.dumps({"choices": [{"delta": {"content": "{}"}, "finish_reason": finish}]}) + '\n').encode())
            with self.assertRaisesRegex(RuntimeError, "finish successfully"):
                self.host._coding_collect_stream(response)

    def test_nvidia_request_matches_supplied_model_options(self):
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()
        response = Response(b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n')
        with patch("coding.urllib.request.urlopen", return_value=response) as send:
            self.host._coding_call_provider(self.profile, "system", "task", max_tokens=16384)
        request = json.loads(send.call_args.args[0].data)
        self.assertEqual(request["model"], self.host.CODING_NEMOTRON_MODEL)
        self.assertTrue(request["stream"])
        self.assertTrue(request["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(request["max_tokens"], 16384)
        self.assertEqual(request["temperature"], 1)
        self.assertEqual(request["seed"], 42)
        self.assertNotIn("thinking_token_budget", request)

    def test_upstream_deepseek_overlay_routes_nvidia_without_saving_credential(self):
        overlay = self.host._coding_deepseek_patch(self.profile)
        encoded = json.dumps(overlay)
        self.assertNotIn("test-key", encoded)
        route = overlay[0]["config"]["providers"]["kira-provider"]
        self.assertEqual(route["baseURL"], self.profile["base_url"])
        self.assertEqual(route["models"][0]["id"], self.profile["model"])
        self.assertTrue(route["compat"]["chatTemplateKwargs"]["enable_thinking"])

    def test_hybrid_uses_real_deepseek_runtime_when_installed(self):
        with patch.object(self.host, "_coding_harness_path", return_value="/fake/dsh"), \
             patch.object(self.host, "_coding_supervisor_decision", return_value={"mode": "single"}), \
             patch.object(self.host, "_coding_external_plan", return_value=json.dumps(self.plan)) as upstream:
            self.prepare()
        self.assertEqual(upstream.call_args.args[0], "deepseek")

    def test_task_cap_counts_prior_worker_calls(self):
        self.host.save_coding_budget("nvidia_nim", 100, 0.01, 1, 1, 0, "hard_stop")
        profile = dict(self.profile, _task_spend_usd=0.009)
        with self.assertRaisesRegex(RuntimeError, "per-task cap"):
            self.host._coding_budget_guard(profile, "system", "task", 4096)

    def test_saved_proposal_is_revalidated_and_reviewed_without_regeneration(self):
        from orchestrator_v15.coding_harness import CodingHarnessSupervisor
        with patch.object(self.host, "_coding_supervisor_decision", return_value={"mode": "coder_reviewer"}), \
             patch.object(self.host, "_coding_external_plan") as upstream, \
             patch.object(self.host, "_coding_call_provider", return_value='{"approved":true,"findings":[]}') as review:
            CodingHarnessSupervisor(self.host, "hybrid").prepare(
                self.profile, "system", "task", str(self.project), self.plan)
        upstream.assert_not_called()
        self.assertEqual(review.call_count, 1)
        self.assertEqual(review.call_args.kwargs["max_tokens"], 16384)

    def test_resume_cannot_bypass_path_validation(self):
        from orchestrator_v15.coding_harness import CodingHarnessSupervisor
        bad = {"files": [{"path": "../escape.js", "content": "bad"}]}
        with patch.object(self.host, "_coding_supervisor_decision", return_value={"mode": "single"}), \
             self.assertRaisesRegex(ValueError, "unsafe coding path"):
            CodingHarnessSupervisor(self.host, "hybrid").prepare(
                self.profile, "system", "task", str(self.project), bad)

    def test_upstream_error_never_returns_reasoning_excerpt(self):
        result = type("Result", (), {"returncode": 1, "stderr": "dsh: reasoning:\nprivate deliberation\n", "stdout": ""})()
        with patch.object(self.host, "_coding_harness_path", return_value="/fake/dsh"), \
             patch("coding.subprocess.run", return_value=result):
            with self.assertRaises(RuntimeError) as error:
                self.host._coding_external_plan("deepseek", self.profile, "system", "task", str(self.project))
        self.assertIn("status 1", str(error.exception))
        self.assertNotIn("private deliberation", str(error.exception))

    def test_generated_tests_are_run_when_worker_only_proposes_syntax_checks(self):
        plan = {"files": [{"path": "game.test.cjs", "content": "const test = require('node:test');"}],
                "commands": [{"command": "node --check game.test.cjs"}]}
        with patch.object(self.host, "_coding_call_provider", return_value=json.dumps(plan)):
            checked = self.host._coding_prepare_plan(self.profile, "system", "task", str(self.project), "hybrid")
        self.assertIn("node --test game.test.cjs", [c["command"] for c in checked["commands"]])

    def test_improvement_review_receives_full_source_and_observations(self):
        from orchestrator_v15.coding_harness import CodingHarnessSupervisor
        source = 'project source ' * 1000 + 'END_OF_SOURCE'
        verdict = {'worker_brief': 'Add a clear collection effect.', 'improvements': ['collection feedback']}
        with patch.object(self.host, '_coding_project_context', return_value=source), \
             patch.object(self.host, '_coding_supervisor_decision', return_value=verdict) as decide:
            decision, context = CodingHarnessSupervisor(self.host, 'hybrid').improvement_brief(
                str(self.project), 'improve game', {'tests_passed': 8})
        self.assertEqual(context, source)
        self.assertEqual(decide.call_args.args[1]['project_source'], source)
        self.assertEqual(decide.call_args.args[1]['observations']['tests_passed'], 8)
        self.assertEqual(decision, verdict)
        self.assertIn('No image pixels', decide.call_args.args[1]['visual_input'])

    def test_improvement_direction_cannot_be_faked_when_v15_is_unavailable(self):
        from orchestrator_v15.coding_harness import CodingHarnessSupervisor
        with patch.object(self.host, '_coding_project_context', return_value='source'), \
             patch.object(self.host, '_coding_supervisor_decision', side_effect=RuntimeError('unavailable')), \
             self.assertRaisesRegex(ValueError, 'actionable improvement brief'):
            CodingHarnessSupervisor(self.host, 'hybrid').improvement_brief(str(self.project), 'improve game', {})

    def test_iteration_review_receives_updated_source_and_fresh_results(self):
        from orchestrator_v15.coding_harness import CodingHarnessSupervisor
        verdict = {'improvement_completed': True, 'next_action': 'stop', 'assessment': 'Motion cue implemented'}
        with patch.object(self.host, '_coding_project_context', return_value='updated source'), \
             patch.object(self.host, '_coding_supervisor_decision', return_value=verdict) as decide:
            result = CodingHarnessSupervisor(self.host, 'hybrid').review_iteration(
                str(self.project), {'worker_brief': 'Add a motion trail'}, {'browser_verified': True, 'tests_passed': 8})
        self.assertEqual(result, verdict)
        self.assertEqual(decide.call_args.args[1]['updated_project_source'], 'updated source')
        self.assertTrue(decide.call_args.args[1]['observations']['browser_verified'])


if __name__ == "__main__":
    unittest.main()
