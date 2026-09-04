import inspect
import os
import queue
import time
import unittest

from interface import KiraBrain


APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return _Encoding([ord(char) for char in str(text)])

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class _LiveProcess:
    returncode = None

    def poll(self):
        return None


class OrchestratorRuntimeStabilityTests(unittest.TestCase):
    def _brain(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.app_root = APP_ROOT
        brain.orch_path = os.path.join(APP_ROOT, "orchestrator_v1_fused")
        brain.prompt_tokenizer = _CharacterTokenizer()
        brain.model_worker_process = _LiveProcess()
        brain.current_brain = "orchestrator"
        brain.model_worker_generation_timeout = 90
        brain.model_worker_max_prompt_tokens = 1800
        brain.model_worker_retry_prompt_tokens = 1200
        brain.model_worker_max_output_tokens = 1100
        brain.model_worker_retry_output_tokens = 700
        brain.model_worker_restart_limit = 2
        brain.model_worker_restart_window_seconds = 120
        brain.model_worker_restart_times = []
        brain.response_queue = queue.Queue()
        brain.events = []
        brain.evictions = []
        brain._mark_model_worker_dead = lambda reason: brain.evictions.append(reason)
        brain._log_agentic_event = lambda name, payload: brain.events.append((name, payload))
        brain._truncate = lambda text, limit=1000: str(text)[: int(limit)]
        return brain

    def test_long_prompt_is_compacted_with_recent_context_preserved(self):
        brain = self._brain()
        prompt = "<bos>" + ("old-context " * 400) + "LATEST VERIFIED RESULT<|turn>model\n"
        compacted = brain._compact_model_prompt(prompt, 900)

        self.assertLessEqual(brain._prompt_token_count(compacted), 930)
        self.assertIn("Older runtime context compacted", compacted)
        self.assertIn("LATEST VERIFIED RESULT", compacted)
        self.assertTrue(compacted.endswith("<|turn>model\n"))
        self.assertTrue(any(name == "model_prompt_compacted" for name, _ in brain.events))

    def test_generation_caps_output_and_recovers_once_with_smaller_context(self):
        brain = self._brain()
        requests = []
        load_calls = []

        def request(payload, timeout=60):
            requests.append(dict(payload))
            if len(requests) == 1:
                raise RuntimeError("Model worker timed out after 90s during generate.")
            return "recovered answer"

        brain._model_worker_request = request
        brain._evict_and_load = lambda target: load_calls.append(target)

        answer = brain._generate_sync("x" * 5000, max_tokens=9000)

        self.assertEqual(answer, "recovered answer")
        self.assertEqual(load_calls, ["orchestrator"])
        self.assertEqual(len(brain.evictions), 1)
        self.assertEqual(len(requests), 2)
        self.assertLessEqual(requests[0]["max_tokens"], 1100)
        self.assertLessEqual(requests[1]["max_tokens"], 700)
        self.assertLessEqual(brain._prompt_token_count(requests[0]["prompt"]), 1830)
        self.assertLessEqual(brain._prompt_token_count(requests[1]["prompt"]), 1230)
        status = brain.response_queue.get_nowait()
        self.assertIn("resuming", status["content"].lower())

    def test_nonrecoverable_generation_error_is_not_retried(self):
        brain = self._brain()
        calls = []

        def request(payload, timeout=60):
            calls.append(payload)
            raise RuntimeError("Template field is malformed")

        brain._model_worker_request = request
        brain._evict_and_load = lambda _target: self.fail("must not restart")

        with self.assertRaisesRegex(RuntimeError, "Template field"):
            brain._generate_sync("short prompt")
        self.assertEqual(len(calls), 1)
        self.assertEqual(brain.evictions, [])

    def test_live_worker_memory_failure_is_evicted_before_reload(self):
        brain = self._brain()
        order = []
        brain._mark_model_worker_dead = lambda reason: order.append("evict")
        brain._evict_and_load = lambda target: order.append("load")

        def request(payload, timeout=60):
            if not order:
                raise RuntimeError("Metal memory allocation failed")
            order.append("generate")
            return "ready"

        brain._model_worker_request = request
        self.assertEqual(brain._generate_sync("short prompt"), "ready")
        self.assertEqual(order, ["evict", "load", "generate"])

    def test_restart_circuit_breaker_blocks_restart_storms(self):
        brain = self._brain()
        brain.model_worker_restart_limit = 2
        self.assertTrue(brain._reserve_model_worker_restart())
        self.assertTrue(brain._reserve_model_worker_restart())
        self.assertFalse(brain._reserve_model_worker_restart())

        brain.model_worker_restart_times = [time.time() - 500]
        self.assertTrue(brain._reserve_model_worker_restart())

    def test_external_worker_has_orchestrator_stop_tokens_and_one_generate_call(self):
        worker_path = os.path.join(APP_ROOT, "model_worker.py")
        with open(worker_path, "r", encoding="utf-8") as handle:
            source = handle.read()

        compile(source, worker_path, "exec")
        self.assertIn("RUNTIME_STOP_TOKEN_IDS = {1, 49, 51, 106}", source)
        self.assertEqual(source.count("content = generate("), 1)
        self.assertNotIn("except Exception:\n                content = generate(", source)

    def test_ui_process_launches_external_worker_instead_of_inline_code(self):
        source = inspect.getsource(KiraBrain._start_model_worker)
        self.assertIn('"model_worker.py"', source)
        self.assertIn("worker_path", source)
        self.assertNotIn("-c", source)


if __name__ == "__main__":
    unittest.main()
