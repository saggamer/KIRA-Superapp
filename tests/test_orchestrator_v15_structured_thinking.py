import json
import queue
import sys
import types
import unittest
from unittest.mock import patch

if "webview" not in sys.modules:
    sys.modules["webview"] = types.SimpleNamespace()

from interface import KiraBrain
from orchestrator_v15.structured_thinking import parse_memo, memo_prompt
from orchestrator_v15.thinking_controller import ThinkingController
from orchestrator_v15.diagonal_level import generate_diagonal


class StructuredThinkingTests(unittest.TestCase):
    def test_only_compact_fixed_schema_is_accepted(self):
        memo = {key: key for key in ("goal", "next_step", "alternative", "verification", "uncertainty")}
        self.assertEqual(parse_memo(json.dumps(memo)), memo)
        self.assertIsNone(parse_memo("I should click the button"))
        self.assertIsNone(parse_memo(json.dumps({**memo, "secret": "x"})))
        self.assertIsNone(parse_memo(json.dumps({**memo, "goal": "x" * 241})))
        self.assertIn("not evidence", memo_prompt(memo))

    def test_diagonal_generation_uses_one_memo_and_three_proposals(self):
        memo = {key: "inspect observed state" for key in
                ("goal", "next_step", "alternative", "verification", "uncertainty")}
        action = "[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]"
        answers = iter([json.dumps(memo), action, action, action])
        prompts = []
        fake_mlx = types.ModuleType("mlx_lm")
        fake_mlx.generate = lambda _model, _tokenizer, *, prompt, **_kwargs: (
            prompts.append(prompt) or next(answers))
        fake_utils = types.ModuleType("mlx_lm.sample_utils")
        fake_utils.make_sampler = lambda **_kwargs: None

        class Tokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                return "\n".join(item["content"] for item in messages)

            def encode(self, text, add_special_tokens=False):
                return list(text)

        controller = ThinkingController("Open https://example.test", diagonal_level=True)
        with patch.dict(sys.modules, {"mlx_lm": fake_mlx, "mlx_lm.sample_utils": fake_utils}):
            chosen, audit = generate_diagonal(None, Tokenizer(),
                [{"role": "user", "content": "Open https://example.test"}], controller,
                structured_thinking=True)
        self.assertEqual(chosen, action)
        self.assertEqual(len(prompts), 4)
        self.assertIn("Unverified model-generated decision memo", prompts[1])
        self.assertTrue(audit["memo_valid"])
        self.assertEqual(audit["model_generations"], 4)

    def _brain(self, model_reply, tool_action=None):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _name: None
        brain._load_chat_messages = lambda _chat: []
        brain._truncate = lambda text, length=1000: str(text)[:length]
        brain._log_agentic_event = lambda *_: None
        brain._extract_agentic_blocks = lambda answer: answer if "[BROWSER_TOOL]" in answer else ""
        brain._model_worker_request = model_reply
        brain._run_agentic_capabilities_with_watchdog = tool_action or (lambda *args: "")
        return brain

    def test_preview_requests_model_deliberation_and_keeps_one_executed_action(self):
        calls = []
        executions = []
        navigate = "[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]"
        observe = "[BROWSER_TOOL]\nACTION: observe\nTABID: tab-1\n[/BROWSER_TOOL]"

        def reply(payload, timeout=60):
            calls.append(payload)
            if len(calls) == 1:
                return {"proposals": [observe, navigate, observe], "memo_tokens": 40,
                        "token_counts": [20, 20, 20], "memo": {"goal": "open"}}
            return {"proposals": ["The title is Example."] * 3,
                    "memo_tokens": 35, "token_counts": [16, 16, 16], "memo": None}

        def execute(block, *_args):
            executions.append(block)
            return block + '\nBROWSER_TOOL:\n{"tabId":"tab-1","observationId":"o1","text":"Example"}'

        brain = self._brain(reply, execute)
        result = brain._run_v15_profile("Open https://example.test and tell me the title.",
                                        "chat-1", "diagonal", structured_thinking=True)
        self.assertEqual(result, "The title is Example.")
        self.assertEqual(executions, [navigate])
        self.assertTrue(all(call["structured_thinking"] for call in calls))

    def test_unapproved_purchase_never_reaches_tool_handler(self):
        calls = []
        executions = []
        navigate = "[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]"
        purchase = ("[BROWSER_TOOL]\nACTION: act\nOPERATION: click\n"
                    "TARGET_NAME: Submit purchase\nTABID: tab-1\n[/BROWSER_TOOL]")

        def reply(payload, timeout=60):
            calls.append(payload)
            return {"proposals": [navigate] if len(calls) == 1 else [purchase],
                    "memo_tokens": 8, "token_counts": [15], "memo": None}

        def execute(block, *_args):
            executions.append(block)
            return block + '\nBROWSER_TOOL:\n{"tabId":"tab-1","observationId":"o1","text":"Checkout"}'

        brain = self._brain(reply, execute)
        result = brain._run_v15_profile("Open https://example.test; do not submit purchase.",
                                        "chat-1", "low", structured_thinking=True)
        self.assertIn("invalid browser proposals", result)
        self.assertEqual(executions, [navigate])

    def test_generated_token_cap_stops_before_extra_action(self):
        calls = []
        executions = []
        navigate = "[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]"

        def reply(payload, timeout=60):
            calls.append(payload)
            return {"proposals": [navigate], "memo_tokens": 128,
                    "token_counts": [1408], "memo": None}

        def execute(block, *_args):
            executions.append(block)
            return block + '\nBROWSER_TOOL:\n{"tabId":"tab-1","observationId":"o1","text":"Example"}'

        brain = self._brain(reply, execute)
        result = brain._run_v15_profile("Open https://example.test and inspect it.",
                                        "chat-1", "low", structured_thinking=True)
        self.assertIn("generated-token limit", result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(executions, [navigate])


if __name__ == "__main__":
    unittest.main()
