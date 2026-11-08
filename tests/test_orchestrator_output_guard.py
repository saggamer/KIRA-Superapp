import queue
import sys
import types
import unittest

from orchestrator_v15.fast_mlx import generate_closed_proposal
from orchestrator_v15.output_guard import output_issue


BROKEN = '[SUBAGENT_TASK]' + 'APP_ACTION:' * 126 + 'APP_'


class OutputGuardTests(unittest.TestCase):
    def test_reported_output_and_truncated_protocol_are_rejected(self):
        self.assertEqual(output_issue(BROKEN), 'repeated_protocol')
        for text in ('[SUBAGENT_TASK', '[SUBAGENT_TASK]APP_ACTION: open',
                     'Here is the result: [BROWSER_TOOL]\nACTION: observe',
                     '[/SUBAGENT_TASK]'):
            self.assertEqual(output_issue(text), 'malformed_action')

    def test_ordinary_answers_code_and_valid_single_actions_remain_usable(self):
        for text in ('A variable stores a value.', 'Use APP_ACTION as a label in your example.',
                     '```python\nfor i in range(10):\n    print(i)\n```',
                     '[OPEN]\nTARGET: Music\n[/OPEN]', '[APP_LIST]\n[/APP_LIST]',
                     '[NATIVE_PDF]\nTITLE: Python\nCONTENT:\nprint() displays text.\n[/NATIVE_PDF]'):
            self.assertIsNone(output_issue(text), text)
        self.assertEqual(output_issue('[OPEN]\nTARGET: Music\n[/OPEN]', allow_tools=False), 'unexpected_action')

    def test_stream_aborts_repeating_labels_before_spending_full_token_budget(self):
        seen = []
        closed = []
        def stream(*_args, **_kwargs):
            try:
                for part in ['[SUBAGENT_TASK]'] + ['APP_ACTION:'] * 126:
                    seen.append(part)
                    yield types.SimpleNamespace(text=part)
            finally:
                closed.append(True)
        text = generate_closed_proposal(None, None, 'prompt', max_tokens=512,
                                        sampler=None, stream_generate_fn=stream)
        self.assertEqual(output_issue(text), 'repeated_protocol')
        self.assertLess(len(seen), 10)
        self.assertTrue(closed)

    def brain(self, responses):
        sys.modules.setdefault('webview', types.SimpleNamespace())
        from interface import KiraBrain
        brain = KiraBrain.__new__(KiraBrain)
        brain.v15_model_tool_planning = False
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda *_: None
        brain._load_chat_messages = lambda *_: [
            {'role': 'user', 'content': 'Why did you stop?'},
            {'role': 'assistant', 'content': BROKEN}]
        brain._smart_memory_context = lambda *_a, **_k: ''
        brain._truncate = lambda text, length=1000: str(text)[:length]
        brain._log_agentic_event = lambda *_a, **_k: None
        captured, executed = [], []
        answers = iter(responses)
        def model(payload, **_kwargs):
            captured.append(payload)
            return {'proposals': [next(answers)]}
        brain._model_worker_request = model
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_: executed.append(block) or 'unexpected execution'
        return brain, captured, executed

    def test_malformed_control_output_is_retried_and_never_executed_or_reused_as_context(self):
        for malformed in (BROKEN, '[SUBAGENT_TASK]APP_ACTION: explain'):
            brain, captured, executed = self.brain([malformed, 'The earlier response was empty, so it could not finish.'])
            answer = brain._run_v15_profile('Why did you stop?', 'output-guard', 'high')
            self.assertIn('earlier response was empty', answer)
            self.assertEqual(executed, [])
            self.assertEqual(len(captured), 2)
            for payload in captured:
                self.assertFalse(any(BROKEN in item['content'] for item in payload['messages']))
            self.assertNotIn('APP_ACTION:', answer)


if __name__ == '__main__':
    unittest.main()
