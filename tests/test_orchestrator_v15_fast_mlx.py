import json
import types
import unittest

from orchestrator_v15.fast_mlx import browser_step_budget, browser_evidence_snapshot, generate_closed_proposal
from orchestrator_v15.thinking_controller import ThinkingController


class FastMlxTests(unittest.TestCase):
    def test_browser_first_step_uses_one_short_proposal(self):
        request = 'Use browser to find an animated comedy film and show an image.'
        self.assertEqual(browser_step_budget(request, []),
                         {'branches': 1, 'max_tokens': 160, 'structured_thinking': False})
        self.assertEqual(browser_step_budget(request, [{'results': []}]),
                         {'branches': 2, 'max_tokens': 192, 'structured_thinking': False})
        self.assertIsNone(browser_step_budget('Click the observed button.', []))
        self.assertEqual(browser_step_budget(
            'Find a movie and show an image for me to see and choose.', []),
            {'branches': 1, 'max_tokens': 160, 'structured_thinking': False})

    def test_stream_stops_after_closed_action(self):
        seen = []

        def stream_fn(*_args, **_kwargs):
            for part in ('[BROWSER_TOOL]\n', 'ACTION: search\n', 'QUERY: animated comedy\n',
                         '[/BROWSER_TOOL]', '\nUnwanted extra text'):
                seen.append(part)
                yield types.SimpleNamespace(text=part)

        result = generate_closed_proposal(None, None, 'prompt', max_tokens=128,
                                          sampler=None, stream_generate_fn=stream_fn)
        self.assertEqual(result, '[BROWSER_TOOL]\nACTION: search\nQUERY: animated comedy\n[/BROWSER_TOOL]')
        self.assertEqual(len(seen), 4)

    def test_plain_answer_is_not_truncated(self):
        def stream_fn(*_args, **_kwargs):
            for part in ('I found two', ' options.'):
                yield types.SimpleNamespace(text=part)

        self.assertEqual(generate_closed_proposal(None, None, 'prompt', max_tokens=128,
                         sampler=None, stream_generate_fn=stream_fn), 'I found two options.')

    def test_browser_snapshot_keeps_recent_evidence_without_prompt_growth(self):
        controller = ThinkingController('Compare two web pages.')
        action = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]'
        for index in range(10):
            controller.record(action, 'BROWSER_TOOL:\n{"url":"https://example.test",'
                              '"title":"Example","text":"' + ('x' * 5000) + '"}')
        snapshot = browser_evidence_snapshot(controller)
        self.assertLess(len(snapshot), 6000)
        self.assertEqual(snapshot.count('"action":"navigate"'), 4)
        self.assertIn('Example', snapshot)

    def test_scrolled_snapshot_keeps_lower_visible_text(self):
        controller = ThinkingController('Scroll down and report the lower heading.')
        action = '[BROWSER_TOOL]\nACTION: scroll\nPIXELS: 700\n[/BROWSER_TOOL]'
        controller.record(action, 'BROWSER_TOOL:\n' + json.dumps({
            'tabId': 'tab-1', 'scroll': {'y': 700, 'atEnd': True},
            'visibleText': 'intro ' * 500 + 'Lower heading: Practical Details'}))
        snapshot = browser_evidence_snapshot(controller)
        self.assertIn('Lower heading: Practical Details', snapshot)


if __name__ == '__main__':
    unittest.main()
