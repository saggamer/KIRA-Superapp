import json
import types
import unittest

from orchestrator_v15.adapter_router import channel_for_request
from orchestrator_v15.diagonal_orchestral_candidate import (
    batch_closed_json, generate_branches, parse_branch, select_branch)
from orchestrator_v15.thinking_controller import ThinkingController


YOUTUBE_REQUEST = ('i want to watch a youtube video surf youtube for the best '
                   'tech review video of a Mac mini')


class FakeTokenizer:
    eos_token_ids = {0}

    def apply_chat_template(self, messages, **_kwargs):
        return '\n'.join(message['content'] for message in messages)

    def encode(self, text, **_kwargs):
        return list(text.encode())


class OrchestralCandidateTests(unittest.TestCase):
    def test_real_failure_prompt_routes_to_browser(self):
        self.assertEqual(channel_for_request(YOUTUBE_REQUEST), 'browser')

    def test_rejects_mixed_action(self):
        bad = {'evidence': '', 'plan': '', 'risk': '',
               'action': '[BROWSER_TOOL]\nACTION: search\nQUERY: Mac mini\n[/BROWSER_TOOL]\nextra'}
        self.assertIsNone(parse_branch(json.dumps(bad)))

    def test_structured_browser_action_is_normalized_without_executing_it(self):
        branch = {'evidence': 'No page loaded.', 'plan': 'Search first.', 'risk': '',
                  'action': {'tool': 'BROWSER_TOOL', 'action': 'search', 'query': 'free shooter games'}}
        parsed = parse_branch(json.dumps(branch))
        self.assertEqual(parsed['action'],
                         '[BROWSER_TOOL]\nACTION: search\nQUERY: free shooter games\n[/BROWSER_TOOL]')
        branch['action']['extra'] = 'ignored'
        self.assertIsNone(parse_branch(json.dumps(branch)))

    def test_live_branch_shapes_normalize_to_the_same_navigation(self):
        url = 'http://127.0.0.1:60480/'
        base = {'evidence': 'N/A', 'plan': 'Navigate.', 'risk': 'Page may fail.'}
        actions = [
            {'name': 'navigate', 'arguments': {'URL': url}},
            {'arguments': {'OPERATION': 'navigate', 'URL': url}},
            '[BROWSER_TOOL]\nACTION: navigate\nURL: ' + url + '\n[/BROWSER_TOOL]',
        ]
        for action in actions:
            raw = json.dumps({**base, 'action': action}).replace('\\n', '\n')
            parsed = parse_branch(raw)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed['action'],
                             '[BROWSER_TOOL]\nACTION: navigate\nURL: ' + url + '\n[/BROWSER_TOOL]')

    def test_observation_continuation_accepts_plain_action_or_answer(self):
        action = '[BROWSER_TOOL]\nACTION: observe\nSCREENSHOT: true\n[/BROWSER_TOOL]'
        self.assertEqual(parse_branch(action)['action'], action)
        answer = 'The page title is Game Directory Fixture; Game Alpha is free.'
        self.assertEqual(parse_branch(answer)['action'], answer)
        self.assertIsNone(parse_branch('{"evidence":"unfinished"'))

    def test_verified_count_survives_verbose_evidence_and_beats_reload(self):
        controller = ThinkingController('Open https://example.test and fill Lookup with Kira, run query, then report count', diagonal_level=True)
        controller.record('[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]',
                          'BROWSER_TOOL:\n{"url":"https://example.test","text":"Lookup Run query","tabId":"tab1"}')
        controller.record('[BROWSER_TOOL]\nACTION: act\nOPERATION: fill\nTARGET_NAME: Lookup\nVALUE: Kira\n[/BROWSER_TOOL]',
                          'BROWSER_TOOL:\n{"url":"https://example.test","text":"Lookup Run query","tabId":"tab1"}')
        controller.record('[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Run query\n[/BROWSER_TOOL]',
                          'BROWSER_TOOL:\n{"url":"https://example.test","text":"Lookup Run query","tabId":"tab1"}')
        controller.record('[BROWSER_TOOL]\nACTION: observe\nTABID: tab1\n[/BROWSER_TOOL]',
                          'BROWSER_TOOL:\n{"url":"https://example.test","text":"Matched 17 records for Kira","tabId":"tab1"}')
        evidence = 'The latest observation shows Matched 17 records for Kira. ' * 5
        finals = [
            {'evidence': evidence, 'plan': 'Report the observed count.', 'risk': '',
             'action': {'text': 'The matched count for Kira is 17.'}},
            {'evidence': evidence, 'plan': 'Report the observed count.', 'risk': '',
             'action': 'FINAL_ANSWER: The matched count for Kira is 17.'},
        ]
        parsed = [parse_branch(json.dumps(item)) for item in finals]
        self.assertTrue(all(parsed))
        self.assertTrue(all(len(item['evidence']) <= 180 for item in parsed))
        self.assertEqual(parsed[1]['action'], 'The matched count for Kira is 17.')
        reload_action = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]'
        self.assertIsNotNone(controller.preflight(reload_action))
        selected, audit = select_branch({'branches': parsed + [
            {'evidence': '', 'plan': '', 'risk': '', 'action': reload_action}],
            'judge_scores': [0, 0, 5]}, controller)
        self.assertEqual(selected, 'The matched count for Kira is 17.')
        self.assertEqual(audit['valid'], 2)

    def test_batch_stops_each_complete_json_before_token_budget(self):
        outputs = ['{"one":1}', '{"two":2}']
        instances = []

        class FakeGenerator:
            def __init__(self, *_args, **_kwargs):
                self.position = [0, 0]
                self.removed = set()
                self.closed = False
                instances.append(self)

            def insert(self, _prompts, _limits):
                return [10, 11]

            def next_generated(self):
                rows = []
                for index, uid in enumerate((10, 11)):
                    if uid in self.removed:
                        continue
                    if self.position[index] >= len(outputs[index]):
                        continue
                    char = outputs[index][self.position[index]]
                    self.position[index] += 1
                    rows.append(types.SimpleNamespace(uid=uid, token=ord(char), finish_reason=None))
                return rows

            def remove(self, uids):
                self.removed.update(uids)

            def close(self):
                self.closed = True

        class CharTokenizer(FakeTokenizer):
            def decode(self, ids):
                return ''.join(map(chr, ids))

        result = batch_closed_json(None, CharTokenizer(), [[1], [2]], max_tokens=64,
                                   sampler=None, generator_cls=FakeGenerator)
        self.assertEqual(result.texts, outputs)
        self.assertEqual(instances[0].removed, {10, 11})
        self.assertTrue(instances[0].closed)

    def test_batched_independent_branches_and_judge_cannot_override_validator(self):
        calls = []
        valid = '[BROWSER_TOOL]\nACTION: search\nQUERY: Mac mini review YouTube\n[/BROWSER_TOOL]'
        invalid = '[BROWSER_TOOL]\nACTION: click\nTARGET_NAME: video\n[/BROWSER_TOOL]'

        def batch(_model, _tokenizer, prompts, **_kwargs):
            calls.append(('batch', len(prompts)))
            outputs = [
                {'evidence': 'No page loaded.', 'plan': 'Search YouTube.', 'risk': 'None.', 'action': valid},
                {'evidence': 'No page loaded.', 'plan': 'Click video.', 'risk': 'Unobserved target.', 'action': invalid},
                {'evidence': 'No page loaded.', 'plan': 'Search.', 'risk': 'None.', 'action': valid},
            ]
            return types.SimpleNamespace(texts=[json.dumps(item) for item in outputs])

        def judge(_model, _tokenizer, **_kwargs):
            calls.append(('judge', 1))
            return '[1,5,2]'

        result = generate_branches(None, FakeTokenizer(),
                                   [{'role': 'user', 'content': YOUTUBE_REQUEST}],
                                   batch_generate_fn=batch, evaluator_generate_fn=judge)
        chosen, audit = select_branch(result, ThinkingController(YOUTUBE_REQUEST, diagonal_level=True))
        self.assertEqual(calls, [('batch', 3)])
        self.assertEqual(chosen, valid)
        self.assertEqual(audit['valid'], 2)
        self.assertEqual(audit['selected_index'], 0)

    def test_unsupported_browser_action_is_discarded_before_evaluation(self):
        raw = {'evidence': 'N/A', 'plan': 'Find app.', 'risk': '',
               'action': {'name': 'APP_DISCOVER', 'arguments': {'URL': 'https://example.test'}}}
        self.assertIsNone(parse_branch(json.dumps(raw)))

    def test_structured_scroll_action_is_supported(self):
        branch = {'evidence': 'Lower page not read.', 'plan': 'Scroll down.', 'risk': '',
                  'action': {'name': 'scroll', 'arguments': {'pixels': 700}}}
        self.assertEqual(parse_branch(json.dumps(branch))['action'],
                         '[BROWSER_TOOL]\nACTION: scroll\nPIXELS: 700\n[/BROWSER_TOOL]')


if __name__ == '__main__':
    unittest.main()
