import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from orchestrator_v15.adapter_router import channel_for_request
from orchestrator_v15.skill_runtime import (
    MAX_CONTEXT_CHARS, NAMES, ROOT, read_skill, select_skills, skill_context,
)


class SkillRuntimeTests(unittest.TestCase):
    def test_history_grader_accepts_equivalent_dates_but_requires_real_cached_read(self):
        import importlib.util
        path = Path(__file__).parents[1] / 'scripts' / 'benchmark_orchestrator_skills.py'
        spec = importlib.util.spec_from_file_location('skill_benchmark', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actions = [
            {'action': '[BROWSER_TOOL]\nACTION: search_history\nQUERY: Cedar\n[/BROWSER_TOOL]',
             'result': 'found a record'},
            {'action': '[BROWSER_TOOL]\nACTION: read_history\nHISTORY_ID: real-id\n[/BROWSER_TOOL]',
             'result': '{"cached":true,"text":"Launch date: 21 November."}'},
        ]
        for answer in ('It is 21 November.', 'It is November 21st.'):
            self.assertTrue(module.score_history(answer, actions))
        self.assertFalse(module.score_history('It is November 22nd.', actions))
        self.assertFalse(module.score_history('It is November 21st.', actions[:1]))

    def test_task_selection_and_negative_controls(self):
        cases = [
            ('Compare two browser pages and fill the form', True, False,
             ('browser-interaction', 'browser-research')),
            ('What did you search for yesterday?', True, False, ('browser-history',)),
            ('Read the saved pages from earlier research', True, False, ('browser-history',)),
            ('Scroll down the page and report the lower heading', True, False, ('browser-interaction',)),
            ('Use MCP to search my notes', False, False, ('mcp-app-workflow',)),
            ('Create an inactive connector draft', False, False, ('mcp-app-workflow',)),
            ('Tell me a joke', False, False, ()),
            ('Create a presentation about browser controls', False, True, ()),
        ]
        for request, browser, artifact, expected in cases:
            with self.subTest(request=request):
                self.assertEqual(select_skills(request, browser_task=browser,
                                               artifact_task=artifact), expected)

    def test_explicit_skill_uses_matching_adapter_without_overriding_artifacts(self):
        request = '$browser-history orchids'
        self.assertEqual(channel_for_request(request), 'browser')
        self.assertEqual(select_skills(request, browser_task=True)[0], 'browser-history')
        self.assertEqual(channel_for_request('Create slides using $browser-research'), 'control')
        self.assertEqual(select_skills('$mcp-app-workflow', browser_task=False), ('mcp-app-workflow',))

    def test_only_selected_bodies_enter_context_and_budget_is_bounded(self):
        loaded = read_skill('mcp-app-workflow')
        unrelated = skill_context('Hello', browser_task=False)
        self.assertIn(loaded.description, unrelated.prompt)
        self.assertNotIn(loaded.body, unrelated.prompt)
        selected = skill_context('Use MCP to search notes', browser_task=False)
        self.assertIn(loaded.body, selected.prompt)
        self.assertEqual(selected.audit[0]['sha256'], loaded.sha256)
        combined = skill_context('Research a browser page, fill and submit its form', browser_task=True)
        self.assertLessEqual(len(combined.prompt), MAX_CONTEXT_CHARS)
        self.assertLessEqual(len(combined.active), 2)

    def test_no_skill_paths_or_external_scripts_can_be_requested(self):
        with self.assertRaises(ValueError):
            read_skill('../../some-file')
        self.assertEqual(select_skills('$unknown-skill', browser_task=False), ())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'skills'
            root.mkdir()
            (root / 'browser-history').symlink_to(ROOT / 'browser-history', target_is_directory=True)
            with self.assertRaises(ValueError):
                read_skill('browser-history', root)

    def test_bad_package_is_skipped_without_breaking_other_skills(self):
        with tempfile.TemporaryDirectory() as temp:
            context = skill_context('Use MCP', browser_task=False, root=temp)
            self.assertEqual(context.active, ())
            self.assertEqual(len(context.errors), len(NAMES))

    def test_disabled_path_has_no_prompt_overhead(self):
        with patch.dict(os.environ, {'KIRA_V15_SKILLS': 'off'}):
            self.assertEqual(skill_context('Use browser', browser_task=True).prompt, '')

    def test_compacted_history_keeps_real_ids_pagination_and_saved_text(self):
        from orchestrator_v15.fast_mlx import browser_evidence_snapshot
        from orchestrator_v15.thinking_controller import ThinkingController
        controller = ThinkingController('Read my saved search history about Cedar.')
        search = '[BROWSER_TOOL]\nACTION: search_history\nQUERY: Cedar\n[/BROWSER_TOOL]'
        controller.record(search, 'BROWSER_TOOL:\n' + json.dumps({
            'total': 1, 'entries': [{'historyId': 'real-saved-id', 'kind': 'page',
                                    'title': 'Cedar', 'snippet': 'Saved launch note'}],
            'nextOffset': None,
        }))
        snapshot = json.loads(browser_evidence_snapshot(controller).split('\n', 1)[1])
        self.assertEqual(snapshot['recent'][0]['entries'][0]['historyId'], 'real-saved-id')
        self.assertEqual(snapshot['recent'][0]['total'], 1)
        read = '[BROWSER_TOOL]\nACTION: read_history\nHISTORY_ID: real-saved-id\n[/BROWSER_TOOL]'
        self.assertIsNone(controller.preflight(read))
        controller.record(read, 'BROWSER_TOOL:\n' + json.dumps({
            'historyId': 'real-saved-id', 'cached': True, 'text': 'Launch date: 21 November.',
            'nextTextOffset': 5000, 'results': [{'title': 'Cedar', 'url': 'https://example.test',
                                               'resultIndex': 10}],
        }))
        snapshot = json.loads(browser_evidence_snapshot(controller).split('\n', 1)[1])
        latest = snapshot['recent'][-1]
        self.assertTrue(latest['cached'])
        self.assertEqual(latest['nextTextOffset'], 5000)
        self.assertEqual(latest['results'][0]['resultIndex'], 10)
        self.assertIn('21 November', latest['text'])

    def test_browser_skill_survives_observation_compaction_in_real_profile(self):
        sys.modules.setdefault('webview', types.SimpleNamespace())
        from interface import KiraBrain
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._truncate = lambda text, length=1000: str(text)[:length]
        brain._load_chat_messages = lambda _chat_id: []
        brain._smart_memory_context = lambda *_a, **_k: ''
        brain._log_agentic_event = lambda *_a: None
        action = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]'
        captured = []
        def model(payload, timeout=60):
            captured.append(payload)
            return {'proposals': [action if len(captured) == 1 else 'The page title is Example.']}
        brain._model_worker_request = model
        brain._run_agentic_capabilities_with_watchdog = lambda *_args: (
            'BROWSER_TOOL:\n' + json.dumps({'url': 'https://example.test', 'title': 'Example',
                                         'text': 'Example', 'observationId': 'o1'}))
        result = brain._run_v15_profile('Open https://example.test and read its title.', 'skill-test', 'medium')
        self.assertEqual(result, 'The page title is Example.')
        self.assertEqual(captured[0]['skill_audit']['active'], ['browser-research'])
        self.assertIn(read_skill('browser-research').body, captured[-1]['messages'][0]['content'])

    def test_mcp_skill_reaches_profile_and_existing_apply_boundary_stays(self):
        sys.modules.setdefault('webview', types.SimpleNamespace())
        from interface import KiraBrain
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._load_chat_messages = lambda _chat_id: []
        brain._smart_memory_context = lambda *_a, **_k: ''
        captured = []
        def model(payload, timeout=60):
            captured.append(payload)
            return {'proposals': ['[MCP_APPLY]\nDRAFT: /tmp/test.json\nTARGET: kira_app\n[/MCP_APPLY]']}
        brain._model_worker_request = model
        brain._run_agentic_capabilities_with_watchdog = lambda *_a: self.fail('Must not apply')
        result = brain._run_v15_profile('Inspect the MCP draft; do not apply it.', 'skill-mcp', 'medium')
        self.assertIn('Keep the draft inactive', result)
        self.assertIn(read_skill('mcp-app-workflow').body, captured[0]['messages'][0]['content'])


if __name__ == '__main__':
    unittest.main()
