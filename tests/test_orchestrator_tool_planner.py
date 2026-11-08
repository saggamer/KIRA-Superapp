import json
import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.modules.setdefault('webview', types.SimpleNamespace())
from interface import KiraBrain
from orchestrator_v15.adapter_router import AdapterRouter, channel_for_request
from orchestrator_v15.tool_planner import parse_plan


class ModelToolPlannerTests(unittest.TestCase):
    def brain(self, plan, answers):
        brain = KiraBrain.__new__(KiraBrain)
        brain.v15_model_tool_planning = True
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda *_a: None
        brain._load_chat_messages = lambda *_a: []
        brain._smart_memory_context = lambda *_a, **_k: ''
        brain._log_agentic_event = lambda *_a: None
        brain._truncate = lambda text, length=1000: str(text)[:length]
        captured, executed = [], []
        def model(payload, timeout=60):
            captured.append(payload)
            if payload['type'] == 'plan_v15_tools':
                return plan
            return {'proposals': [answers.pop(0)]}
        brain._model_worker_request = model
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_a: (
            executed.append(block) or 'BROWSER_TOOL:\n{"results":[{"title":"Cafe source","url":"https://example.test/cafe"}]}')
        return brain, captured, executed

    def test_model_chooses_browser_without_search_or_browser_keywords(self):
        request = 'I am visiting Jaipur. Which boba tea cafe is worth a visit this week?'
        self.assertEqual(channel_for_request(request), 'control')
        action = '[BROWSER_TOOL]\nACTION: search\nQUERY: Jaipur boba tea cafes\n[/BROWSER_TOOL]'
        brain, captured, executed = self.brain(json.dumps({'route':'browser','purpose':'Look up current cafe information.'}),
                                               [action, 'Source: https://example.test/cafe'])
        answer = brain._run_v15_profile(request, 'planner-test', 'medium')
        self.assertEqual(executed, [action])
        self.assertIn('https://example.test/cafe', answer)
        self.assertEqual(captured[1]['tool_channel'], 'browser')
        self.assertEqual(brain.last_v15_skill_audit['active'], ['browser-research'])
        self.assertEqual(brain.last_v15_tool_plan['route'], 'browser')
        self.assertIn('Look up current cafe information.', brain.response_queue.get()['content'])

    def test_model_can_choose_no_tools_even_when_request_contains_search_word(self):
        request = 'Rewrite this sentence: search me the best cafe.'
        brain, captured, executed = self.brain('{"route":"answer","purpose":"Rewrite the supplied sentence."}',
                                               ['Find the best cafe for me.'])
        answer = brain._run_v15_profile(request, 'rewrite-test', 'medium')
        self.assertEqual(executed, [])
        self.assertEqual(answer, 'Find the best cafe for me.')
        self.assertEqual(captured[1]['tool_channel'], 'control')

    def test_cloned_app_clarification_carries_context_and_recovers_false_capability_denial(self):
        locate = '[LIST_DIR]\nPATH: ~/Downloads\n[/LIST_DIR]'
        open_folder = '[OPEN]\nTARGET: /tmp/Downloads/polyagent\n[/OPEN]'
        brain, captured, executed = self.brain(
            '{"route":"control","purpose":"Locate the cloned project and open its folder."}',
            ['I cannot directly interact with your local file system or open applications like Finder.',
             locate, open_folder, 'Opened the Polyagent folder in Finder.'])
        brain._load_chat_messages = lambda *_: [
            {'role': 'user', 'content': 'Open this cloned app in Finder.'},
            {'role': 'assistant', 'content': 'Which app?'},
            {'role': 'user', 'content': 'It is called Polyagent, I cloned it from GitHub.'}]
        results = iter(['LIST_DIR:\n/tmp/Downloads/polyagent', 'Opened: /tmp/Downloads/polyagent'])
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_a: executed.append(block) or next(results)
        result = brain._run_v15_profile('open it', 'desktop-followup', 'medium')
        self.assertIn('Polyagent folder', result)
        self.assertEqual(executed, [locate, open_folder])
        self.assertIn('Open this cloned app in Finder', captured[0]['conversation_context'])
        self.assertIn('Polyagent', captured[0]['conversation_context'])

    def test_waiting_for_vision_does_not_consume_the_remaining_task_time(self):
        analyze = '[IMAGE_ANALYZE]\nPATH: /tmp/image.png\nQUESTION: Identify this app\n[/IMAGE_ANALYZE]'
        open_folder = '[OPEN]\nTARGET: Finder\n[/OPEN]'
        brain, _, executed = self.brain('{"route":"control","purpose":"Identify the app and open Finder."}',
                                       [analyze, open_folder, 'Opened Finder.'])
        clock = [0.0]
        def run(block, *_a):
            executed.append(block)
            if block == analyze:
                clock[0] += 10000
                return 'IMAGE_ANALYSIS:\nApp identified.'
            return 'Opened: Finder'
        brain._run_agentic_capabilities_with_watchdog = run
        with patch('interface.time.monotonic', side_effect=lambda: clock[0]):
            self.assertEqual(brain._run_v15_profile('Identify this app and open Finder.', 'vision-wait', 'medium'), 'Opened Finder.')
        self.assertEqual(executed, [analyze, open_folder])

    def test_explicit_finder_action_cannot_be_lost_to_answer_only_routing(self):
        action = '[OPEN]\nTARGET: Finder\n[/OPEN]'
        for plan in ('{"route":"answer","purpose":"Reply directly."}', 'invalid plan'):
            brain, captured, executed = self.brain(plan, [action, 'Opened Finder.'])
            brain._run_agentic_capabilities_with_watchdog = lambda block, *_a: executed.append(block) or 'Opened: Finder'
            self.assertEqual(brain._run_v15_profile('Open Finder.', 'finder-route', 'medium'), 'Opened Finder.')
            self.assertEqual(brain.last_v15_tool_plan['route'], 'control')
            self.assertEqual(captured[0]['required_route'], 'control')
            self.assertEqual(executed, [action])
        for prompt in ("Don't open Finder", 'Why did Finder open?'):
            brain, captured, executed = self.brain('{"route":"answer","purpose":"Reply directly."}', ['No action was requested.'])
            brain._run_v15_profile(prompt, 'not-action', 'medium')
            self.assertIsNone(captured[0]['required_route'])
            self.assertEqual(executed, [])

    def test_desktop_observation_returns_to_model_before_navigation(self):
        actions = ['[WINDOW_REPORT]\n[/WINDOW_REPORT]',
                   '[SCREENSHOT_CONTEXT]\n[/SCREENSHOT_CONTEXT]',
                   '[IMAGE_ANALYZE]\nPATH: /tmp/screen.png\nQUESTION: Which app is visible?\n[/IMAGE_ANALYZE]',
                   '[OPEN]\nTARGET: Music\n[/OPEN]',
                   '[CLOSE]\nTARGET: Music\n[/CLOSE]']
        brain, captured, executed = self.brain(
            '{"route":"control","purpose":"Inspect and operate the requested desktop app."}',
            actions + ['Music was closed.'])
        results = iter(['WINDOW_REPORT: Finder', 'SCREENSHOT_CONTEXT: /tmp/screen.png',
                        'IMAGE_ANALYZE: Finder is visible', 'Opened: Music', 'CLOSE: Closed Music'])
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_a: (
            executed.append(block) or next(results))
        answer = brain._run_v15_profile('Inspect my desktop, open Music, then close Music.', 'desktop', 'high')
        self.assertEqual(executed, actions)
        self.assertEqual(answer, 'Music was closed.')
        self.assertIn('Local desktop harness', captured[1]['messages'][0]['content'])
        # Each action result is supplied before the next model decision.
        self.assertTrue(any('WINDOW_REPORT: Finder' in item['content']
                            for item in captured[-1]['messages']))

    def test_negated_launch_mention_can_be_answered_without_device_access(self):
        brain, captured, executed = self.brain(
            '{"route":"answer","purpose":"Explain the unexpected launch."}',
            ['The previous navigation shortcut matched the whole sentence.'])
        brain._run_v15_profile("I don't want Music opened. Why was it opened?", 'negated', 'medium')
        self.assertEqual(executed, [])
        self.assertNotIn('Local desktop harness', captured[1]['messages'][0]['content'])

    def test_routine_diagonal_browser_avoids_extra_passes_and_reports_unchanged_reads(self):
        observe = '[BROWSER_TOOL]\nACTION: observe\nTABID: video\n[/BROWSER_TOOL]'
        brain, captured, executed = self.brain(
            '{"route":"browser","purpose":"Open the requested video page."}', [observe, observe,
            'The page at https://example.test/video returned no readable content. I could not verify video playback.'])
        page = {'tabId': 'video', 'url': 'https://example.test/video', 'title': 'Video fixture',
                'text': '', 'visibleText': '', 'elements': [], 'scroll': {'y': 0}}
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_a: (
            executed.append(block) or 'BROWSER_TOOL:\n' + json.dumps(page))
        result = brain._run_v15_profile('Open https://example.test/video in the browser.', 'routine', 'diagonal',
                                       structured_thinking=True, orchestral_candidate=True)
        self.assertEqual(len(executed), 3)  # Navigate and two bounded readbacks.
        proposals = [item for item in captured if item['type'] == 'propose_v15']
        self.assertEqual(len(proposals), 3)
        for item in proposals:
            self.assertEqual(item['branches'], 1)
            self.assertFalse(item['structured_thinking'])
            self.assertFalse(item['diagonal_orchestral_candidate'])
        self.assertIn('https://example.test/video', result)
        self.assertIn('no readable content', result)
        self.assertNotIn('[BROWSER_TOOL]', result)

    def test_browser_continues_past_old_step_and_tool_call_caps(self):
        actions = [f'[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/source-{i}\n[/BROWSER_TOOL]'
                   for i in range(18)]
        brain, captured, executed = self.brain(
            '{"route":"browser","purpose":"Review the requested sources."}',
            actions + ['The requested sources have been reviewed.'])
        def run(block, *_args):
            executed.append(block)
            index = len(executed)
            return 'BROWSER_TOOL:\n' + json.dumps({
                'tabId': 'research', 'url': f'https://example.test/source-{index-1}',
                'title': f'Source {index}', 'text': f'Observed content {index}', 'elements': []})
        brain._run_agentic_capabilities_with_watchdog = run
        result = brain._run_v15_profile('Browse and review the requested sources.', 'long-browser', 'high')
        self.assertEqual(executed, actions)
        self.assertEqual(result, 'The requested sources have been reviewed.')
        statuses = []
        while not brain.response_queue.empty():
            statuses.append(brain.response_queue.get())
        self.assertTrue(any('step 19' in item.get('content', '') for item in statuses))
        self.assertFalse(any('step 8/8' in item.get('content', '') for item in statuses))
        self.assertIn('no fixed decision or tool-call cap', captured[1]['messages'][0]['content'])

    def test_blank_public_output_resumes_without_repeating_browser_actions(self):
        action = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/python\n[/BROWSER_TOOL]'
        for blank in ('', '   ', '<think>unfinished private text'):
            with self.subTest(blank=blank):
                brain, captured, executed = self.brain(
                    '{"route":"browser","purpose":"Read a Python learning source."}',
                    [action, blank, 'Start with variables and print(). Source: https://example.test/python'])
                brain._run_agentic_capabilities_with_watchdog = lambda block, *_a: (
                    executed.append(block) or 'BROWSER_TOOL:\n' + json.dumps({
                        'url': 'https://example.test/python', 'title': 'Python basics',
                        'text': 'A variable stores a value. print() displays it.', 'elements': []}))
                result = brain._run_v15_profile('Use browser sources to teach me Python.', 'blank-recovery', 'high')
                self.assertIn('variables', result)
                self.assertEqual(executed, [action])
                proposals = [item for item in captured if item['type'] == 'propose_v15']
                self.assertGreaterEqual(proposals[-1]['max_tokens'], 512)
                self.assertIn('print()', proposals[-1]['messages'][-1]['content'])
                self.assertNotIn('unfinished private', result)

    def test_empty_proposal_list_is_retried(self):
        brain, _, executed = self.brain('{"route":"answer","purpose":"Reply directly."}', [])
        calls = []
        def model(payload, **_kwargs):
            calls.append(payload)
            if payload['type'] == 'plan_v15_tools':
                return '{"route":"answer","purpose":"Reply directly."}'
            return {'proposals': [] if len(calls) == 2 else ['Hello.']}
        brain._model_worker_request = model
        self.assertEqual(brain._run_v15_profile('Hello', 'empty-list', 'medium'), 'Hello.')
        self.assertEqual(executed, [])

    def test_repeated_blank_browser_output_uses_answer_only_recovery(self):
        action = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/python\n[/BROWSER_TOOL]'
        brain, _, executed = self.brain('{"route":"browser","purpose":"Read Python source."}', [])
        responses = iter([action, '', '', ''])
        def model(payload, **_kwargs):
            if payload['type'] == 'plan_v15_tools':
                return '{"route":"browser","purpose":"Read Python source."}'
            if payload['type'] == 'finish_v15':
                self.assertIn('Python basics', payload['evidence'])
                return 'Python basics source: https://example.test/python'
            return {'proposals': [next(responses)]}
        brain._model_worker_request = model
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_a: (
            executed.append(block) or 'BROWSER_TOOL:\n' + json.dumps({
                'url': 'https://example.test/python', 'title': 'Python basics', 'text': 'print() displays text.'}))
        result = brain._run_v15_profile('Read a browser Python source.', 'finish-recovery', 'high')
        self.assertEqual(result, 'Python basics source: https://example.test/python')
        self.assertEqual(executed, [action])

    def test_persistent_empty_output_preserves_real_progress_without_leaking_private_text(self):
        action = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/python\n[/BROWSER_TOOL]'
        brain, _, executed = self.brain('{"route":"browser","purpose":"Read a source."}', [])
        responses = iter([action, '<think>secret</think>', '', ''])
        def model(payload, **_kwargs):
            if payload['type'] == 'plan_v15_tools':
                return '{"route":"browser","purpose":"Read a source."}'
            if payload['type'] == 'finish_v15':
                return '<analysis>private</analysis>'
            return {'proposals': [next(responses)]}
        brain._model_worker_request = model
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_a: (
            executed.append(block) or 'BROWSER_TOOL:\n' + json.dumps({
                'url': 'https://example.test/python', 'title': 'Python basics', 'text': 'Observed source text.'}))
        result = brain._run_v15_profile('Find Python sources in the browser.', 'empty-progress', 'high')
        self.assertEqual(executed, [action])
        self.assertIn('Python basics', result)
        self.assertIn('https://example.test/python', result)
        self.assertNotIn('secret', result)
        self.assertNotIn('private', result)
        self.assertNotIn('stopped without a verified answer', result)

    def test_model_cannot_create_unrequested_mcp_setup(self):
        action = '[MCP_BOOTSTRAP]\nTARGET: cursor_project\n[/MCP_BOOTSTRAP]'
        brain, _, executed = self.brain('{"route":"control","purpose":"Use local tools."}', [action]*10)
        answer = brain._run_v15_profile('Recommend a cafe.', 'scope-test', 'medium')
        self.assertEqual(executed, [])
        self.assertIn('no connector draft was created', answer)

    def test_answer_plan_cannot_silently_execute_tools(self):
        brain, _, executed = self.brain('{"route":"answer","purpose":"Reply directly."}',
                                       ['[APP_LIST]\n[/APP_LIST]']*10)
        answer = brain._run_v15_profile('Hello', 'answer-test', 'medium')
        self.assertEqual(executed, [])
        self.assertIn('no action was executed', answer)

    def test_mcp_plan_selects_connection_workflow_without_installing_anything(self):
        brain, captured, executed = self.brain('{"route":"mcp","purpose":"Retrieve the requested reminder."}', [])
        answer = brain._run_v15_profile('What does my charger reminder say?', 'app-test', 'medium')
        self.assertEqual(executed, [])
        self.assertIn('Connect an approved server', answer)
        self.assertEqual(len(captured), 1)

    def test_invalid_plan_and_worker_failure_stop_without_keyword_fallback(self):
        for bad in ('not json', '{"route":"shell","purpose":"run"}',
                    '{"route":"browser","purpose":"lookup","command":"rm"}'):
            brain, captured, executed = self.brain(bad, [])
            answer = brain._run_v15_profile('Search the web for cafes', 'invalid-test', 'medium')
            self.assertEqual(executed, [])
            self.assertEqual(len(captured), 1)
            self.assertIn('no action was executed', answer)
        brain, _, executed = self.brain('', [])
        brain._model_worker_request = MagicMock(side_effect=RuntimeError('timeout'))
        self.assertIn('no action was executed', brain._run_v15_profile('Find a cafe', 'timeout', 'medium'))
        self.assertEqual(executed, [])

    def test_plan_does_not_accept_action_markup_as_public_purpose(self):
        with self.assertRaises(ValueError):
            parse_plan('{"route":"control","purpose":"[MCP_APPLY]"}')

    def test_planned_adapter_selection_is_not_replaced_by_keyword_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ('control', 'browser'):
                path = Path(directory)/name
                path.mkdir()
                (path/'architecture.json').write_text('{"base_snapshot":"same"}')
                checkpoint = path/'checkpoint.safetensors'
                checkpoint.touch()
                paths.append(checkpoint)
            router = AdapterRouter(*paths)
            model, mx = MagicMock(), MagicMock()
            with patch.dict(sys.modules, {'mlx': MagicMock(core=mx), 'mlx.core': mx}):
                self.assertEqual(router.activate(model, 'Hello', channel='browser'), 'browser')
                self.assertEqual(router.activate(model, 'Use the browser', channel='control'), 'control')
                with self.assertRaises(ValueError):
                    router.activate(model, 'anything', channel='invented')


if __name__=='__main__':unittest.main()
