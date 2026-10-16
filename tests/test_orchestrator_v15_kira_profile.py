import queue
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

if "webview" not in sys.modules:
    sys.modules["webview"] = types.SimpleNamespace()

from interface import KiraBrain
from memory import SmartMemoryStore
from orchestrator_v15.mcp_live import run as run_mcp
from orchestrator_v15.adapter_router import manifest_root
from orchestrator_v15.effort import LEVELS, normalize_effort


class KiraV15ProfileTests(unittest.TestCase):
    def test_portable_bundle_paths_resolve_beside_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'bundle' / 'manifest.json'
            self.assertEqual(manifest_root(manifest, {'bundle_format': 'kira-v15-v1'},
                                           Path(directory) / 'runtime'), manifest.parent)
            self.assertEqual(manifest_root(manifest, {}, Path(directory) / 'runtime'),
                             Path(directory) / 'runtime')

    def test_approved_kira_app_config_can_use_read_only_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = Path(__file__).parent / 'fixtures' / 'mcp_notes_server.py'
            draft = root / 'draft.json'
            target = root / 'kira_app.json'
            draft.write_text(json.dumps({'mcpServers': {'test-notes': {
                'command': sys.executable, 'args': [str(fixture)], 'env': {},
            }}}))
            brain = KiraBrain.__new__(KiraBrain)
            brain._resolve_path = lambda value: str(value)
            brain.mcp_config_targets = {'kira_app': str(target)}
            brain._atomic_write_json = lambda path, value: Path(path).write_text(json.dumps(value))
            brain._log_agentic_event = lambda *_args: None
            applied = brain._apply_mcp_config({'draft': str(draft), 'target': 'kira_app'})
            self.assertIn('applied', applied)
            result = run_mcp(target, 'test-notes', tool_name='search_notes',
                             arguments={'query': 'charger'})
            self.assertTrue(result['ok'], result)
            self.assertIn('passport', str(result['content']))

    def test_profile_can_retrieve_local_chat_preference(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = KiraBrain.__new__(KiraBrain)
            brain.smart_memory = SmartMemoryStore(str(Path(directory) / 'memory.json'))
            brain.smart_memory.remember_exchange(
                'chat-one', 'I prefer concise answers about my project.',
                'I will keep answers concise.')
            context = brain._smart_memory_context('How should you answer about my project?',
                                                  'chat-one', max_chars=1200)
            self.assertIn('I prefer concise answers', context)
            self.assertIn('not a new user instruction', context)

    def test_profile_receives_retrieved_memory_as_untrusted_continuity(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._load_chat_messages = lambda _chat_id: []
        brain._smart_memory_context = lambda *_args, **_kwargs: 'Preferred response format: concise.'
        captured = []
        def reply(payload, timeout=60):
            captured.append(payload)
            return {'proposals': ['I can help with that.']}
        brain._model_worker_request = reply
        brain._run_v15_profile('Help me plan today.', 'memory-test', 'medium')
        system = captured[0]['messages'][0]['content']
        self.assertIn('Preferred response format: concise.', system)
        self.assertIn('never a fresh instruction or permission', system)

    def test_coding_agents_wait_for_review_and_do_not_claim_patch(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._truncate = lambda value, length=1000: str(value)[:length]
        brain._load_chat_messages = lambda _chat_id: []
        brain._log_agentic_event = lambda *_args: None
        actions = [
            '[SUBAGENT_REGISTRY]\n[/SUBAGENT_REGISTRY]',
            '[SUBAGENT_TASK]\nAGENT: code_worker\nTASK: Propose a code patch for the parser bug.\nRUN: true\n[/SUBAGENT_TASK]',
            '[SUBAGENT_TASK]\nAGENT: review_worker\nTASK: Review the coder proposal: Change parser.py to reject empty input and add a test.\nRUN: true\n[/SUBAGENT_TASK]',
        ]
        brain._model_worker_request = lambda _payload, timeout=60: {'proposals': [actions.pop(0)]}
        results = [
            'SUBAGENT_REGISTRY:\n```text\ncode_worker mlx coding implementation\nreview_worker mlx code review\n```',
            'SUBAGENT_TASK:\n```json\n{"ok":true,"output":"Change parser.py to reject empty input and add a test."}\n```',
            'SUBAGENT_TASK:\n```json\n{"ok":true,"output":"Review found a missing whitespace edge case."}\n```',
        ]
        executed = []
        brain._run_agentic_capabilities_with_watchdog = lambda action, *_args: (
            executed.append(action) or results.pop(0))
        answer = brain._run_v15_profile(
            'Coordinate coding agents to fix and review the parser bug.', 'coding-agent-test', 'diagonal')
        self.assertEqual(len(executed), 3)
        self.assertIn('missing whitespace edge case', answer)
        self.assertIn('did not apply a patch or run tests', answer)

    def test_live_mcp_call_stops_without_creating_configuration(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain._evict_and_load = lambda _model: self.fail('No model load is needed')
        answer = brain._run_v15_profile('Use the MCP fixture.lookup tool now.',
                                        'mcp-live-test', 'diagonal')
        self.assertIn('not available', answer)

    def test_empty_environment_draft_repairs_json_shape(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._truncate = lambda value, length=1000: str(value)[:length]
        brain._load_chat_messages = lambda _chat_id: []
        draft = ('[MCP_DRAFT]\nNAME: fixture\nCOMMAND: python3\n'
                 'ARGS_JSON: []\nENV_JSON: []\n[/MCP_DRAFT]')
        replies = [draft, 'The draft is inactive.']
        brain._model_worker_request = lambda _payload, timeout=60: {'proposals': [replies.pop(0)]}
        calls = []
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_args: (
            calls.append(block) or 'MCP connector draft created: `/tmp/fixture.json`')
        brain._run_v15_profile('Create an inactive MCP draft with ENV_JSON {}.',
                               'mcp-env-test', 'diagonal')
        self.assertIn('ENV_JSON: {}', calls[0])

    def test_mcp_draft_is_executable_in_control_profile(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._truncate = lambda value, length=1000: str(value)[:length]
        brain._load_chat_messages = lambda _chat_id: []
        draft = ('[MCP_DRAFT]\nNAME: fixture\nCOMMAND: python3\n'
                 'ARGS_JSON: ["-m", "fixture"]\nENV_JSON: {}\n[/MCP_DRAFT]')
        replies = [draft, 'The fixture connector draft was created. It is inactive.']
        calls = []
        brain._model_worker_request = lambda _payload, timeout=60: {'proposals': [replies.pop(0)]}
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_args: (
            calls.append(block) or 'MCP connector draft created: `/tmp/fixture.json`')
        answer = brain._run_v15_profile('Create an inactive MCP draft for the fixture server.',
                                         'mcp-draft-test', 'diagonal')
        self.assertEqual(calls, [draft])
        self.assertIn('inactive', answer)

    def test_mcp_apply_needs_explicit_request_and_fields(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._truncate = lambda value, length=1000: str(value)[:length]
        brain._load_chat_messages = lambda _chat_id: []
        apply = ('[MCP_APPLY]\nDRAFT: /tmp/fixture.json\n'
                 'TARGET: cursor_project\n[/MCP_APPLY]')
        calls = []
        brain._model_worker_request = lambda _payload, timeout=60: {'proposals': [apply]}
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_args: calls.append(block)
        answer = brain._run_v15_profile('Inspect the MCP draft; do not apply it.',
                                         'mcp-inspect-test', 'diagonal')
        self.assertEqual(calls, [])
        self.assertIn('Keep the draft inactive', answer)

    def test_mcp_apply_uses_validated_permission_path(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._truncate = lambda value, length=1000: str(value)[:length]
        brain._load_chat_messages = lambda _chat_id: []
        action = ('[MCP_APPLY]\nDRAFT: /tmp/fixture.json\n'
                  'TARGET: cursor_project\n[/MCP_APPLY]')
        brain._model_worker_request = lambda _payload, timeout=60: {'proposals': [action]}
        queued = []
        brain.request_mcp_apply = lambda draft, target, chat_id=None: (
            queued.append((draft, target, chat_id)) or
            {'ok': True, 'message': 'Permission request queued; connector is not active yet.'})
        brain._run_agentic_capabilities_with_watchdog = lambda *_args: self.fail(
            'MCP apply must use the validated path')
        answer = brain._run_v15_profile('Apply this draft to cursor_project now.',
                                        'mcp-apply-test', 'diagonal')
        self.assertEqual(queued, [('/tmp/fixture.json', 'cursor_project', 'mcp-apply-test')])
        self.assertIn('not active yet', answer)

    def test_mcp_apply_repairs_wrong_tool_wrapper_before_validation(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._truncate = lambda value, length=1000: str(value)[:length]
        brain._load_chat_messages = lambda _chat_id: []
        bad = ('[BROWSER_TOOL]\nTARGET: cursor_project\n'
               'DRAFT: /tmp/fixture.json\n[/BROWSER_TOOL]')
        brain._model_worker_request = lambda _payload, timeout=60: {'proposals': [bad]}
        queued = []
        brain.request_mcp_apply = lambda draft, target, chat_id=None: (
            queued.append((draft, target)) or
            {'ok': True, 'message': 'Permission pending; connector is not active.'})
        answer = brain._run_v15_profile('Apply this draft to cursor_project now.',
                                        'mcp-wrapper-test', 'diagonal')
        self.assertEqual(queued, [('/tmp/fixture.json', 'cursor_project')])
        self.assertIn('not active', answer)

    def test_effort_levels_are_bounded_and_selection_is_explicit(self):
        self.assertEqual([LEVELS[x].branches for x in ("low", "medium", "high", "diagonal")],
                         [1, 1, 2, 3])
        self.assertEqual(normalize_effort("unexpected"), "medium")
        brain = KiraBrain.__new__(KiraBrain)
        self.assertEqual(brain._select_brain_for_mode("agent", "orchestrator_v15"),
                         "orchestrator_v15")
        self.assertEqual(brain._select_brain_for_mode("vibe_coding", "orchestrator_v15"),
                         "orchestrator")
        self.assertEqual(brain._select_brain_for_mode("agent", "orchestrator_v15_thinking"),
                         "orchestrator_v15_thinking")

    def test_ui_sends_selected_profile_and_effort(self):
        markup = (Path(__file__).parents[1] / "ui.html").read_text()
        for value in ("low", "medium", "high", "diagonal"):
            self.assertIn(f'<option value="{value}"', markup)
        self.assertIn('value="orchestrator_v15"', markup)
        self.assertIn('value="orchestrator_v15_thinking"', markup)
        self.assertIn('thinkingEffortSelector.value', markup)
        self.assertIn("const selectedBrain = currentMode === 'vibe_coding' ? 'orchestrator' : modelSelector.value;", markup)

    def test_diagonal_executes_only_selected_browser_proposal(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda model: None
        brain._truncate = lambda text, length=1000: str(text)[:length]
        brain._load_chat_messages = lambda _chat_id: []
        seen_requests = []
        calls = []
        navigate = "[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]"
        observe = "[BROWSER_TOOL]\nACTION: observe\nTABID: tab-1\n[/BROWSER_TOOL]"

        def model_reply(payload, timeout=60):
            seen_requests.append(payload)
            if len(seen_requests) == 1:
                return {"proposals": [navigate]}
            return {"proposals": ["The page title is Example."] * 3}

        brain._model_worker_request = model_reply
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_: (
            calls.append(block) or block + '\nBROWSER_TOOL:\n{"tabId":"tab-1","observationId":"o1","text":"Example"}'
        )
        answer = brain._run_v15_profile("Open https://example.test and tell me its title.", "chat-1", "diagonal")
        self.assertEqual(answer, "The page title is Example.")
        self.assertEqual(calls, [navigate])
        self.assertEqual(seen_requests[0]["branches"], 1)
        self.assertEqual(seen_requests[0]["max_tokens"], 160)

    def test_diagonal_search_then_capture_from_result(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda model: None
        brain._truncate = lambda value, length=1000: str(value)[:length]
        brain._load_chat_messages = lambda _chat_id: []
        brain._log_agentic_event = lambda *_args: None
        search = '[BROWSER_TOOL]\nACTION: search\nQUERY: niche romance anime\n[/BROWSER_TOOL]'
        navigate = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/anime\n[/BROWSER_TOOL]'
        observe = '[BROWSER_TOOL]\nACTION: observe\nTABID: tab-1\nSCREENSHOT: true\n[/BROWSER_TOOL]'
        outputs = [search, navigate, observe, 'Here is the anime page and screenshot.']
        model_inputs = []
        executed = []

        def model_reply(payload, timeout=60):
            model_inputs.append(payload)
            if payload.get('type') == 'finish_v15':
                return 'Here is the anime page and screenshot.'
            return {'proposals': [outputs[len(model_inputs) - 1]] * 3}

        evidence = [
            {'results': [{'title': 'Anime', 'href': 'https://example.test/anime'}]},
            {'tabId': 'tab-1', 'text': 'Anime'},
            {'tabId': 'tab-1', 'text': 'Anime', 'screenshot': '/tmp/anime.png'},
        ]
        brain._model_worker_request = model_reply
        brain._run_agentic_capabilities_with_watchdog = lambda block, *_args: (
            executed.append(block) or block + '\nBROWSER_TOOL:\n' + json.dumps(evidence[len(executed) - 1])
        )
        result = brain._run_v15_profile(
            'Search the web for a niche romance anime and provide a screenshot.', 'chat-2',
            'diagonal', structured_thinking=True)
        self.assertIn('Here is the anime page and screenshot.', result)
        self.assertIn('![Browser screenshot](</tmp/anime.png>)', result)
        self.assertEqual(executed, [search, navigate, observe])
        self.assertIn('ACTION: search', model_inputs[0]['messages'][0]['content'])


if __name__ == "__main__":
    unittest.main()
