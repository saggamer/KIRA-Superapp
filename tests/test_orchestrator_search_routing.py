import queue
import sys
import types
import unittest
import threading
from unittest.mock import Mock, patch

sys.modules.setdefault('webview', types.SimpleNamespace())
from interface import KiraBrain
from orchestrator_v15.adapter_router import channel_for_request
from orchestrator_v15.mcp_policy import draft_requested


class NaturalSearchRoutingTests(unittest.TestCase):
    def test_public_lookup_phrasing_routes_to_browser(self):
        for request in (
            'search me the best boba tea cafe in india',
            'Search for boba tea cafes in India',
            'Please search me a good cafe nearby',
            'Can you look up boba tea cafes?',
            'Find me the best boba tea cafe in India',
            'search the web for MCP research papers',
        ):
            with self.subTest(request=request):
                self.assertEqual(channel_for_request(request), 'browser')

    def test_local_mcp_and_artifact_requests_keep_their_channels(self):
        for request in (
            'Search my notes in the connected app',
            'Search files for cafe receipts',
            'Search the repository for a parser',
            'Search MCP tools for my note',
            'Search installed apps for a browser',
            'Search MCP connectors for my notes',
            'Create a presentation researching cafes in India',
            'Create an inactive MCP draft for the fixture server',
        ):
            with self.subTest(request=request):
                self.assertEqual(channel_for_request(request), 'control')

    def test_draft_creation_needs_requested_setup_not_a_search_or_inspection(self):
        for request in ('search me the best boba tea cafe in india',
                        'Inspect the MCP bootstrap draft; do not apply it.',
                        'Find research about MCP', 'Do not create an MCP draft'):
            self.assertFalse(draft_requested(request), request)
        self.assertTrue(draft_requested('Create a filesystem MCP bootstrap draft for cursor_project.'))
        self.assertTrue(draft_requested('Draft an MCP connector for the notes server.'))

    def brain(self, outputs):
        brain = KiraBrain.__new__(KiraBrain)
        brain.v15_model_tool_planning = False  # This fixture tests execution after tool selection.
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda *_a: None
        brain._load_chat_messages = lambda *_a: []
        brain._smart_memory_context = lambda *_a, **_k: ''
        brain._truncate = lambda text, length=1000: str(text)[:length]
        brain._log_agentic_event = lambda *_a: None
        captured, actions = [], []
        def reply(payload, timeout=60):
            captured.append(payload)
            return {'proposals': [outputs.pop(0)]}
        brain._model_worker_request = reply
        brain._run_agentic_capabilities_with_watchdog = lambda action, *_a: (
            actions.append(action) or 'BROWSER_TOOL:\n{"results":[{"title":"Cafe","url":"https://example.test/cafe"}]}')
        return brain, captured, actions

    def test_screenshot_prompt_searches_instead_of_creating_a_connector(self):
        search = '[BROWSER_TOOL]\nACTION: search\nQUERY: best boba tea cafe in India\n[/BROWSER_TOOL]'
        brain, captured, actions = self.brain([search, 'I found a cafe result: https://example.test/cafe'])
        answer = brain._run_v15_profile('search me the best boba tea cafe in india', 'routing-test', 'diagonal', orchestral_candidate=True)
        self.assertEqual(actions, [search])
        self.assertIn('https://example.test/cafe', answer)
        self.assertEqual(brain.last_v15_skill_audit['active'], ['browser-research'])
        self.assertNotIn('MCP_BOOTSTRAP', captured[0]['messages'][0]['content'])

    def test_stale_mcp_answer_does_not_authorize_setup_on_a_search(self):
        bad = '[MCP_BOOTSTRAP]\nTARGET: cursor_project\n[/MCP_BOOTSTRAP]'
        search = '[BROWSER_TOOL]\nACTION: search\nQUERY: boba tea cafes India\n[/BROWSER_TOOL]'
        brain, _, actions = self.brain([bad, search, 'Cafe result: https://example.test/cafe'])
        brain._load_chat_messages = lambda *_a: [
            {'role': 'assistant', 'content': 'MCP bootstrap draft created: old-draft.json'},
        ]
        answer = brain._run_v15_profile('search me the best boba tea cafe in india', 'stale-test', 'medium')
        self.assertEqual(actions, [search])
        self.assertIn('https://example.test/cafe', answer)

    def test_unrelated_bootstrap_proposal_cannot_create_a_file_even_in_control_profile(self):
        bad = '[MCP_BOOTSTRAP]\nTARGET: cursor_project\n[/MCP_BOOTSTRAP]'
        brain, _, actions = self.brain([bad] * 12)
        answer = brain._run_v15_profile('Tell me a joke.', 'scope-test', 'diagonal')
        self.assertEqual(actions, [])
        self.assertIn('no connector draft was created', answer)

    def test_worker_deadline_does_not_depend_on_wall_clock(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.v15_model_tool_planning = False  # This fixture tests execution after tool selection.
        brain.model_worker_io_lock = threading.Lock()
        brain.model_worker_process = Mock()
        brain.model_worker_process.poll.return_value = None
        brain.model_worker_request_counter = 0
        brain.model_worker_pending_responses = {}
        brain.model_worker_response_queue = Mock()
        brain.model_worker_response_queue.get.side_effect = AssertionError('Deadline should expire before waiting')
        brain._mark_model_worker_dead = lambda reason: setattr(brain, 'model_worker_last_error', reason)
        with patch('interface.time.time', return_value=0), patch('interface.time.monotonic', side_effect=[0, 2]):
            with self.assertRaisesRegex(RuntimeError, 'timed out after 1s'):
                brain._model_worker_request({'type': 'propose_v15'}, timeout=1)


if __name__ == '__main__':
    unittest.main()
