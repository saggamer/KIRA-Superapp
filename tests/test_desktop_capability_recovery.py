import queue
import sys
import types
import unittest
from unittest.mock import patch
sys.modules.setdefault('webview', types.SimpleNamespace())
from interface import KiraBrain
from orchestrator_v15.desktop_harness import desktop_action_requested, local_desktop_request


class DesktopCapabilityRecoveryTests(unittest.TestCase):
    def brain(self, replies):
        app = KiraBrain.__new__(KiraBrain)
        app.v15_model_tool_planning = False
        app.response_queue = queue.Queue()
        app._evict_and_load = lambda *_: None
        app._load_chat_messages = lambda *_: [
            {'role': 'user', 'content': 'Open the cloned app in Finder.'},
            {'role': 'user', 'content': 'Its called Polyagent.'},
            {'role': 'user', 'content': 'open it'}]
        app._smart_memory_context = lambda *_a, **_k: ''
        app._truncate = lambda value, limit=1000: str(value)[:limit]
        app._log_agentic_event = lambda *_a, **_k: None
        responses = iter(replies)
        app._model_worker_request = lambda *_a, **_k: {'proposals': [next(responses)]}
        return app

    def test_false_capability_denial_recovers_to_observed_project_path(self):
        locate = '[LIST_DIR]\nPATH: ~/Downloads\n[/LIST_DIR]'
        open_folder = '[OPEN]\nTARGET: /tmp/Downloads/polyagent\n[/OPEN]'
        app = self.brain(['I cannot directly interact with your local file system or open applications like Finder.',
                          locate, open_folder, 'Opened the Polyagent folder in Finder.'])
        executed = []
        results = iter(['LIST_DIR:\n/tmp/Downloads/polyagent', 'Opened: /tmp/Downloads/polyagent'])
        app._run_agentic_capabilities_with_watchdog = lambda block, *_a: executed.append(block) or next(results)
        self.assertIn('Polyagent folder', app._run_v15_profile('open it', 'followup', 'medium'))
        self.assertEqual(executed, [locate, open_folder])

    def test_vision_wait_does_not_use_the_remaining_task_time(self):
        analyze = '[IMAGE_ANALYZE]\nPATH: /tmp/image.png\nQUESTION: Identify the app\n[/IMAGE_ANALYZE]'
        open_app = '[OPEN]\nTARGET: Finder\n[/OPEN]'
        app = self.brain([analyze, open_app, 'Opened Finder.'])
        clock, executed = [0.0], []
        def execute(block, *_a):
            executed.append(block)
            if block == analyze:
                clock[0] += 10000
                return 'IMAGE_ANALYSIS:\nApp identified.'
            return 'Opened: Finder'
        app._run_agentic_capabilities_with_watchdog = execute
        with patch('interface.time.monotonic', side_effect=lambda: clock[0]):
            self.assertEqual(app._run_v15_profile('Identify the app and open Finder.', 'vision', 'medium'), 'Opened Finder.')
        self.assertEqual(executed, [analyze, open_app])

    def test_correction_requires_a_current_action_not_a_mention_or_negation(self):
        history = [{'role': 'user', 'content': 'Open the cloned app in Finder.'}]
        self.assertTrue(desktop_action_requested('its called Polyagent', history))
        for text in ("Don't open Finder", 'Why did Finder open?', 'Explain opening apps', 'Finder opened unexpectedly'):
            self.assertFalse(desktop_action_requested(text, history))
        self.assertTrue(local_desktop_request('open it', history))
        self.assertFalse(local_desktop_request('Find good YouTube videos about Python.', history))
        self.assertFalse(local_desktop_request('Open https://example.test', history))


if __name__ == '__main__':
    unittest.main()
