from pathlib import Path
import queue
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault('webview', types.SimpleNamespace())
from actions import KiraActionsMixin
from interface import KiraBrain


class DesktopFollowupAndVisionWaitTests(unittest.TestCase):
    def test_vision_process_has_no_wall_clock_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            app = KiraActionsMixin()
            app.app_root = directory
            image = Path(directory) / 'image.png'
            image.write_bytes(b'fixture image')
            app._resolve_path = lambda path: str(path)
            app._truncate = lambda text, limit=1000: str(text)[:limit]
            with patch('actions.subprocess.run', return_value=types.SimpleNamespace(
                    returncode=0, stdout='IMAGE_ANALYSIS:\nAn app icon is visible.', stderr='')) as process:
                result = app._run_image_analysis_blocks(f'[IMAGE_ANALYZE]\nPATH: {image}\n[/IMAGE_ANALYZE]', 'chat')
            self.assertIn('An app icon is visible', result)
            self.assertIsNone(process.call_args.kwargs['timeout'])

    def test_vision_watchdog_waits_for_completion_but_other_tools_keep_their_timeout(self):
        for block, expected in [('[IMAGE_ANALYZE]\nPATH: /tmp/image.png\n[/IMAGE_ANALYZE]', None),
                                ('[OPEN]\nTARGET: Finder\n[/OPEN]', 38)]:
            app = KiraBrain.__new__(KiraBrain)
            app.response_queue = queue.Queue()
            app._run_agentic_capabilities = Mock(return_value='completed')
            joined = []
            def capture(*args, **kwargs):
                return types.SimpleNamespace(start=kwargs['target'], join=lambda timeout: joined.append(timeout), is_alive=lambda: False)
            with patch('interface.threading.Thread', side_effect=capture):
                app._run_agentic_capabilities_with_watchdog(block, 'task', 'chat', 0)
            self.assertEqual(joined, [expected])

    def test_open_it_without_a_generated_artifact_goes_back_to_the_model(self):
        app = KiraActionsMixin()
        app._latest_artifact_for_followup = lambda *_: ''
        self.assertEqual(app._handle_artifact_followup('open it', 'chat'), '')
        for prompt in ('open Word', 'open PowerPoint', 'open password manager'):
            self.assertEqual(app._handle_artifact_followup(prompt, 'chat'), '')
        self.assertIn('cannot find a generated artifact', app._handle_artifact_followup('open the report', 'chat'))

    def test_recent_app_discussion_does_not_open_an_older_artifact(self):
        app = KiraActionsMixin()
        app._latest_artifact_for_followup = lambda *_: '/tmp/old.pdf'
        app._load_chat_messages = lambda *_: [{'role': 'assistant', 'content': 'Which cloned project do you mean?'}]
        app._open_target_tool = Mock()
        self.assertEqual(app._handle_artifact_followup('open it', 'chat'), '')
        app._open_target_tool.assert_not_called()
        app._load_chat_messages = lambda *_: [{'role': 'assistant', 'content': 'Created `/tmp/old.pdf`.'}]
        app._open_target_tool.return_value = 'Opened'
        self.assertIn('Opened', app._handle_artifact_followup('open it', 'chat'))
        app._open_target_tool.assert_called_once_with('/tmp/old.pdf')


if __name__ == '__main__':
    unittest.main()
