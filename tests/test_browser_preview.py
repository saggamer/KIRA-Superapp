import sys
import types
import unittest
from unittest.mock import Mock, patch
sys.modules.setdefault('webview', types.SimpleNamespace())
from interface import KiraBrain
from diagonal_browser.kira_client import BrowserMCPClient


class BrowserPreviewTests(unittest.TestCase):
    def test_preview_does_not_start_browser_and_does_not_wait_for_action_lock(self):
        client = BrowserMCPClient()
        client._start = Mock(side_effect=AssertionError('No browser startup'))
        self.assertFalse(client.preview()['available'])
        client.lock.acquire()
        try:
            self.assertTrue(client.preview()['busy'])
        finally:
            client.lock.release()
        client._start.assert_not_called()

    def test_preview_frames_are_rate_limited(self):
        client = BrowserMCPClient()
        client.process = Mock()
        client.process.poll.return_value = None
        client._send = Mock(return_value={'content': [{'type': 'text', 'text': '{"available":true,"dataUrl":"data:image/jpeg;base64,AAAA"}'}]})
        self.assertEqual(client.preview(), client.preview())
        client._send.assert_called_once()

    def test_preview_is_scoped_to_active_task_chat(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain._browser_preview_active = True
        brain._browser_preview_chat_id = 'task'
        with patch('diagonal_browser.kira_client.browser_preview', return_value={'available': True}) as preview:
            self.assertFalse(brain.get_browser_preview('other')['active'])
            preview.assert_not_called()
            self.assertTrue(brain.get_browser_preview('task')['available'])
            brain._browser_preview_active = False
            self.assertFalse(brain.get_browser_preview('task')['active'])


if __name__ == '__main__':
    unittest.main()
