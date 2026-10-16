import io
import json
import unittest
from unittest.mock import Mock, patch

from diagonal_browser.kira_client import BrowserMCPClient


class BrowserMCPClientTests(unittest.TestCase):
    def test_non_json_stdout_does_not_discard_following_reply(self):
        client = BrowserMCPClient()
        client.process = Mock()
        client.process.stdin = io.StringIO()
        client.process.stdout = io.StringIO(
            'Chrome diagnostic\n'
            + json.dumps({'jsonrpc': '2.0', 'id': 1, 'result': {'ok': True}})
            + '\n'
        )
        with patch('diagonal_browser.kira_client.select.select', side_effect=lambda files, *_: (files, [], [])):
            self.assertEqual(client._send('ping'), {'ok': True})

    def test_close_terminates_child_process(self):
        client = BrowserMCPClient()
        process = Mock()
        process.poll.return_value = None
        process.stdin = io.StringIO()
        process.stdout = io.StringIO()
        client.process = process

        client.close()

        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=2)
        self.assertIsNone(client.process)


if __name__ == '__main__':
    unittest.main()
