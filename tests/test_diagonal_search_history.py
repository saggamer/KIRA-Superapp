import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from actions import KiraActionsMixin
from diagonal_browser.kira_client import BrowserMCPClient
from diagonal_browser.search_history import save_history
from orchestrator_v15.adapter_router import channel_for_request
from orchestrator_v15.spark_tools import native_browser_call_to_block
from orchestrator_v15.thinking_controller import ThinkingController


class SearchHistoryIntegrationTests(unittest.TestCase):
    def test_natural_history_requests_route_to_browser_tools(self):
        for request in ('What did you search for yesterday?', 'What have you searched about MCP?',
                        'Read the saved pages from our earlier research', 'Show my search history'):
            with self.subTest(request=request):
                self.assertEqual(channel_for_request(request), 'browser')
        self.assertEqual(channel_for_request('Create slides from earlier research'), 'control')

    def test_native_model_and_controller_accept_history_without_opening_a_page(self):
        request = 'Search history for https://example.com and read the saved text'
        self.assertEqual(channel_for_request(request), 'browser')
        controller = ThinkingController(request)
        search = '[BROWSER_TOOL]\nACTION: search_history\nQUERY: example\n[/BROWSER_TOOL]'
        self.assertIsNone(controller.preflight(search))
        controller.record(search, 'BROWSER_TOOL:\n' + json.dumps({'entries': [{'historyId': 'saved-id'}]}))
        native = '<tool_call>BROWSER_TOOL<arg_key>action</arg_key><arg_value>read_history</arg_value><arg_key>history_id</arg_key><arg_value>saved-id</arg_value></tool_call>'
        read = native_browser_call_to_block(native)
        self.assertIsNotNone(read)
        self.assertIsNone(controller.preflight(read))
        self.assertIn('HISTORY_ID', controller.preflight(read.replace('saved-id', 'invented-id')))
        controller.record(read, 'BROWSER_TOOL:\n' + json.dumps({'cached': True, 'historyId': 'saved-id', 'text': 'Earlier source text'}))
        self.assertIsNone(controller.preflight('The saved source said: Earlier source text.'))

    def test_regular_web_search_and_fetch_write_shared_history(self):
        agent = KiraActionsMixin()
        agent._log_agentic_event = Mock()
        agent._web_fetch_html_source = Mock(return_value=('<title>Orchids</title><p>Orchid evidence</p>', 'fixture', 'text/html'))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'DIAGONAL_PROFILE_DIR': directory}):
            with patch.object(agent, '_extract_web_links', return_value=[{'title': 'Orchids', 'url': 'https://example.com/orchids'}]):
                agent._web_search_tool('orchid research', visible=False)
            agent._web_fetch_tool('https://example.com/orchids')
            records = [json.loads(file.read_text()) for file in (Path(directory) / 'search_history').glob('*.json')]
            self.assertEqual(sum(record['kind'] == 'search' for record in records), 1)
            self.assertEqual(sum(record['kind'] == 'page' for record in records), 2)
            self.assertTrue(all(record['source'] == 'llm' for record in records))
            self.assertTrue(any('Orchid evidence' in record.get('text', '') for record in records))

    @unittest.skipUnless(shutil.which('node') and (Path(__file__).parents[1] / 'diagonal_browser/node_modules').is_dir(), 'Local Node MCP dependencies required')
    def test_real_mcp_search_survives_restart_and_reads_without_chrome(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'ddgs-fixture'
            script.write_text(f'#!{sys.executable}\nimport json\nprint(json.dumps([{{"title":"Orchid source","url":"https://example.com/orchids","snippet":"saved botanical evidence"}}]))\n')
            script.chmod(0o700)
            with patch.dict(os.environ, {'DIAGONAL_PROFILE_DIR': directory, 'DIAGONAL_DDGS_BIN': str(script), 'DIAGONAL_CHROME_BIN': '/nonexistent-chrome-history-must-still-work'}):
                client = BrowserMCPClient()
                try:
                    search = client.call('search', {'query': 'orchid care'})
                    history_id = search['historyId']
                    client.close()
                    found = client.call('search_history', {'query': 'botanical evidence', 'source': 'llm'})
                    self.assertEqual(found['entries'][0]['historyId'], history_id)
                    cached = client.call('read_history', {'historyId': history_id})
                    self.assertTrue(cached['cached'])
                    self.assertEqual(cached['results'][0]['resultIndex'], 0)
                    self.assertTrue(client.call('revisit_history', {'historyId': history_id})['cached'])
                    page_id = save_history(kind='page', url='https://example.com/orchids', text='page from another process')
                    self.assertIn('another process', client.call('read_history', {'historyId': page_id})['text'])
                finally:
                    client.close()


if __name__ == '__main__':
    unittest.main()
