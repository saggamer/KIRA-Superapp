import json
from pathlib import Path
import sys
import tempfile
import unittest

from orchestrator_v15.mcp_live import run


class LiveMCPTests(unittest.TestCase):
    def test_connected_app_lists_and_calls_read_only_tool(self):
        fixture = Path(__file__).parent / 'fixtures' / 'mcp_notes_server.py'
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'kira_mcp_app.json'
            config.write_text(json.dumps({'mcpServers': {'test-notes': {
                'command': sys.executable, 'args': [str(fixture)], 'env': {},
            }}}))
            listed = run(config, 'test-notes')
            self.assertTrue(listed['ok'], listed)
            self.assertIn('search_notes', [tool['name'] for tool in listed['tools']])
            found = run(config, 'test-notes', tool_name='search_notes',
                        arguments={'query': 'charger'})
            self.assertTrue(found['ok'], found)
            self.assertIn('charger', str(found['content']))
            blocked = run(config, 'test-notes', tool_name='delete_note',
                          arguments={'title': 'trip'})
            self.assertFalse(blocked['ok'])
            self.assertIn('not authorized', blocked['error'])

    def test_unconfigured_server_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(Path(directory) / 'missing.json', 'notes')
            self.assertFalse(result['ok'])


if __name__ == '__main__':
    unittest.main()
