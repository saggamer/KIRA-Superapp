import unittest

from orchestrator_v15.mcp_policy import (apply_requested, live_call_requested,
                                         normalize_live_call, broaden_empty_search)


class MCPApplyScopeTests(unittest.TestCase):
    def test_explicit_apply(self):
        self.assertTrue(apply_requested('Apply this draft to cursor_project now.'))
        self.assertTrue(apply_requested('Please enable this connector.'))

    def test_draft_and_negative_requests(self):
        for request in (
            'Create a draft, but do not apply it.',
            'Do not connect this server yet.',
            'Ask me before applying this draft.',
            'Can you show the MCP targets and explain how to apply one?',
        ):
            with self.subTest(request=request):
                self.assertFalse(apply_requested(request))

    def test_live_call_detection(self):
        self.assertTrue(live_call_requested('Use the MCP fixture.lookup tool now.'))
        self.assertTrue(live_call_requested('Invoke an MCP server tool for this record.'))
        self.assertTrue(live_call_requested('Use the connected app to find my note.'))
        self.assertFalse(live_call_requested('Create an inactive MCP connector draft.'))
        self.assertFalse(live_call_requested('Apply MCP draft /tmp/fixture-notes.json to cursor_project.'))

    def test_observed_mcp_wrapper_repair(self):
        malformed = ('[BROWSER_TOOL]\nMCP_CALL\nTARGET: test-notes\n'
                     'TOOL: search_notes\nARGS_JSON: {"query":"charger"}\n[/BROWSER_TOOL]')
        repaired = normalize_live_call(malformed, 'test-notes', {'search_notes'})
        self.assertIn('[MCP_CALL]', repaired)
        self.assertIn('SERVER: test-notes', repaired)
        self.assertEqual(normalize_live_call(malformed, 'test-notes', {'other_tool'}), malformed)

    def test_empty_read_only_search_has_one_grounded_broader_query(self):
        self.assertEqual(broaden_empty_search('search_notes', {'query': 'charger reminder'},
                                              {'content': ['']}), {'query': 'charger'})
        self.assertIsNone(broaden_empty_search('delete_note', {'query': 'charger reminder'},
                                               {'content': ['']}))
        self.assertIsNone(broaden_empty_search('search_notes', {'query': 'charger reminder'},
                                               {'content': ['Found it']}))


if __name__ == '__main__':
    unittest.main()
