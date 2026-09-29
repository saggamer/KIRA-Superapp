import unittest
from kira_live.action_contracts import wants_search, wants_docx, search_query, search_wants_open, docx_contract, tool_acknowledgement


class LiveActionContractsTests(unittest.TestCase):
    def test_followup_search_keeps_topic(self):
        turns = [{'role': 'user', 'content': 'I want horror spooky midnight camp stories.'}]
        query = search_query('Browser tool to search it using the tree.', turns)
        self.assertIn('horror', query)
        self.assertNotIn('using the tree', query)

    def test_explicit_docx_and_search_intents(self):
        self.assertTrue(wants_docx('Create a small Word document on stories.'))
        self.assertTrue(wants_search('Search other stories on good websites.'))
        self.assertFalse(wants_docx('Tell me a spooky story.'))

    def test_followup_open_requires_prior_same_search_request(self):
        turns = [{'role': 'user', 'content': 'Search horror websites and open it.'}]
        self.assertTrue(search_wants_open('Use the browser tool to search it.', turns))
        self.assertFalse(search_wants_open("Search it but do not open it.", turns))

    def test_generated_document_has_four_real_paragraph_fields(self):
        body = ' '.join(('A camper heard a distant whisper beyond the midnight camp fire and followed it into the silent horror forest. ' * 6).split())
        spec = docx_contract(body)
        self.assertEqual(spec.count('PARAGRAPH:'), 4)
        self.assertTrue(spec.endswith('[/NATIVE_DOCX]'))
        with self.assertRaises(ValueError):
            docx_contract("I cannot create files.")

    def test_web_ack_uses_returned_links_not_model_denial(self):
        evidence = 'WEB_SEARCH:\nReal result links:\n1. Stories\nhttps://example.com/stories\nFetched top pages:\nignored'
        self.assertIn('https://example.com/stories', tool_acknowledgement(evidence))
        self.assertIsNone(tool_acknowledgement('PERMISSION REQUIRED: approve first'))
        self.assertIsNone(tool_acknowledgement('BLOCKED SAFELY: no action'))

    def test_requested_open_uses_returned_url_through_tree(self):
        from voice import KiraVoiceMixin
        brain = KiraVoiceMixin()
        calls = []
        brain._extract_agentic_blocks = lambda text: text
        brain._agentic_branch_labels_from_text = lambda blocks: ["Web"]
        brain._emit_agent_progress = lambda *args: None
        brain._record_text_execution_evidence = lambda *args, **kwargs: None
        def runner(blocks, request, chat, scope):
            calls.append((blocks, scope))
            if '[WEB_OPEN]' in blocks:
                return 'WEB_OPEN: Opened `https://example.com/story` in the system default browser.'
            return 'WEB_SEARCH:\nReal result links:\n1. Story\nhttps://example.com/story\nFetched top pages:\ntext'
        brain._run_agentic_capabilities_with_watchdog = runner
        evidence = brain._execute_live_tree_output('[WEB_SEARCH]\nQUERY: horror stories\n[/WEB_SEARCH]', 'Find stories and open it.', 'fixture')
        self.assertEqual(len(calls), 2)
        self.assertIn('URL: https://example.com/story', calls[1][0])
        self.assertEqual(calls[1][1], 'native_live_tree')
        self.assertIn('opened the first result', tool_acknowledgement(evidence))
        calls.clear()
        brain._execute_live_tree_output('[WEB_SEARCH]\nQUERY: horror stories\n[/WEB_SEARCH]', 'Find stories but do not open them.', 'fixture')
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
