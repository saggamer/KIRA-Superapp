import json
import unittest

from orchestrator_v15.browser_action_repair import repair_observed_action
from orchestrator_v15.browser_state import page_state
from orchestrator_v15.fast_mlx import browser_evidence_snapshot
from orchestrator_v15.thinking_controller import ThinkingController


def record(controller, action, observation):
    controller.record(action, 'BROWSER_TOOL:\n' + json.dumps(observation))


def navigate(url):
    return f'[BROWSER_TOOL]\nACTION: navigate\nURL: {url}\n[/BROWSER_TOOL]'


class BrowserHandoffTests(unittest.TestCase):
    def controller(self, request='Find Cedar launch date and official source.'):
        return ThinkingController(request, diagonal_level=True, unlimited_steps=True)

    def test_relevant_earlier_evidence_survives_recent_context_and_later_pages(self):
        controller = self.controller()
        for index in range(12):
            record(controller, navigate(f'https://example.test/{index}'), {
                'url': f'https://example.test/{index}', 'title': 'Cedar launch date' if index == 0 else f'Unrelated page {index}',
                'text': 'Cedar launches on 21 November.' if index == 0 else f'General navigation {index}',
                'tabId': 'tab', 'observationId': str(index), 'elements': []})
        value = json.loads(browser_evidence_snapshot(controller).split('\n', 1)[1])
        self.assertFalse(any(row['url'].endswith('/0') for row in value['recent']))
        self.assertTrue(any('21 November' in ' '.join(row['excerpts'])
                            for row in value['working_state']['sources']))
        self.assertLessEqual(len(controller.working_state.sources), 8)

    def test_blocked_source_keeps_alternatives_from_older_observations(self):
        controller = self.controller('Read https://example.test/start and find Cedar launch date.')
        record(controller, navigate('https://example.test/start'), {
            'url': 'https://example.test/start', 'text': 'Sources',
            'elements': [{'role': 'link', 'name': 'Cedar official alternative', 'href': 'https://example.test/official'}]})
        for index in range(5):
            record(controller, navigate(f'https://example.test/{index}'), {
                'url': f'https://example.test/{index}', 'text': f'Page {index}'})
        record(controller, navigate('https://example.test/blocked'), {
            'url': 'https://example.test/blocked', 'title': 'Attention Required! | Cloudflare',
            'text': 'Verify you are human', 'status': 403})
        self.assertEqual(controller.working_state.current['state'], 'challenge')
        self.assertIn('alternative', controller.hint())
        self.assertIsNone(controller.preflight(navigate('https://example.test/official')))
        proposed = '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_ROLE: link\nTARGET_NAME: Cedar official alternative\n[/BROWSER_TOOL]'
        self.assertEqual(repair_observed_action(proposed, controller), navigate('https://example.test/official'))

    def test_inspection_preserves_facts_links_and_forms_without_inventing_click_targets(self):
        controller = self.controller()
        record(controller, '[BROWSER_TOOL]\nACTION: inspect_page\n[/BROWSER_TOOL]', {
            'tabId': 'tab', 'page': {'url': 'https://example.test', 'title': 'Cedar',
                'text': 'Cedar launches on 21 November.', 'headings': ['Launch'],
                'links': [{'text': 'Official detail', 'url': 'https://example.test/detail'}],
                'forms': [{'method': 'get', 'fields': [{'name': 'query', 'type': 'text'}]}]}})
        snapshot = json.loads(browser_evidence_snapshot(controller).split('\n', 1)[1])
        latest = snapshot['recent'][-1]
        self.assertIn('21 November', latest['text'])
        self.assertEqual(latest['elements'], [])
        self.assertEqual(latest['results'][0]['url'], 'https://example.test/detail')
        self.assertTrue(latest['forms'])
        self.assertIsNotNone(controller.preflight(
            '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Official detail\n[/BROWSER_TOOL]'))

    def test_index_binding_uses_only_current_real_observation(self):
        controller = self.controller()
        record(controller, navigate('https://example.test'), {
            'url': 'https://example.test', 'tabId': 'tab', 'observationId': 'fresh-id',
            'elements': [{'index': 17, 'role': 'button', 'name': 'Continue'}]})
        action = '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nINDEX: 17\n[/BROWSER_TOOL]'
        repaired = repair_observed_action(action, controller)
        self.assertIn('OBSERVATION_ID: fresh-id', repaired)
        self.assertIn('TABID: tab', repaired)
        self.assertNotIn('OBSERVATION_ID', repair_observed_action(action.replace('17', '99'), controller))
        stale = action.replace('INDEX: 17', 'INDEX: 17\nOBSERVATION_ID: old-id')
        self.assertIn('OBSERVATION_ID: old-id', repair_observed_action(stale, controller))
        conflict = action.replace('INDEX: 17', 'INDEX: 17\nTARGET_NAME: Submit payment')
        self.assertNotIn('OBSERVATION_ID', repair_observed_action(conflict, controller))

    def test_repeated_reads_require_recovery_instead_of_ending_the_task(self):
        controller = self.controller()
        record(controller, navigate('https://example.test'), {'url': 'https://example.test', 'text': ''})
        observe = '[BROWSER_TOOL]\nACTION: observe\n[/BROWSER_TOOL]'
        for _ in range(2):
            record(controller, observe, {'url': 'https://example.test', 'text': '', 'elements': []})
        self.assertIn('inspect_page', controller.preflight(observe))
        self.assertIsNone(controller.preflight('[BROWSER_TOOL]\nACTION: inspect_page\n[/BROWSER_TOOL]'))
        self.assertIsNone(controller.preflight(navigate('https://example.test/alternative')))

    def test_current_relevant_controls_and_sensitive_values_survive_compaction_correctly(self):
        controller = self.controller('Choose Coimbatore location.')
        elements = [{'index': i, 'role': 'button', 'name': f'Item {i}'} for i in range(60)]
        elements[55] = {'index': 55, 'role': 'combobox', 'name': 'Coimbatore location',
                        'options': [{'label': 'Coimbatore', 'value': 'cbe'}]}
        elements[0] = {'index': 0, 'role': 'textbox', 'name': 'Password', 'value': 'must-not-leak', 'sensitive': True}
        record(controller, navigate('https://example.test'), {'url': 'https://example.test', 'elements': elements})
        snapshot = browser_evidence_snapshot(controller)
        self.assertIn('Coimbatore location', snapshot)
        self.assertNotIn('must-not-leak', snapshot)
        self.assertLess(len(snapshot), 18500)

    def test_error_classification_and_recovery_are_observation_based(self):
        self.assertEqual(page_state({'title': 'About Cloudflare', 'text': 'Cloudflare documentation', 'url': 'https://example.test'}), 'ready')
        controller = self.controller()
        controller.record('[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nINDEX: 1\n[/BROWSER_TOOL]',
                          'BROWSER_TOOL ERROR: Stale browser observation. Observe the tab again.')
        self.assertEqual(controller.working_state.current['state'], 'stale')
        self.assertIn('Observe the intended tab', controller.hint())

    def test_large_pages_are_compacted_without_cutting_json_or_link_urls(self):
        controller = self.controller()
        urls = [f'https://example.test/{i}/' + 'a' * 1700 for i in range(40)]
        record(controller, navigate('https://example.test'), {
            'url': 'https://example.test', 'title': 'Cedar', 'text': 'x' * 20000,
            'elements': [{'index': i, 'role': 'combobox', 'name': f'Choice {i}', 'href': urls[i],
                          'options': [{'label': 'Label' * 100, 'value': 'v' * 1000} for _ in range(30)]}
                         for i in range(40)],
            'results': [{'url': url, 'title': 'Cedar official source', 'body': 'snippet' * 500} for url in urls]})
        snapshot = browser_evidence_snapshot(controller)
        value = json.loads(snapshot.split('\n', 1)[1])
        self.assertLess(len(snapshot), 18500)
        for item in value['working_state']['candidate_links']:
            self.assertIn(item['url'], urls)


if __name__ == '__main__':
    unittest.main()
