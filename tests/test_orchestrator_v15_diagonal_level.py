import unittest

from orchestrator_v15.diagonal_level import select_proposal
from orchestrator_v15.thinking_controller import ThinkingController


class DiagonalLevelTests(unittest.TestCase):
    def test_repaired_authorized_branch_is_ranked_before_retry(self):
        controller = ThinkingController('Submit the purchase exactly once.', diagonal_level=True)
        controller.observations.append({'tabId': 'tab-1', 'elements': [
            {'role': 'button', 'name': 'Submit purchase'}]})
        malformed = ('[BROWSER_TOOL]\nACTION: click\nROLE: button\n'
                     'NAME: Submit purchase\nTABID: tab-1\n[/BROWSER_TOOL]')
        chosen, trace = select_proposal([malformed, 'The purchase was submitted.'], controller)
        self.assertIn('ACTION: act\nOPERATION: click', chosen)
        self.assertEqual(trace['valid'], 1)
        self.assertEqual(trace['selected_index'], 0)
        self.assertEqual(trace['answers'][0], malformed)
        self.assertIsNone(controller.preflight(chosen))

    def test_stale_recovery_does_not_rank_unfinished_narrative(self):
        controller = ThinkingController(
            'A previous click was stale. Take a fresh observation before clicking Run query; '
            'report whether it was applied.', diagonal_level=True)
        controller.observations.append({'tabId': 'tab-1', 'observationId': 'old',
                                        'elements': [{'role': 'button', 'name': 'Run query'}]})
        observe = '[BROWSER_TOOL]\nACTION: observe\nTABID: tab-1\n[/BROWSER_TOOL]'
        chosen, trace = select_proposal(['The query was applied.', observe], controller)
        self.assertEqual(chosen, observe)
        self.assertEqual(trace['valid'], 1)
        controller.fresh_observation = True
        self.assertIn('has not occurred', controller.preflight('The query was applied.'))

    def test_search_beats_premature_answer_and_unneeded_observe(self):
        controller = ThinkingController('Search the web for a niche romance anime.', diagonal_level=True)
        search = '[BROWSER_TOOL]\nACTION: search\nQUERY: niche romance anime\n[/BROWSER_TOOL]'
        chosen, trace = select_proposal([
            'Here is a niche romance anime.',
            '[BROWSER_TOOL]\nACTION: observe\nTABID: latest\n[/BROWSER_TOOL]',
            search,
        ], controller)
        self.assertEqual(chosen, search)
        self.assertEqual(trace['valid'], 1)

    def test_branch_selection_reuses_state_and_requires_navigation(self):
        controller = ThinkingController('Open https://example.test and capture a screenshot.', diagonal_level=True)
        proposals = [
            '[BROWSER_TOOL]\nACTION: observe\nSCREENSHOT: true\nTABID: tab-1\n[/BROWSER_TOOL]',
            '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]',
            'The screenshot is complete.',
        ]
        chosen, trace = select_proposal(proposals, controller)
        self.assertEqual(chosen, proposals[1])
        self.assertEqual(trace['valid'], 1)

    def test_unapproved_effect_loses_to_safe_action(self):
        controller = ThinkingController('Inspect the page and click Run query.', diagonal_level=True)
        controller.record('[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]',
                          'BROWSER_TOOL:\n{"tabId":"tab-1","text":"Run query Submit purchase"}')
        proposals = [
            '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Submit purchase\n[/BROWSER_TOOL]',
            '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Run query\n[/BROWSER_TOOL]',
        ]
        chosen, trace = select_proposal(proposals, controller)
        self.assertEqual(chosen, proposals[1])
        self.assertEqual(trace['valid'], 1)

    def test_unsupported_weekday_loses_to_grounded_answer(self):
        controller = ThinkingController('What hours is the library open?', diagonal_level=True)
        controller.record('[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]',
                          'BROWSER_TOOL:\n{"text":"Open Monday to Friday 09:00-18:00. Closed Sunday."}')
        chosen, _ = select_proposal([
            'Open Monday to Friday 09:00-18:00; closed Saturday and Sunday.',
            'Open Monday to Friday 09:00-18:00.',
        ], controller)
        self.assertEqual(chosen, 'Open Monday to Friday 09:00-18:00.')

    def test_grounded_answer_beats_repeated_capture(self):
        controller = ThinkingController('Show me an image of an animated movie.', diagonal_level=True)
        capture = '[BROWSER_TOOL]\nACTION: observe\nTABID: tab-1\nSCREENSHOT: true\n[/BROWSER_TOOL]'
        controller.record(capture, 'BROWSER_TOOL:\n{"tabId":"tab-1","title":"Movie",'
                          '"text":"An animated comedy.","screenshot":"/tmp/movie.png"}')
        answer = 'This is an animated comedy from the observed page.'
        chosen, _ = select_proposal([capture, answer], controller)
        self.assertEqual(chosen, answer)


if __name__ == '__main__':
    unittest.main()
