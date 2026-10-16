import unittest

from orchestrator_v15.thinking_controller import ThinkingController


class FollowthroughTests(unittest.TestCase):
    def test_research_handoff_uses_observed_sources_and_form(self):
        base = 'https://example.test'
        controller = ThinkingController('Research the linked plans, compare them, then open the recommendation form and save the result.')
        hub = {'url': base + '/brief', 'text': 'Compare plans', 'status': 200,
               'elements': [
                   {'role': 'link', 'name': 'Plan Amber details', 'href': base + '/amber'},
                   {'role': 'link', 'name': 'Plan Blue details', 'href': base + '/blue'},
                   {'role': 'link', 'name': 'Recommendation form', 'href': base + '/record'},
               ]}
        controller.observations.append(hub)
        controller.observations.append({'url': base + '/amber', 'text': 'Amber evidence', 'status': 200})
        self.assertIsNone(controller.research_ready_form_url())
        self.assertEqual(controller.research_next_source_url(), base + '/blue')
        self.assertIn('Read all linked source', controller.preflight(
            '[BROWSER_TOOL]\nACTION: navigate\nURL: ' + base + '/record\n[/BROWSER_TOOL]'))
        controller.observations.append({'url': base + '/blue', 'text': 'Blue evidence', 'status': 200})
        self.assertEqual(controller.research_ready_form_url(), base + '/record')
        controller.observations.append(hub)
        self.assertIn('instead of reopening', controller.preflight(
            '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Plan Amber details\n[/BROWSER_TOOL]'))
        controller.observations.append({'url': base + '/record', 'text': 'Form', 'status': 200,
                                        'elements': [{'role': 'textbox', 'name': 'Evidence'},
                                                     {'role': 'combobox', 'name': 'Plan', 'options': [
                                                         {'value': 'B', 'label': 'Plan Blue'}]}]})
        self.assertIsNone(controller.research_ready_form_url())
        self.assertFalse(controller.ready_to_finish_interaction())
        controller.action_fields = {
            0: {'ACTION': 'act', 'OPERATION': 'select_option', 'TARGET_NAME': 'Plan', 'VALUE': 'B'},
            1: {'ACTION': 'act', 'OPERATION': 'fill', 'TARGET_NAME': 'Evidence', 'VALUE': '$39, 94%, 2 days'},
            2: {'ACTION': 'act', 'OPERATION': 'click', 'TARGET_NAME': 'Save recommendation'},
        }
        controller.observations.append({'url': base + '/save', 'text': 'Saved result. Submission recorded exactly once.'})
        self.assertEqual(controller.verified_saved_work_answer(),
                         'Saved and verified Plan Blue. Recorded evidence: $39, 94%, 2 days')

    def test_no_handoff_without_source_evidence(self):
        controller = ThinkingController('Compare two plans and fill the form.')
        controller.observations.append({'url': 'https://example.test/brief', 'text': 'Brief',
                                        'elements': [
                                            {'role': 'link', 'name': 'Plan A details', 'href': 'https://example.test/a'},
                                            {'role': 'link', 'name': 'Plan B details', 'href': 'https://example.test/b'},
                                            {'role': 'link', 'name': 'Work form', 'href': 'https://example.test/form'},
                                        ]})
        self.assertIsNone(controller.research_ready_form_url())


if __name__ == '__main__':
    unittest.main()
