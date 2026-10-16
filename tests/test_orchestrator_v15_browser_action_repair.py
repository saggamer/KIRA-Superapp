import unittest

from orchestrator_v15.browser_action_repair import repair_observed_action
from orchestrator_v15.thinking_controller import ThinkingController
from orchestrator_v15.screenshot_answer import with_screenshot


class RepairTests(unittest.TestCase):
    def test_compact_click_requires_fresh_unique_target_and_matching_tab(self):
        answer = ('[BROWSER_TOOL]\nACTION: click\nROLE: button\n'
                  'NAME: Submit purchase\nTABID: tab-1\n[/BROWSER_TOOL]')
        controller = ThinkingController('Submit the purchase exactly once.')
        controller.observations.append({'tabId': 'tab-1', 'elements': [
            {'role': 'button', 'name': 'Submit purchase'}]})
        repaired = repair_observed_action(answer, controller)
        self.assertIn('ACTION: act\nOPERATION: click', repaired)
        self.assertIn('TARGET_NAME: Submit purchase', repaired)
        self.assertIsNone(controller.preflight(repaired))
        unapproved = ThinkingController('Ask me before paying for this item.')
        unapproved.observations = controller.observations
        self.assertIn('not authorized', unapproved.preflight(
            repair_observed_action(answer, unapproved)))
        controller.observations.append({'tabId': 'tab-2', 'elements': [
            {'role': 'button', 'name': 'Submit purchase'}]})
        self.assertEqual(repair_observed_action(answer, controller), answer)
        controller.observations[-1]['tabId'] = 'tab-1'
        controller.observations[-1]['elements'].append(
            {'role': 'button', 'name': 'Submit purchase'})
        self.assertEqual(repair_observed_action(answer, controller), answer)

    def test_visible_link_target_alias(self):
        controller = ThinkingController('Follow Delivery details')
        controller.observations.append({'elements': [{'role': 'link', 'name': 'Delivery details'}]})
        answer = '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET: Delivery details\n[/BROWSER_TOOL]'
        repaired = repair_observed_action(answer, controller)
        self.assertIn('TARGET_NAME: Delivery details', repaired)
        self.assertIn('TARGET_ROLE: link', repaired)
        self.assertIsNone(controller.preflight(repaired))

    def test_missing_select_value_uses_observed_option(self):
        controller = ThinkingController('Choose Newest in Sort by')
        controller.observations.append({'elements': [{'role': 'combobox', 'name': 'Sort by',
            'options': [{'label': 'Default', 'value': 'default'}, {'label': 'Newest', 'value': 'newest'}]}]})
        answer = '[BROWSER_TOOL]\nACTION: act\nOPERATION: select_option\nTARGET: act\n[/BROWSER_TOOL]'
        repaired = repair_observed_action(answer, controller)
        self.assertIn('VALUE: newest', repaired)
        self.assertIn('TARGET_NAME: Sort by', repaired)
        self.assertIsNone(controller.preflight(repaired))

    def test_unobserved_target_stays_invalid(self):
        controller = ThinkingController('Click Continue')
        controller.observations.append({'elements': [{'role': 'button', 'name': 'Cancel'}]})
        answer = '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET: Continue\n[/BROWSER_TOOL]'
        self.assertEqual(repair_observed_action(answer, controller), answer)
        self.assertIsNotNone(controller.preflight(answer))

    def test_css_name_alias_requires_exact_observed_role_and_name(self):
        answer = ('[BROWSER_TOOL]\nACTION: act\nOPERATION: click\n'
                  'TARGET: button[name="Submit purchase"]\n[/BROWSER_TOOL]')
        controller = ThinkingController('Submit the purchase exactly once.')
        controller.observations.append({'elements': [{'role': 'button', 'name': 'Submit purchase'}]})
        repaired = repair_observed_action(answer, controller)
        self.assertIn('TARGET_ROLE: button', repaired)
        self.assertIn('TARGET_NAME: Submit purchase', repaired)
        self.assertIsNone(controller.preflight(repaired))
        unapproved = ThinkingController('Ask me before paying for this item.')
        unapproved.observations = controller.observations
        self.assertIn('not authorized', unapproved.preflight(repair_observed_action(answer, unapproved)))
        missing = ThinkingController('Submit the purchase exactly once.')
        missing.observations.append({'elements': [{'role': 'button', 'name': 'Cancel'}]})
        self.assertEqual(repair_observed_action(answer, missing), answer)

    def test_interaction_result_ready_to_finish(self):
        controller = ThinkingController('Fill Lookup with Ada, click Find tasks, and report count')
        controller.observations.extend([{'text': 'No lookup run'}, {'text': 'Found 12 tasks for Ada'}])
        controller.action_fields[0] = {'ACTION': 'act', 'OPERATION': 'fill', 'TARGET_NAME': 'Lookup', 'VALUE': 'Ada'}
        controller.action_fields[1] = {'ACTION': 'act', 'OPERATION': 'click', 'TARGET_NAME': 'Find tasks'}
        self.assertTrue(controller.ready_to_finish_interaction())

    def test_search_url_requires_grounded_result(self):
        controller = ThinkingController('Search the web for the official Python documentation homepage and give its URL')
        controller.observations.append({'results': [
            {'title': 'Our Documentation | Python.org', 'url': 'https://www.python.org/doc/', 'snippet': 'Official docs'},
            {'title': 'Python 3 documentation', 'url': 'https://docs.python.org/', 'snippet': 'Official documentation'}]})
        self.assertEqual(controller.observed_search_url(), 'https://docs.python.org/')

    def test_follows_only_user_named_observed_link(self):
        controller = ThinkingController('Open https://example.org/overview, follow Delivery details')
        controller.observations.append({'elements': [
            {'role': 'link', 'name': 'Delivery details', 'href': 'https://example.org/delivery'},
            {'role': 'link', 'name': 'Delete account', 'href': 'https://example.org/delete'}]})
        self.assertEqual(controller.requested_observed_link(), 'https://example.org/delivery')

    def test_generic_open_website_does_not_follow_unrelated_named_link(self):
        controller = ThinkingController(
            'Search for the official Python documentation homepage, open that website, '
            'and report its page title and URL.')
        controller.observations.append({'elements': [
            {'role': 'link', 'name': 'Python', 'href': 'https://www.python.org/'},
            {'role': 'link', 'name': 'Docs', 'href': 'https://docs.python.org/'}]})
        self.assertIsNone(controller.requested_observed_link())

    def test_current_page_grounding_rejects_old_link_after_navigation(self):
        controller = ThinkingController('Research all plans, then save a recommendation.')
        controller.observations.append({'tabId': 'tab-1', 'elements': [
            {'role': 'link', 'name': 'Plan A details'}]})
        controller.observations.append({'tabId': 'tab-1', 'elements': [
            {'role': 'button', 'name': 'Save recommendation'}]})
        old_link = ('[BROWSER_TOOL]\nACTION: act\nOPERATION: click\n'
                    'TARGET_ROLE: link\nTARGET_NAME: Plan A details\n'
                    'TABID: tab-1\n[/BROWSER_TOOL]')
        self.assertIn('latest observed controls', controller.preflight(old_link))

    def test_explicit_research_followup_returns_to_work_page(self):
        target = 'http://127.0.0.1:12345/evidence'
        controller = ThinkingController('Search for official documentation, open it, then open '
                                        + target + ', fill the form, and save evidence.')
        controller.record('[BROWSER_TOOL]\nACTION: navigate\nURL: ' + target +
                          '\n[/BROWSER_TOOL]', 'BROWSER_TOOL:\n{"url":"' + target +
                          '","title":"Evidence","status":200}')
        controller.record('[BROWSER_TOOL]\nACTION: search\nQUERY: official documentation\n'
                          '[/BROWSER_TOOL]', 'BROWSER_TOOL:\n{"results":[]}')
        controller.record('[BROWSER_TOOL]\nACTION: navigate\nURL: https://docs.python.org/\n'
                          '[/BROWSER_TOOL]', 'BROWSER_TOOL:\n{"url":"https://docs.python.org/",'
                          '"title":"Python docs","status":200}')
        self.assertIn('URL: ' + target, controller.requested_revisit_after_research())

    def test_prior_observed_link_reopens_exact_href(self):
        controller = ThinkingController('Research all linked plans.')
        controller.observations.append({'tabId': 'tab-1', 'elements': [
            {'role': 'link', 'name': 'Plan B details',
             'href': 'https://example.test/source/b'}]})
        controller.observations.append({'tabId': 'tab-1', 'elements': [],
                                        'url': 'https://example.test/source/a'})
        click = ('[BROWSER_TOOL]\nACTION: act\nOPERATION: click\n'
                 'TARGET_ROLE: link\nTARGET_NAME: Plan B details\n[/BROWSER_TOOL]')
        repaired = repair_observed_action(click, controller)
        self.assertIn('ACTION: navigate', repaired)
        self.assertIn('https://example.test/source/b', repaired)
        self.assertIsNone(controller.preflight(repaired))

    def test_repeated_search_after_verified_source_and_form_is_rejected(self):
        target = 'http://127.0.0.1:12345/evidence'
        controller = ThinkingController('Search official docs, then open ' + target +
                                        ', fill the form, and save.')
        controller.actions = ['navigate', 'search', 'navigate', 'navigate']
        controller.observations = [
            {'url': target, 'status': 200, 'title': 'Form'},
            {'results': [{'url': 'https://docs.python.org/'}]},
            {'url': 'https://docs.python.org/', 'status': 200, 'title': 'Python docs'},
            {'url': target, 'status': 200, 'title': 'Form',
             'elements': [{'role': 'textbox', 'name': 'Page title'}]},
        ]
        search = '[BROWSER_TOOL]\nACTION: search\nQUERY: official docs\n[/BROWSER_TOOL]'
        self.assertIn('Fill its observed fields', controller.preflight(search))

    def test_observed_select_label_maps_to_value_and_repetition_is_rejected(self):
        controller = ThinkingController('Choose Plan B and save it.')
        controller.observations.append({'elements': [{'role': 'combobox', 'name': 'Plan',
            'value': 'A', 'options': [{'label': 'Plan A', 'value': 'A'},
                                    {'label': 'Plan B', 'value': 'B'}]}]})
        raw = ('[BROWSER_TOOL]\nACTION: act\nOPERATION: select_option\n'
               'TARGET_ROLE: combobox\nTARGET_NAME: Plan\nVALUE: Plan B\n[/BROWSER_TOOL]')
        repaired = repair_observed_action(raw, controller)
        self.assertIn('VALUE: B', repaired)
        self.assertIsNone(controller.preflight(repaired))
        controller.observations[-1]['elements'][0]['value'] = 'B'
        self.assertIn('already selected', controller.preflight(repaired))

    def test_saved_research_answer_requires_matching_source_and_confirmation(self):
        controller = ThinkingController('Research docs, fill Page title and Page URL, save evidence.')
        controller.observations = [
            {'title': 'Python Docs', 'url': 'https://docs.python.org/', 'status': 200},
            {'title': 'Saved result', 'text': 'Submission recorded exactly once.'},
        ]
        controller.action_fields = {
            0: {'ACTION': 'act', 'OPERATION': 'fill',
                'TARGET_NAME': 'Page title', 'VALUE': 'Python Docs'},
            1: {'ACTION': 'act', 'OPERATION': 'fill',
                'TARGET_NAME': 'Page URL', 'VALUE': 'https://docs.python.org/'},
            2: {'ACTION': 'act', 'OPERATION': 'click', 'TARGET_NAME': 'Save evidence'},
        }
        self.assertIn('Python Docs', controller.verified_saved_research_answer())
        controller.action_fields[1]['VALUE'] = 'https://invented.example/'
        self.assertIsNone(controller.verified_saved_research_answer())

    def test_no_submit_status_uses_page_evidence(self):
        controller = ThinkingController('Do not submit a purchase. Report whether it was submitted.')
        controller.observations.append({'visibleText': 'Order review\nStatus: not submitted.\nSubmit purchase'})
        self.assertIn('not submitted', controller.observed_no_submit_status())
        review = ThinkingController('Review the order total, but do not submit the purchase. Tell me the total.')
        review.observations = controller.observations
        self.assertIsNone(review.observed_no_submit_status())

    def test_missing_page_finishes_from_http_status(self):
        controller = ThinkingController('Tell me whether the page loaded successfully')
        controller.record('[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.org/missing\n[/BROWSER_TOOL]',
                          'BROWSER_TOOL ERROR:\n{"status":404,"text":"Not found","url":"https://example.org/missing"}')
        self.assertIn('404', controller.observed_load_failure())

    def test_screenshot_answer_has_renderable_image(self):
        path = '/tmp/browser image.png'
        malformed = 'Here is the screenshot:\n[Screenshot of page]\n[' + path + ']'
        result = with_screenshot(malformed, path)
        self.assertIn('![Browser screenshot](<' + path + '>)', result)
        self.assertNotIn('[Screenshot of page]', result)
        self.assertEqual(with_screenshot('Captured:\n' + path, path).count(path), 1)


if __name__ == '__main__':
    unittest.main()
