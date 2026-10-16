import unittest

from orchestrator_v15.thinking_controller import ThinkingController


class ThinkingControllerTests(unittest.TestCase):
    def test_public_site_title_and_url_finishes_from_loaded_page(self):
        controller = ThinkingController(
            'Search for the official Python documentation, open the website, and report its page title and URL.')
        self.assertIsNone(controller.observed_page_title_url())
        search = '[BROWSER_TOOL]\nACTION: search\nQUERY: Python documentation\n[/BROWSER_TOOL]'
        controller.record(search, 'BROWSER_TOOL:\n{"results":[{"url":"https://www.python.org/doc/"}]}')
        self.assertIsNone(controller.observed_page_title_url())
        navigate = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://www.python.org/doc/\n[/BROWSER_TOOL]'
        controller.record(navigate, 'BROWSER_TOOL:\n{"url":"https://www.python.org/doc/",'
                          '"title":"Our Documentation | Python.org","status":200,'
                          '"text":"Browse Python documentation"}')
        self.assertEqual(controller.observed_page_title_url(),
                         'Page title: Our Documentation | Python.org\nURL: https://www.python.org/doc/')

    def test_web_search_action_is_accepted_and_results_support_an_answer(self):
        controller = ThinkingController('Search the web for a niche romance anime.', diagonal_level=True)
        search = '[BROWSER_TOOL]\nACTION: search\nQUERY: niche romance anime\n[/BROWSER_TOOL]'
        self.assertIsNone(controller.preflight(search))
        controller.record(search, 'BROWSER_TOOL:\n{"query":"niche romance anime","results":[{"title":"Example","href":"https://example.test"}]}')
        self.assertIsNone(controller.preflight('The result is Example.'))

    def test_visual_recommendation_needs_capture_and_bounds_research(self):
        controller = ThinkingController('Find an animated comedy movie and provide an image to see.')
        self.assertTrue(controller.needs_screenshot)
        first = '[BROWSER_TOOL]\nACTION: search\nQUERY: animated comedy movie\n[/BROWSER_TOOL]'
        second = '[BROWSER_TOOL]\nACTION: search\nQUERY: animated comedy poster\n[/BROWSER_TOOL]'
        controller.record(first, 'BROWSER_TOOL:\n{"results":[{"href":"https://example.test/movie"}]}')
        controller.record(second, 'BROWSER_TOOL:\n{"results":[{"href":"https://example.test/poster"}]}')
        self.assertIn('Two searches already returned results', controller.preflight(second))
        self.assertIn('Navigate', controller.preflight('Try this movie.'))

    def test_open_browser_uses_new_tab_and_rejects_placeholder_tab_id(self):
        controller = ThinkingController('Open any browser on my laptop.', diagonal_level=True)
        open_tab = '[BROWSER_TOOL]\nACTION: new_tab\n[/BROWSER_TOOL]'
        self.assertIsNone(controller.preflight(open_tab))
        self.assertIn('actual tab ID', controller.preflight(
            '[BROWSER_TOOL]\nACTION: observe\nTABID: latest\n[/BROWSER_TOOL]'))
        controller.record(open_tab, 'BROWSER_TOOL:\n{"tabId":"tab-1","url":"about:blank"}')
        self.assertIsNone(controller.preflight('I opened a browser tab.'))

    def test_stale_click_requires_fresh_observation(self):
        controller = ThinkingController('A previous observation was stale. Click Run query after refreshing.')
        click = '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Run query\n[/BROWSER_TOOL]'
        self.assertIn('observe first', controller.preflight(click))
        observe = '[BROWSER_TOOL]\nACTION: observe\nTABID: tab-1\n[/BROWSER_TOOL]'
        controller.record(observe, 'BROWSER_TOOL:\n{"tabId":"tab-1","observationId":"new-1"}')
        self.assertIsNone(controller.preflight(click))

    def test_screenshot_requires_visual_evidence(self):
        controller = ThinkingController('Capture a screenshot of the page.')
        navigate = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]'
        controller.record(navigate, 'BROWSER_TOOL:\n{"tabId":"tab-1","title":"Example"}')
        self.assertIn('screenshot has not been observed', controller.preflight('The title is Example.'))
        visual = '[BROWSER_TOOL]\nACTION: observe\nSCREENSHOT: true\nTABID: tab-1\n[/BROWSER_TOOL]'
        controller.record(visual, 'BROWSER_TOOL:\n{"tabId":"tab-1","screenshot":"/tmp/shot.png"}')
        self.assertIsNone(controller.preflight('The title is Example.'))
        self.assertIn('already been captured', controller.preflight(visual))

    def test_malformed_operation_gets_bounded_retry(self):
        controller = ThinkingController('Sort a web page.')
        bad = '[BROWSER_TOOL]\nACTION: select_option\nTARGET_NAME: Sort by\n[/BROWSER_TOOL]'
        self.assertIn('Invalid ACTION', controller.preflight(bad))
        self.assertTrue(controller.can_retry())
        self.assertTrue(controller.can_retry())
        self.assertFalse(controller.can_retry())

    def test_open_requires_navigation_before_visual_observation(self):
        controller = ThinkingController('Open https://example.test then capture a screenshot.')
        visual = '[BROWSER_TOOL]\nACTION: observe\nSCREENSHOT: true\nTABID: tab-1\n[/BROWSER_TOOL]'
        self.assertIn('Navigate', controller.preflight(visual))

    def test_unapproved_effect_click_is_corrected(self):
        controller = ThinkingController('Open the page and click Run query after a stale observation.')
        controller.fresh_observation = True
        bad = '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Submit purchase\n[/BROWSER_TOOL]'
        self.assertIn('not authorized', controller.preflight(bad))

    def test_checkout_scope_and_action_format(self):
        click = ('[BROWSER_TOOL]\nACTION: act\nOPERATION: click\n'
                 'TARGET_ROLE: button\nTARGET_NAME: Submit purchase\nTABID: cart-1\n[/BROWSER_TOOL]')
        for request in ('Help me checkout this item.', 'Should I pay for this?',
                        'Do not purchase this; review it.', 'Ask me before paying.',
                        'Review the order without submitting.', 'Tell me how to buy this item.'):
            with self.subTest(request=request):
                self.assertIn('not authorized', ThinkingController(request).preflight(click))
        for request in ('Buy this item now.', 'Submit the purchase exactly once.',
                        'Pay for this order now.', 'Can you buy this item now?'):
            with self.subTest(request=request):
                self.assertIsNone(ThinkingController(request).preflight(click))
        malformed = click.replace('TARGET_ROLE: button\nTARGET_NAME: Submit purchase',
                                  'TARGET: button[name="Submit purchase"]')
        self.assertIn('TARGET is unsupported', ThinkingController('Buy this item now.').preflight(malformed))

    def test_ready_review_asks_once_without_submitting(self):
        controller = ThinkingController('Review the workshop seat and ask me before confirming the booking.')
        controller.observations.append({'visibleText': 'Workshop seat B4. Total $12. Status: not booked.',
                                        'elements': [{'role': 'button', 'name': 'Confirm booking'}]})
        answer = controller.observed_pending_effect_review()
        self.assertIn('$12', answer)
        self.assertIn('May I confirm booking?', answer)
        self.assertFalse(controller.effect_executed)
        explicit = ThinkingController('Book seat B4 now for $12.')
        explicit.observations = controller.observations
        self.assertIsNone(explicit.observed_pending_effect_review())

    def test_explicit_booking_still_requires_one_verified_click(self):
        click = ('[BROWSER_TOOL]\nACTION: act\nOPERATION: click\n'
                 'TARGET_ROLE: button\nTARGET_NAME: Confirm booking\nTABID: seat-6\n[/BROWSER_TOOL]')
        controller = ThinkingController('Book seat B4 now for $16.')
        controller.observations.append({'tabId': 'seat-6', 'elements': [
            {'role': 'button', 'name': 'Confirm booking'}]})
        self.assertIn('explicitly requested', controller.preflight('I did not book it.'))
        self.assertIsNone(controller.preflight(click))
        controller.record(click, 'BROWSER_TOOL:\n{"tabId":"seat-6","text":"Processing"}')
        self.assertIn('already occurred', controller.preflight(click))
        self.assertIn('observe', controller.required_action())

    def test_multiple_requested_urls_must_be_visited_before_answer(self):
        controller = ThinkingController('Compare https://example.test/a and https://example.test/b.',
                                        diagonal_level=True)
        self.assertEqual(controller.unvisited_urls(), ['https://example.test/a', 'https://example.test/b'])
        first = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/a\n[/BROWSER_TOOL]'
        controller.record(first, 'BROWSER_TOOL:\n{"url":"https://example.test/a","text":"A costs $39"}')
        self.assertIn('another URL', controller.preflight('A is cheaper.'))
        second = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/b\n[/BROWSER_TOOL]'
        controller.record(second, 'BROWSER_TOOL:\n{"url":"https://example.test/b","text":"B costs $49"}')
        self.assertIsNone(controller.preflight('A is cheaper.'))

    def test_two_observed_monthly_prices_finish_without_reopening_pages(self):
        controller = ThinkingController('Compare https://example.test/a and https://example.test/b. Which is cheaper and by how much?')
        first = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/a\n[/BROWSER_TOOL]'
        second = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/b\n[/BROWSER_TOOL]'
        controller.record(first, 'BROWSER_TOOL:\n{"url":"https://example.test/a","title":"Plan A","text":"Monthly price: $39."}')
        self.assertIsNone(controller.observed_price_comparison_answer())
        controller.record(second, 'BROWSER_TOOL:\n{"url":"https://example.test/b","title":"Plan B","text":"Monthly price: $49."}')
        self.assertIn('Plan A is cheaper by $10.00', controller.observed_price_comparison_answer())

    def test_navigation_can_follow_an_observed_link(self):
        controller = ThinkingController('Open https://example.test/overview and follow Delivery details.')
        first = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/overview\n[/BROWSER_TOOL]'
        controller.record(first, 'BROWSER_TOOL:\n{"url":"https://example.test/overview",'
                          '"elements":[{"role":"link","name":"Delivery details",'
                          '"href":"https://example.test/delivery"}]}')
        linked = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test/delivery\n[/BROWSER_TOOL]'
        self.assertIsNone(controller.preflight(linked))
        self.assertIn('returned by the browser', controller.preflight(
            '[BROWSER_TOOL]\nACTION: navigate\nURL: https://unseen.test/\n[/BROWSER_TOOL]'))

    def test_unsupported_weekday_is_rejected(self):
        controller = ThinkingController('What are the weekday opening hours?', diagonal_level=True)
        navigate = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]'
        controller.record(navigate, 'BROWSER_TOOL:\n{"text":"Open Monday to Friday 09:00-18:00. Closed Sunday."}')
        self.assertIn('does not mention saturday', controller.preflight('Closed Saturday and Sunday.'))
        self.assertIsNone(controller.preflight('Open Monday to Friday 09:00-18:00.'))

    def test_requested_selection_cannot_be_replaced_by_early_answer(self):
        controller = ThinkingController('Open https://example.test and choose Newest in Sort by.',
                                        diagonal_level=True)
        navigate = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]'
        controller.record(navigate, 'BROWSER_TOOL:\n{"text":"Sort by Default Newest"}')
        self.assertIn('has not been selected', controller.preflight('Newest appears in the menu.'))

    def test_authorized_external_effect_cannot_repeat(self):
        controller = ThinkingController('Submit the purchase exactly once and verify it.')
        click = '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Submit purchase\n[/BROWSER_TOOL]'
        self.assertIsNone(controller.preflight(click))
        controller.record(click, 'BROWSER_TOOL:\n{"tabId":"tab-1","text":"Still pending"}')
        self.assertIn('already occurred', controller.preflight(click))
        observe = '[BROWSER_TOOL]\nACTION: observe\nTABID: tab-1\n[/BROWSER_TOOL]'
        self.assertEqual(controller.required_action(), observe)
        self.assertIsNone(controller.preflight(observe))
        controller.record(observe, 'BROWSER_TOOL:\n{"tabId":"tab-1","text":"Purchase submitted"}')
        self.assertIsNone(controller.required_action())
        self.assertTrue(controller.effect_verified)
        self.assertTrue(controller.ready_to_finish_interaction())
        self.assertIn('Purchase submitted', controller.verified_effect_answer())
        self.assertIn('already occurred once', controller.preflight(click))
        self.assertIsNone(controller.preflight('The purchase was submitted.'))

    def test_unconfirmed_effect_never_retries(self):
        controller = ThinkingController('Submit the purchase exactly once.')
        click = '[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Submit purchase\n[/BROWSER_TOOL]'
        controller.record(click, 'BROWSER_TOOL:\n{"tabId":"tab-1","text":"Processing"}')
        observe = '[BROWSER_TOOL]\nACTION: observe\nTABID: tab-1\n[/BROWSER_TOOL]'
        controller.record(observe, 'BROWSER_TOOL:\n{"tabId":"tab-1","text":"Still processing"}')
        self.assertIsNotNone(controller.required_action())
        controller.record(observe, 'BROWSER_TOOL:\n{"tabId":"tab-1","text":"Still processing"}')
        self.assertTrue(controller.effect_unverified)
        self.assertFalse(controller.effect_verified)
        self.assertIsNone(controller.verified_effect_answer())
        self.assertIsNone(controller.required_action())
        self.assertIn('already occurred', controller.preflight(click))

    def test_scroll_reveals_lower_content_and_stops_at_end(self):
        controller = ThinkingController('Open https://example.test, scroll down and report the lower heading.',
                                        diagonal_level=True)
        navigate = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]'
        controller.record(navigate, 'BROWSER_TOOL:\n{"tabId":"tab-1","url":"https://example.test",'
                          '"scroll":{"y":0,"maxY":800,"atTop":true,"atEnd":false},"text":"Top"}')
        self.assertIn('scroll has not occurred', controller.preflight('The lower heading is Hidden.'))
        scroll = '[BROWSER_TOOL]\nACTION: scroll\nPIXELS: 700\nTABID: tab-1\n[/BROWSER_TOOL]'
        self.assertIsNone(controller.preflight(scroll))
        self.assertIn('PIXELS', controller.preflight(scroll.replace('700', '2000')))
        controller.record(scroll, 'BROWSER_TOOL:\n{"tabId":"tab-1","url":"https://example.test",'
                          '"visibleText":"Lower heading: Details","scroll":{"y":800,"maxY":800,'
                          '"atTop":false,"atEnd":true},"moved":true}')
        self.assertTrue(controller.ready_to_finish_scrolled_page())
        self.assertIsNone(controller.preflight('The lower heading is Details.'))
        self.assertIn('scroll boundary', controller.preflight(scroll))

    def test_screenshot_after_scroll_must_be_current(self):
        controller = ThinkingController('Open https://example.test, scroll and capture a screenshot.')
        navigate = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]'
        controller.record(navigate, 'BROWSER_TOOL:\n{"tabId":"tab-1","url":"https://example.test",'
                          '"scroll":{"y":0,"maxY":900,"atEnd":false}}')
        shot = '[BROWSER_TOOL]\nACTION: observe\nSCREENSHOT: true\nTABID: tab-1\n[/BROWSER_TOOL]'
        controller.record(shot, 'BROWSER_TOOL:\n{"tabId":"tab-1","screenshot":"/tmp/top.png"}')
        scroll = '[BROWSER_TOOL]\nACTION: scroll\nPIXELS: 700\n[/BROWSER_TOOL]'
        controller.record(scroll, 'BROWSER_TOOL:\n{"tabId":"tab-1","scroll":{"y":700,"maxY":900,'
                          '"atEnd":false},"moved":true}')
        self.assertIn('fresh screenshot', controller.preflight('Here is the screenshot.'))
        controller.record(shot, 'BROWSER_TOOL:\n{"tabId":"tab-1","screenshot":"/tmp/lower.png"}')
        self.assertIsNone(controller.preflight('Here is the lower screenshot.'))

    def test_scroll_to_bottom_does_not_finish_before_requested_click(self):
        controller = ThinkingController('Scroll to the bottom and click the Details link.')
        controller.record('[BROWSER_TOOL]\nACTION: scroll\nPIXELS: 700\n[/BROWSER_TOOL]',
                          'BROWSER_TOOL:\n{"tabId":"tab-1","visibleText":"Details",'
                          '"scroll":{"y":700,"maxY":700,"atEnd":true}}')
        self.assertFalse(controller.ready_to_finish_scrolled_page())


if __name__ == '__main__':
    unittest.main()
