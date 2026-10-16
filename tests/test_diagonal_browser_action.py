import json
import unittest
from unittest.mock import Mock, patch

from actions import KiraActionsMixin


class DiagonalBrowserActionTests(unittest.TestCase):
    def test_browser_action_audit_excludes_filled_values(self):
        agent = KiraActionsMixin()
        agent._log_agentic_event = Mock()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: fill\nTARGET_ROLE: textbox\nTARGET_NAME: Search\nVALUE: private query\nTABID: tab-1\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"observationId": "obs-2"}):
            agent._run_web_search_tools(block, chat_id="chat-1")
        event, payload = agent._log_agentic_event.call_args.args
        self.assertEqual(event, "browser_tool_decision")
        self.assertEqual(payload["result_observation_id"], "obs-2")
        self.assertTrue(payload["success"])
        self.assertNotIn("private query", json.dumps(payload))

    def test_unapproved_purchase_click_is_blocked_before_browser_dispatch(self):
        agent = KiraActionsMixin()
        agent._log_agentic_event = Mock()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_ROLE: button\nTARGET_NAME: Submit purchase\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool") as call:
            output = agent._run_web_search_tools(block, "Inspect the order. Do not submit purchase.", "chat-1")
        call.assert_not_called()
        self.assertIn("do not ask again", output)
        self.assertEqual(agent._log_agentic_event.call_args.args[1]["reason"], "unapproved_purchase_action")

    def test_price_question_does_not_authorize_purchase(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Place order\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool") as call:
            output = agent._run_web_search_tools(block, "What is the purchase price of this item?", "chat-1")
        call.assert_not_called()
        self.assertIn("Explicit user authorization is required", output)

    def test_requested_purchase_proceeds_without_reasking(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_ROLE: button\nTARGET_NAME: Place order\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"observationId": "obs-2"}) as call:
            output = agent._run_web_search_tools(block, "Place this order now.", "chat-1")
        call.assert_called_once()
        self.assertNotIn("Explicit user authorization is required", output)

    def test_direct_order_request_also_counts_as_authorization(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_ROLE: button\nTARGET_NAME: Place order\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"observationId": "obs-2"}) as call:
            agent._run_web_search_tools(block, "Order these headphones now.", "chat-1")
        call.assert_called_once()

    def test_explicit_ask_before_final_click_is_honored(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_NAME: Place order\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool") as call:
            output = agent._run_web_search_tools(block, "Buy this item, but ask me before placing the order.", "chat-1")
        call.assert_not_called()
        self.assertIn("Ask once at this final step", output)

    def test_checkout_preparation_proceeds_without_purchase_approval(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_ROLE: button\nTARGET_NAME: Checkout\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"observationId": "obs-2"}) as call:
            agent._run_web_search_tools(block, "Review my cart and shipping options.", "chat-1")
        call.assert_called_once()

    def test_kira_browser_block_dispatches_snapshot_index(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: click\nINDEX: 2\nTABID: tab-1\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"url": "https://example.com"}) as call:
            output = agent._run_web_search_tools(block)
        call.assert_called_once_with("click", {"index": 2, "tabId": "tab-1"})
        self.assertIn("https://example.com", output)

    def test_kira_browser_block_preserves_empty_fill_value(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: fill\nSELECTOR: #query\nVALUE:\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"valueLength": 0}) as call:
            agent._run_web_search_tools(block)
        call.assert_called_once_with("fill", {"selector": "#query", "value": ""})

    def test_kira_can_request_structured_page_inspection(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: inspect_page\nTABID: tab-1\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"page": {"title": "Example"}}) as call:
            output = agent._run_web_search_tools(block)
        call.assert_called_once_with("inspect_page", {"tabId": "tab-1"})
        self.assertIn("Example", output)

    def test_compact_browser_control_dispatches_and_returns_updated_view(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nTARGET_ROLE: button\nTARGET_NAME: Continue\nTABID: tab-1\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"url": "https://example.com/next", "elements": []}) as call:
            output = agent._run_web_search_tools(block)
        call.assert_called_once_with("act", {"operation": "click", "tabId": "tab-1", "role": "button", "targetName": "Continue"})
        self.assertIn("https://example.com/next", output)

    def test_observe_can_request_viewport_image(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: observe\nSCREENSHOT: true\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"screenshot": "/tmp/view.png"}) as call:
            agent._run_web_search_tools(block)
        call.assert_called_once_with("observe", {"screenshot": True})

    def test_scroll_dispatches_pixels_and_rejects_out_of_bounds(self):
        agent = KiraActionsMixin()
        block = '[BROWSER_TOOL]\nACTION: scroll\nPIXELS: 700\nTABID: tab-1\n[/BROWSER_TOOL]'
        with patch('diagonal_browser.kira_client.call_browser_tool', return_value={
                'scroll': {'y': 700, 'atEnd': False}, 'visibleText': 'Lower section'}) as call:
            output = agent._run_web_search_tools(block)
        call.assert_called_once_with('scroll', {'pixels': 700, 'tabId': 'tab-1'})
        self.assertIn('Lower section', output)
        with patch('diagonal_browser.kira_client.call_browser_tool') as call:
            bad = agent._run_web_search_tools(block.replace('700', '5000'))
        call.assert_not_called()
        self.assertIn('PIXELS must be', bad)

    def test_kira_can_choose_an_email_nickname(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: use_email_profile\nNICKNAME: Personal\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"submitted": False}) as call:
            output = agent._run_web_search_tools(block)
        call.assert_called_once_with("use_email_profile", {"nickname": "Personal"})
        self.assertFalse(json.loads(output.split("BROWSER_TOOL:\n", 1)[1])["submitted"])

    def test_kira_can_click_a_visual_map_coordinate(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: click\nX: 314\nY: 159\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"url": "https://example.com"}) as call:
            agent._run_web_search_tools(block)
        call.assert_called_once_with("act", {"operation": "click", "x": 314, "y": 159})

    def test_kira_passes_observation_id_and_form_operation(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: act\nOPERATION: select_option\nINDEX: 4\nVALUE: b\nOBSERVATION_ID: obs-1\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"observationId": "obs-2"}) as call:
            output = agent._run_web_search_tools(block)
        call.assert_called_once_with("act", {"operation": "select_option", "index": 4, "value": "b", "observationId": "obs-1"})
        self.assertIn('"observationId":"obs-2"', output)

    def test_http_error_is_not_recorded_as_success(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.com/missing\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"status": 404, "url": "https://example.com/missing"}):
            output = agent._run_web_search_tools(block)
        self.assertIn("BROWSER_TOOL ERROR:", output)

    def test_explicit_browser_url_has_safe_navigation_fallback(self):
        agent = KiraActionsMixin()
        agent.current_brain = "kira"
        blocks = agent._build_deterministic_agentic_blocks("Use Diagonal Browser to open https://example.com/docs.")
        self.assertIn("ACTION: navigate", blocks)
        self.assertIn("URL: https://example.com/docs", blocks)

    def test_browser_search_fallback_keeps_regular_web_search_routing(self):
        agent = KiraActionsMixin()
        agent.current_brain = "kira"
        browser_blocks = agent._build_deterministic_agentic_blocks("/browser search for accessible tabs")
        regular_blocks = agent._build_deterministic_agentic_blocks("search the web for accessible tabs")
        self.assertIn("[BROWSER_TOOL]\nACTION: search", browser_blocks)
        self.assertIn("[WEB_SEARCH]", regular_blocks)

    def test_kira_can_open_and_cancel_visible_login(self):
        agent = KiraActionsMixin()
        block = "[BROWSER_TOOL]\nACTION: begin_login\nURL: https://example.com/login\n[/BROWSER_TOOL]\n[BROWSER_TOOL]\nACTION: finish_login\nSAVE: false\n[/BROWSER_TOOL]"
        with patch("diagonal_browser.kira_client.call_browser_tool", return_value={"saved": False}) as call:
            agent._run_web_search_tools(block)
        self.assertEqual(call.call_args_list[0].args, ("begin_login", {"url": "https://example.com/login"}))
        self.assertEqual(call.call_args_list[1].args, ("finish_login", {"save": False}))


if __name__ == "__main__":
    unittest.main()
