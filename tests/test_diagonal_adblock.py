import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from actions import KiraActionsMixin
from diagonal_browser.kira_client import BrowserMCPClient
from orchestrator_v15.adapter_router import channel_for_request
from orchestrator_v15.spark_tools import native_browser_call_to_block
from orchestrator_v15.thinking_controller import ThinkingController


class DiagonalAdblockTests(unittest.TestCase):
    def test_model_bridge_maps_false_site_exceptions_and_filter_rules(self):
        blocks = '''[BROWSER_TOOL]
ACTION: adblock_set
ENABLED: true
SITE: example.com
SITE_ENABLED: false
COSMETIC: false
AUTO_UPDATE: false
[/BROWSER_TOOL]
[BROWSER_TOOL]
ACTION: adblock_rule
OPERATION: add
RULE: example.com##.advertisement
[/BROWSER_TOOL]
[BROWSER_TOOL]
ACTION: adblock_check
URL: https://ads.example.com/script.js
PAGE_URL: https://example.com/
RESOURCE_TYPE: script
[/BROWSER_TOOL]'''
        with patch('diagonal_browser.kira_client.call_browser_tool', return_value={'enabled': True}) as call:
            KiraActionsMixin()._run_web_search_tools(blocks)
        self.assertEqual(call.call_args_list[0].args, ('adblock_set', {'enabled': True, 'site': 'example.com', 'siteEnabled': False, 'cosmetic': False, 'autoUpdate': False}))
        self.assertEqual(call.call_args_list[1].args, ('adblock_rule', {'operation': 'add', 'rule': 'example.com##.advertisement'}))
        self.assertEqual(call.call_args_list[2].args, ('adblock_check', {'url': 'https://ads.example.com/script.js', 'pageUrl': 'https://example.com/', 'resourceType': 'script'}))
        with patch('diagonal_browser.kira_client.call_browser_tool') as call:
            result = KiraActionsMixin()._run_web_search_tools('[BROWSER_TOOL]\nACTION: adblock_set\nENABLED: maybe\n[/BROWSER_TOOL]')
        call.assert_not_called(); self.assertIn('true or false', result)

    def test_controller_and_native_model_accept_adblock_controls_without_navigation(self):
        self.assertEqual(channel_for_request('Enable the ad blocker'), 'browser')
        controller = ThinkingController('Pause ad blocking on https://example.com')
        block = '[BROWSER_TOOL]\nACTION: adblock_set\nSITE: example.com\nSITE_ENABLED: false\n[/BROWSER_TOOL]'
        self.assertIsNone(controller.preflight(block))
        native = '<tool_call>BROWSER_TOOL<arg_key>action</arg_key><arg_value>adblock_rule</arg_value><arg_key>operation</arg_key><arg_value>add</arg_value><arg_key>rule</arg_key><arg_value>||ads.example.com^</arg_value></tool_call>'
        parsed = native_browser_call_to_block(native)
        self.assertIsNotNone(parsed); self.assertIsNone(controller.preflight(parsed))
        self.assertIn('PAGE_URL', controller.preflight('[BROWSER_TOOL]\nACTION: adblock_check\nURL: https://example.com/ad.js\n[/BROWSER_TOOL]'))

    @unittest.skipUnless(shutil.which('node') and (Path(__file__).parents[1] / 'diagonal_browser/node_modules').is_dir(), 'Local Node MCP dependencies required')
    def test_real_mcp_controls_work_without_chrome_and_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'DIAGONAL_PROFILE_DIR': directory, 'DIAGONAL_CHROME_BIN': '/nonexistent-chrome', 'DIAGONAL_ADBLOCK_AUTO_UPDATE': '0'}):
            client = BrowserMCPClient()
            try:
                status = client.call('adblock_status')
                self.assertTrue(status['ready']); self.assertGreater(status['networkRules'], 100000)
                request = {'url': 'https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js', 'pageUrl': 'https://example.com/', 'resourceType': 'script'}
                self.assertTrue(client.call('adblock_check', request)['blocked'])
                client.call('adblock_set', {'site': 'example.com', 'siteEnabled': False})
                self.assertFalse(client.call('adblock_check', request)['blocked'])
                client.call('adblock_rule', {'operation': 'add', 'rule': '||diagonal-fixture.test^$script'})
                client.close()
                status = client.call('adblock_status', {'url': 'https://example.com/'})
                self.assertFalse(status['siteEnabled']); self.assertIn('||diagonal-fixture.test^$script', status['customRules'])
                client.call('adblock_set', {'site': 'example.com', 'siteEnabled': True})
                self.assertTrue(client.call('adblock_check', request)['blocked'])
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
