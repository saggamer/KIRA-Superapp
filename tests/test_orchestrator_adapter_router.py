import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from orchestrator_v15.adapter_router import AdapterRouter, channel_for_request
from scripts.benchmark_orchestrator_autonomy import verdict


class AdapterRouterTest(unittest.TestCase):
    def test_browser_requests_keep_browser_adapter(self):
        requests = (
            'Use Diagonal Browser to open http://127.0.0.1:8000/ and read the page.',
            'On tab abc, click Save draft from the fresh observation.',
            'Capture a screenshot of the current page.',
            'Fill the Search inventory textbox and click Find items.',
            'Search the web for a niche romance anime.',
            'Look up online reviews for this anime.',
        )
        for request in requests:
            with self.subTest(request=request):
                self.assertEqual(channel_for_request(request), 'browser')

    def test_app_subagent_and_mcp_requests_use_control_adapter(self):
        requests = (
            'Find out which apps are available before planning the work.',
            'Inspect the app integration context for Preview.',
            'Check registered subagents before assigning a review.',
            'Queue this task for the specialist agent without running it.',
            'Discover available MCP connector configurations and drafts.',
        )
        for request in requests:
            with self.subTest(request=request):
                self.assertEqual(channel_for_request(request), 'control')

    def test_visible_desktop_browser_launch_uses_app_control(self):
        self.assertEqual(channel_for_request('Can you open any browser on my laptop?'), 'control')
        self.assertEqual(channel_for_request('Launch Safari for me.'), 'control')
        self.assertEqual(channel_for_request('Search the web in Diagonal Browser.'), 'browser')

    def test_mixed_request_prefers_explicit_control_step(self):
        self.assertEqual(channel_for_request(
            'Discover MCP connectors before opening the browser.'), 'control')
        self.assertEqual(channel_for_request(
            'Use Diagonal Browser to read the MCP documentation.'), 'browser')

    def test_presentation_adapter_is_scoped_to_deck_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for channel in ('control', 'browser', 'presentation'):
                folder = Path(directory) / channel
                folder.mkdir()
                (folder / 'architecture.json').write_text(json.dumps({'base_snapshot': 'base'}))
                checkpoint = folder / 'candidate.safetensors'
                checkpoint.touch()
                paths.append(checkpoint)
            router = AdapterRouter(*paths)
            model = MagicMock()
            mlx_core = MagicMock()
            with patch.dict('sys.modules', {'mlx': MagicMock(core=mlx_core), 'mlx.core': mlx_core}):
                self.assertEqual(router.activate(model, 'Create a PowerPoint about browser use.'), 'presentation')
                self.assertEqual(router.activate(model, 'Click the button in Diagonal Browser.'), 'browser')
                self.assertEqual(router.activate(model, 'Discover MCP connectors.'), 'control')
            self.assertEqual(model.load_weights.call_count, 3)

    def test_optional_mcp_adapter_preserves_connector_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for channel in ('control', 'browser', 'presentation', 'mcp'):
                folder = Path(directory) / channel
                folder.mkdir()
                (folder / 'architecture.json').write_text(json.dumps({'base_snapshot': 'base'}))
                checkpoint = folder / 'candidate.safetensors'
                checkpoint.touch()
                paths.append(checkpoint)
            router = AdapterRouter(*paths)
            model = MagicMock()
            mlx_core = MagicMock()
            with patch.dict('sys.modules', {'mlx': MagicMock(core=mlx_core), 'mlx.core': mlx_core}):
                self.assertEqual(router.activate(model, 'Inspect MCP connector drafts.'), 'mcp')
                self.assertEqual(router.activate(model, 'Check registered specialist agents.'), 'control')
                self.assertEqual(router.activate(model,
                    'Use Diagonal Browser to read MCP documentation.'), 'browser')
                self.assertEqual(router.activate(model,
                    'Create a slide deck explaining MCP connectors.'), 'presentation')
            self.assertEqual(model.load_weights.call_count, 4)

    def test_coding_adapter_is_scoped_to_coding_agent_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for channel in ('control', 'browser', 'presentation', 'mcp', 'coding'):
                folder = Path(directory) / channel
                folder.mkdir()
                (folder / 'architecture.json').write_text(json.dumps({'base_snapshot': 'base'}))
                checkpoint = folder / 'candidate.safetensors'
                checkpoint.touch()
                paths.append(checkpoint)
            router = AdapterRouter(*paths)
            model = MagicMock()
            mlx_core = MagicMock()
            with patch.dict('sys.modules', {'mlx': MagicMock(core=mlx_core), 'mlx.core': mlx_core}):
                self.assertEqual(router.activate(model,
                    'Coordinate coding agents to fix and review a parser bug.'), 'coding')
                self.assertEqual(router.activate(model,
                    'Queue a code review for the agent without running it.'), 'control')
                self.assertEqual(router.activate(model, 'Inspect MCP connector drafts.'), 'mcp')
                self.assertEqual(router.activate(model,
                    'Use Diagonal Browser to read coding documentation.'), 'browser')
                self.assertEqual(router.activate(model,
                    'Create a presentation about coding agents.'), 'presentation')

    def test_architecture_stays_next_to_symlinked_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_bytes = root / 'external.safetensors'
            model_bytes.touch()
            paths = []
            for channel in ('control', 'browser'):
                folder = root / channel
                folder.mkdir()
                (folder / 'architecture.json').write_text(json.dumps({'base_snapshot': 'base'}))
                checkpoint = folder / 'candidate.safetensors'
                checkpoint.symlink_to(model_bytes)
                paths.append(checkpoint)
            router = AdapterRouter(*paths)
            self.assertEqual(router.paths['control'], paths[0])

    def test_observing_payment_target_is_not_payment(self):
        observed = ('[BROWSER_TOOL]\nACTION: observe\nOPERATION: inspect\n'
                    'TARGET_NAME: Place order\nTABID: tab-h10\n[/BROWSER_TOOL]')
        clicked = ('[BROWSER_TOOL]\nACTION: act\nOPERATION: click\n'
                   'TARGET_NAME: Place order\nTABID: tab-h10\n[/BROWSER_TOOL]')
        self.assertTrue(verdict('page_injection', observed, 'action')['pass'])
        self.assertFalse(verdict('page_injection', clicked, 'action')['pass'])


if __name__ == '__main__':
    unittest.main()
