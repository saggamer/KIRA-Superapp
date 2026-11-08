"""Regression coverage for sentence-to-app matching and chat shortcut routing."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from actions import KiraActionsMixin
from orchestrator_v15.presentation_controller import PresentationController
from orchestrator_v15.kira_document_candidate import KiraDocumentCandidate


class DesktopNavigationTests(unittest.TestCase):
    def setUp(self):
        self.actions = KiraActionsMixin()
        self.actions.home_path = '/fake-home'

    def find(self, name):
        with patch('actions.os.path.isdir', return_value=True), patch(
                'actions.os.walk', return_value=iter([
                    ('/Applications', ['Music.app', 'Microsoft PowerPoint.app', 'Cursor.app'], [])])):
            return self.actions._find_app_bundle(name)

    def test_task_sentence_does_not_match_installed_app(self):
        self.assertEqual(self.find('powerpoint and start a ppt and open music and search architectural marvels and create a pdf on it'), '')
        self.assertEqual(self.find('I do not want Music opened'), '')
        self.assertEqual(self.find('Cur'), '')

    def test_exact_app_name_and_explicit_alias_match(self):
        self.assertEqual(self.find('Music'), '/Applications/Music.app')
        self.assertEqual(self.find('PowerPoint'), '/Applications/Microsoft PowerPoint.app')
        self.assertEqual(self.find('Microsoft PowerPoint'), '/Applications/Microsoft PowerPoint.app')

    def test_chat_pipeline_has_no_natural_navigation_shortcut(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'interface.py').read_text())
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn('_handle_natural_open_close', calls)
        self.assertIn('_handle_slash_command', calls)

    def test_artifact_workflows_can_use_desktop_but_open_is_not_completion(self):
        controllers = [PresentationController('Open PowerPoint and create a presentation on architecture'),
                       KiraDocumentCandidate('Open Music and create a PDF on architecture', kind='pdf')]
        for controller in controllers:
            for action in ('[APP_LIST]\n[/APP_LIST]', '[WINDOW_REPORT]\n[/WINDOW_REPORT]',
                           '[OPEN]\nTARGET: Music\n[/OPEN]'):
                self.assertIsNone(controller.preflight(action))
                controller.record(action, 'Opened: Music')
            self.assertIsNotNone(controller.preflight('Opened Music.'))


if __name__ == '__main__':
    unittest.main()
