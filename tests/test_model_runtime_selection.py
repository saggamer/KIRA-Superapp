import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault('webview', types.SimpleNamespace())
from interface import KiraBrain, LazyKiraAPI


class ModelRuntimeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.brain = KiraBrain.__new__(KiraBrain)
        self.brain.app_root = '/project'
        self.installed = os.path.expanduser('~/.local/share/kira-superapp/runtime/bin/python')

    def test_managed_runtime_precedes_documents_virtualenv(self):
        with patch.dict(os.environ, {}, clear=True), patch('interface.sys.platform', 'darwin'), \
                patch('interface.os.path.isfile', return_value=True), patch('interface.os.access', return_value=True):
            self.assertEqual(self.brain._model_worker_python('orchestrator_v15'), self.installed)

    def test_explicit_override_and_invalid_override(self):
        with patch.dict(os.environ, {'KIRA_V15_PYTHON': '/custom/python'}, clear=True), \
                patch('interface.sys.platform', 'darwin'), patch('interface.os.access', return_value=True):
            with patch('interface.os.path.isfile', return_value=True):
                self.assertEqual(self.brain._model_worker_python('orchestrator_v15'), '/custom/python')
            with patch('interface.os.path.isfile', return_value=False):
                with self.assertRaises(FileNotFoundError):
                    self.brain._model_worker_python('orchestrator_v15')

    def test_project_runtime_remains_fallback(self):
        project = '/project/.venv-orchestrator-v15/bin/python'
        with patch.dict(os.environ, {}, clear=True), patch('interface.sys.platform', 'darwin'), \
                patch('interface.os.path.isfile', side_effect=lambda path: path == project), \
                patch('interface.os.access', return_value=True):
            self.assertEqual(self.brain._model_worker_python('orchestrator_v15'), project)

    def test_other_models_keep_current_interpreter(self):
        self.assertEqual(self.brain._model_worker_python('kira'), sys.executable)

    def test_live_startup_is_pending_not_a_component_failure(self):
        api = LazyKiraAPI()
        api._ensure_brain = Mock(return_value=None)
        status = api.get_kira_live_status()
        self.assertEqual(status['state'], 'starting')
        self.assertEqual(status['blockers'], [])
        self.assertFalse(status['conversation_ready'])
        api._ensure_brain.assert_called_once_with(wait=False)
        api._brain = types.SimpleNamespace(get_kira_live_status=lambda: {'conversation_ready': True})
        self.assertTrue(api.get_kira_live_status()['conversation_ready'])

    def test_live_backend_failure_reports_actual_error(self):
        api = LazyKiraAPI()
        api._boot_error = 'Backend failed: missing configuration'
        status = api.get_kira_live_status()
        self.assertEqual(status['state'], 'error')
        self.assertEqual(status['blockers'], [api._boot_error])


if __name__ == '__main__':
    unittest.main()
