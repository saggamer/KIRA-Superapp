import json
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from interface import KiraBrain


class SchedulingIntegrationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.brain = b = KiraBrain.__new__(KiraBrain)
        b.scheduled_tasks_path = str(self.root / 'tasks.json')
        b.scheduled_tasks_lock = threading.RLock()
        b.scheduled_tasks = []
        b.response_queue = queue.Queue()
        b.task_queue = queue.Queue()
        b.safe_shell_commands = set()
        b._log_agentic_event = Mock()
        b._record_execution_evidence = Mock()
        b._append_chat_message = Mock()
        b.new_chat = Mock(return_value={'id': 'isolated-chat'})
        b.mcp_workspace = str(self.root)
        b.plugins_workspace = str(self.root)
        b.plugin_state_path = str(self.root / 'state.json')
        b.mcp_config_targets = {'test': str(self.root / 'target.json')}
        b._resolve_path = lambda value: value

    def schedule(self, **kwargs):
        result = self.brain.create_scheduled_item('reminder', 'Take a break', time.time()+60, **kwargs)
        self.assertTrue(result['ok'], result)
        return self.brain.scheduled_tasks[-1]

    def test_management_persistence_and_snapshot_isolation(self):
        task = self.schedule()
        b = self.brain
        for action, status in [('pause', 'paused'), ('resume', 'pending'), ('cancel', 'cancelled')]:
            self.assertTrue(b.manage_scheduled_task(task['id'], action)['ok'])
            self.assertEqual(json.loads(Path(b.scheduled_tasks_path).read_text())[0]['status'], status)
        b.get_scheduled_tasks()[0]['status'] = 'pending'
        self.assertEqual(task['status'], 'cancelled')
        self.assertFalse(b.manage_scheduled_task(task['id'], 'resume')['ok'])

    def test_reschedule_and_invalid_inputs(self):
        task = self.schedule()
        self.assertFalse(self.brain.manage_scheduled_task(task['id'], 'reschedule', float('nan'))['ok'])
        future = time.time()+120
        self.assertTrue(self.brain.manage_scheduled_task(task['id'], 'reschedule', future)['ok'])
        self.assertEqual(task['run_at'], future)
        for value in [time.time()-1, float('inf')]:
            self.assertFalse(self.brain.create_scheduled_item('reminder', 'test', value)['ok'])
        with self.assertRaises(ValueError):
            self.brain._parse_schedule_delay_seconds('at 25:99')
        with self.assertRaises(ValueError):
            self.brain._parse_schedule_delay_seconds('next something')

    def test_restart_pauses_interrupted_task(self):
        task = self.schedule()
        task['status'] = 'running'
        self.brain._save_scheduled_tasks()
        self.assertEqual(self.brain._load_scheduled_tasks()[0]['status'], 'paused')

    def test_reminder_fires_without_model_and_failure_does_not_block_next(self):
        first = self.schedule()
        second = self.schedule()
        first['run_at'] = second['run_at'] = time.time()-1
        self.brain._append_chat_message.side_effect = [RuntimeError('test storage failure'), None]
        self.brain.is_running = True
        worker = threading.Thread(target=self.brain._scheduled_task_loop)
        worker.start()
        try:
            event = self.brain.response_queue.get(timeout=3)
            self.assertEqual(event['type'], 'reminder')
            self.assertEqual(event['task_id'], second['id'])
        finally:
            self.brain.is_running = False
            worker.join(timeout=4)
        self.assertFalse(worker.is_alive())
        self.assertEqual(first['status'], 'failed')
        self.assertEqual(second['status'], 'completed')
        self.assertTrue(self.brain.task_queue.empty())

    def test_repeating_reminder_skips_missed_intervals(self):
        task = self.schedule(repeat='daily')
        task['run_at'] = time.time()-3*86400
        self.brain._deliver_scheduled_reminder(dict(task))
        self.assertEqual(task['status'], 'pending')
        self.assertGreater(task['run_at'], time.time())
        self.assertEqual(self.brain.response_queue.qsize(), 1)

    def test_app_schedule_uses_exact_installed_bundle(self):
        self.brain._kira_menu_app_items = Mock(return_value=[{'title': 'Cursor', 'path': '/Applications/Cursor.app'}])
        result = self.brain.create_scheduled_item('app', 'cursor', time.time()+60)
        self.assertTrue(result['ok'])
        self.assertEqual(self.brain.scheduled_tasks[0]['prompt'], '[OPEN]\n/Applications/Cursor.app\n[/OPEN]')
        self.assertFalse(self.brain.create_scheduled_item('app', 'cur', time.time()+60)['ok'])

    def test_failed_save_rolls_back_schedule(self):
        with patch.object(self.brain, '_atomic_write_json', side_effect=OSError('disk full')):
            self.assertFalse(self.brain.create_scheduled_item('reminder', 'test', time.time()+60)['ok'])
        self.assertEqual(self.brain.scheduled_tasks, [])

    def test_tree_schedule_block_creates_timed_reminder(self):
        result = self.brain._run_scheduler_tools(
            '[SCHEDULE_TASK]\nKIND: reminder\nTASK: Review notes\nDELAY: 10 minutes\nREPEAT: daily\n[/SCHEDULE_TASK]',
            'Remind me to review notes', 'isolated-chat',
        )
        self.assertIn('SCHEDULED TASK:', result)
        task = self.brain.scheduled_tasks[0]
        self.assertEqual(task['mode'], 'reminder')
        self.assertEqual(task['delay_seconds'], 600)
        self.assertEqual(task['interval_seconds'], 86400)

    def test_due_app_is_enqueued_through_existing_action_pathway(self):
        b = self.brain
        b._kira_menu_app_items = Mock(return_value=[{'title': 'Cursor', 'path': '/Applications/Cursor.app'}])
        b.create_scheduled_item('app', 'Cursor', time.time()+60)
        b.scheduled_tasks[0]['run_at'] = time.time()-1
        b.is_running = True
        worker = threading.Thread(target=b._scheduled_task_loop)
        worker.start()
        try:
            task = b.task_queue.get(timeout=3)
            self.assertEqual(task['prompt'], '[OPEN]\n/Applications/Cursor.app\n[/OPEN]')
            self.assertEqual(task['scheduled_task_id'], b.scheduled_tasks[0]['id'])
        finally:
            b.is_running = False
            worker.join(timeout=4)
        self.assertFalse(worker.is_alive())

    def test_mcp_static_checks_and_secret_rejection(self):
        b = self.brain
        result = b.create_mcp_draft('test-server', sys.executable)
        self.assertTrue(b.test_mcp_draft(result['path'])['ok'])
        nonexec = self.root / 'not-executable'
        nonexec.write_text('plain text')
        self.assertFalse(b._mcp_server_summary('test', {'command': str(nonexec)})['executable_found'])
        self.assertFalse(b._mcp_server_summary('test', {'command': sys.executable, 'args': 'wrong'})['valid'])
        self.assertFalse(b.create_mcp_draft('secret', sys.executable, env_json='{"TOKEN":"plaintext"}')['ok'])

    def test_mcp_merge_preserves_settings_and_rejects_corrupt_target(self):
        b = self.brain
        draft = b.create_mcp_draft('new-server', sys.executable)['path']
        target = Path(b.mcp_config_targets['test'])
        target.write_text('{broken')
        self.assertIn('failed', b._apply_mcp_config({'draft': draft, 'target': 'test'}))
        self.assertEqual(target.read_text(), '{broken')
        target.write_text(json.dumps({'other': True, 'mcpServers': {'old': {'command': 'old'}}}))
        self.assertIn('applied', b._apply_mcp_config({'draft': draft, 'target': 'test'}))
        result = json.loads(target.read_text())
        self.assertTrue(result['other'])
        self.assertEqual(set(result['mcpServers']), {'old', 'new-server'})

    def test_plugin_invalid_ids_and_escaped_entrypoints_are_disabled(self):
        entry = self.root / 'plugin.py'
        entry.write_text('')
        for index, (plugin_id, entrypoint) in enumerate([('..', 'plugin.py'), ('escape', sys.executable), ('safe', 'plugin.py')]):
            (self.root / f'{index}.plugin.json').write_text(json.dumps({'id': plugin_id, 'entrypoint': entrypoint}))
        records = self.brain._external_plugin_records()
        self.assertEqual([r['healthy'] for r in records], [False, False, True])
        self.assertFalse(any(r['enabled'] for r in records))


if __name__ == '__main__':
    unittest.main()
