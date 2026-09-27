import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voice import KiraVoiceMixin


class DictationTests(unittest.TestCase):
    def setUp(self):
        self.brain = KiraVoiceMixin()
        self.brain._dictation_lock = threading.Lock()
        self.brain._dictation_cancel = threading.Event()
        self.brain.send_prompt = Mock(side_effect=AssertionError('Dictation must not send'))
        self.brain._trigger_voice = Mock(side_effect=AssertionError('Dictation must not speak'))

    def run_capture(self, ready=True, start_error=False, cancel=False):
        session = Mock()
        def factory(**callbacks):
            def start():
                if cancel:
                    self.brain.cancel_dictation()
                    callbacks['on_transcript']('discard this', {})
                elif not start_error:
                    callbacks['on_transcript']('  Build a presentation  ', {})
                return {'running': not start_error, 'last_error': 'Microphone denied'}
            session.start.side_effect = start
            return session
        engine = SimpleNamespace(
            LocalSTT=lambda: SimpleNamespace(ready=ready, error='Missing STT model'),
            LocalVoiceSession=factory,
        )
        with patch('voice.ensure_listen_engine', return_value=True), patch('voice._listen_module', engine):
            result = self.brain.dictate_once()
        self.assertFalse(self.brain._dictation_lock.locked())
        self.brain.send_prompt.assert_not_called()
        self.brain._trigger_voice.assert_not_called()
        if ready:
            session.stop.assert_called_once()
        return result

    def test_transcript_only_without_agent_or_tts(self):
        result = self.run_capture()
        self.assertTrue(result['ok'])
        self.assertEqual(result['text'], 'Build a presentation')
        self.assertFalse(self.brain.voice_mode_active)

    def test_missing_model(self):
        self.assertEqual(self.run_capture(ready=False)['error'], 'Missing STT model')

    def test_microphone_failure_cleans_up(self):
        self.assertEqual(self.run_capture(start_error=True)['error'], 'Microphone denied')

    def test_cancel_discards_late_transcript(self):
        self.assertEqual(self.run_capture(cancel=True)['text'], '')

    def test_duplicate_capture_rejected(self):
        self.brain._dictation_lock.acquire()
        try:
            self.assertFalse(self.brain.dictate_once()['ok'])
        finally:
            self.brain._dictation_lock.release()

    def test_live_entrypoint_fails_closed_when_weights_are_unavailable(self):
        with patch.object(self.brain, 'get_kira_live_status', return_value={'conversation_ready': False}):
            self.assertFalse(self.brain.start_live_voice()['ok'])

    def test_live_releases_idle_text_worker_before_loading_voice_weights(self):
        self.brain.model_worker_process = object()
        self.brain.model_worker_io_lock = threading.Lock()
        self.brain._stop_model_worker = Mock()
        self.brain.active_model = object()
        self.brain.active_tokenizer = object()
        self.brain.current_brain = 'kira'
        self.brain._release_text_worker_for_live()
        self.brain._stop_model_worker.assert_called_once()
        self.assertIsNone(self.brain.active_model)
        self.assertIsNone(self.brain.current_brain)

    def test_live_does_not_interrupt_a_busy_text_answer(self):
        self.brain.model_worker_process = object()
        self.brain.model_worker_io_lock = threading.Lock()
        self.brain.model_worker_io_lock.acquire()
        self.brain._stop_model_worker = Mock()
        try:
            with self.assertRaisesRegex(RuntimeError, 'text answer is still running'):
                self.brain._release_text_worker_for_live()
            self.brain._stop_model_worker.assert_not_called()
        finally:
            self.brain.model_worker_io_lock.release()


if __name__ == '__main__':
    unittest.main()
