import threading
import unittest
from types import SimpleNamespace
import numpy as np
from pathlib import Path
from kira_live.mlx_voice import IntegratedSpeechPlan, KiraIntegratedQwenVoice
from kira_live.policy import live_text_system_instruction


class SentenceVoiceTests(unittest.TestCase):
    def test_complete_reply_is_one_contextual_streaming_generation(self):
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(audio=np.ones(2400), sample_rate=24000)
        voice = KiraIntegratedQwenVoice.__new__(KiraIntegratedQwenVoice)
        voice.model = SimpleNamespace(generate_custom_voice=generate)
        plan = IntegratedSpeechPlan((), 'Hello there. How are you?', 'Aiden', 'English', 'Warm')
        chunks = list(voice.iter_audio(plan, threading.Event()))
        self.assertEqual(len(chunks), 1)
        self.assertTrue(calls[0]['stream'])
        self.assertEqual(calls[0]['text'], plan.public_text)

    def test_cancelled_audio_never_reaches_playback(self):
        voice = KiraIntegratedQwenVoice.__new__(KiraIntegratedQwenVoice)
        voice.model = SimpleNamespace(generate_custom_voice=lambda **kwargs: iter([SimpleNamespace(audio=np.ones(20), sample_rate=24000)]))
        event = threading.Event(); event.set()
        plan = IntegratedSpeechPlan((), 'Hello.', 'Aiden', 'English', '')
        self.assertEqual(list(voice.iter_audio(plan, event)), [])

    def test_live_session_isolation_and_spoken_policy(self):
        ui = (Path(__file__).parents[1] / 'ui.html').read_text()
        self.assertIn("new_chat('Live conversation')", ui)
        self.assertIn("voiceLiveTranscript.textContent = '';", ui)
        self.assertIn('eventChatId !== currentChatId', ui)
        self.assertIn('complete, naturally spoken sentences', live_text_system_instruction())
