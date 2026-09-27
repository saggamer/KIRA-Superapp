import threading
import unittest
from kira_live.audio_pipeline import overlap_audio


class AudioPipelineTests(unittest.TestCase):
    def test_synthesis_continues_while_playback_is_waiting(self):
        playback_started = threading.Event()
        third_generated = threading.Event()
        consumed = []
        def chunks():
            yield 1, 24000
            yield 2, 24000
            self.assertTrue(playback_started.wait(1))
            third_generated.set()
            yield 3, 24000
        def consume(value, rate):
            playback_started.set()
            self.assertTrue(third_generated.wait(1))
            consumed.append(value)
        self.assertEqual(overlap_audio(chunks(), consume, threading.Event()), 3)
        self.assertEqual(consumed, [1, 2, 3])

    def test_short_utterance_flushes_and_cancelled_audio_does_not_play(self):
        output = []
        event = threading.Event()
        self.assertEqual(overlap_audio(iter([(1, 24000)]), lambda *args: output.append(args), event), 1)
        event.set()
        self.assertEqual(overlap_audio(iter([(2, 24000)]), lambda *args: output.append(args), event), 0)
        self.assertEqual(len(output), 1)
