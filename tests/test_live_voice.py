import os
import sys
import threading
import time
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import listen


class FakeSTT:
    ready = True
    error = ""

    def status(self):
        return {"ready": True, "provider": "fake_local", "model": "test"}

    def transcribe(self, samples, sample_rate):
        self.last_sample_count = len(samples)
        return "open my project", {"provider": "fake_local", "processing_seconds": 0.001}


class DeterministicVad:
    def process(self, frame):
        level = float(np.max(np.abs(frame)))
        return level > 0.02, min(1.0, level * 10.0)

    def status(self):
        return {"ready": True, "provider": "deterministic_test"}

    def close(self):
        return None


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.started = False


class LiveVoiceTests(unittest.TestCase):
    def test_endpoint_routes_one_local_transcript(self):
        transcripts = []
        events = []
        finished = threading.Event()

        def on_transcript(text, metadata):
            transcripts.append((text, metadata))
            finished.set()

        session = listen.LocalVoiceSession(
            stt=FakeSTT(),
            vad=DeterministicVad(),
            on_transcript=on_transcript,
            on_event=lambda kind, payload: events.append((kind, payload)),
            stream_factory=lambda **kwargs: FakeStream(**kwargs),
            endpoint_seconds=0.08,
            max_turn_seconds=2.0,
            pre_roll_seconds=0.05,
        )
        status = session.start()
        self.assertTrue(status["running"])
        session.feed_audio(np.zeros(int(listen.SAMPLE_RATE * 0.08), dtype=np.float32))
        session.feed_audio(np.full(int(listen.SAMPLE_RATE * 0.24), 0.08, dtype=np.float32))
        session.feed_audio(np.zeros(int(listen.SAMPLE_RATE * 0.20), dtype=np.float32))
        self.assertTrue(finished.wait(2.0))
        session.stop()

        self.assertEqual([item[0] for item in transcripts], ["open my project"])
        event_names = [item[0] for item in events]
        self.assertIn("SPEECH_START", event_names)
        self.assertIn("SPEECH_END", event_names)
        self.assertIn("STT_PROCESSING", event_names)

    def test_missing_recognizer_is_reported_without_cloud(self):
        original = os.environ.get("KIRA_STT_PROVIDER")
        os.environ["KIRA_STT_PROVIDER"] = "qwen3_asr"
        try:
            stt = listen.LocalSTT()
            status = stt.status()
            self.assertFalse(status["ready"])
            self.assertFalse(status["cloud_enabled"])
            self.assertIn("setup_live_voice.py", status["error"])
        finally:
            if original is None:
                os.environ.pop("KIRA_STT_PROVIDER", None)
            else:
                os.environ["KIRA_STT_PROVIDER"] = original

    def test_audio_levels_are_emitted_for_sphere(self):
        levels = []
        session = listen.LocalVoiceSession(
            stt=FakeSTT(),
            vad=DeterministicVad(),
            on_audio_level=levels.append,
            stream_factory=lambda **kwargs: FakeStream(**kwargs),
        )
        session.start()
        session.feed_audio(np.full(listen.FRAME_SAMPLES * 8, 0.06, dtype=np.float32))
        deadline = time.monotonic() + 1.0
        while not levels and time.monotonic() < deadline:
            time.sleep(0.01)
        session.stop()
        self.assertTrue(levels)
        self.assertGreater(levels[0]["level"], 0.0)


if __name__ == "__main__":
    unittest.main()
