import threading
import unittest

import numpy as np
from listen import LocalVoiceSession, SAMPLE_RATE, FRAME_SAMPLES


class StreamingDictationTests(unittest.TestCase):
    def make_session(self, **kwargs):
        class Vad:
            def process(self, frame):
                return True, 1.0
            def close(self):
                pass
        return LocalVoiceSession(vad=Vad(), stt=object(), on_partial=lambda *_: None, **kwargs)

    def test_partials_available_before_silence_and_queue_is_bounded(self):
        session = self.make_session()
        frame = np.full(FRAME_SAMPLES, 0.1, dtype=np.float32)
        for _ in range(int(4 * SAMPLE_RATE / FRAME_SAMPLES)):
            session._process_frame(frame)
        self.assertTrue(session._in_turn)
        self.assertTrue(session._turn_queue.empty())
        self.assertEqual(session._partial_queue.qsize(), 1)
        audio, metadata = session._partial_queue.get_nowait()
        self.assertGreater(len(audio), 3 * SAMPLE_RATE)
        self.assertEqual(metadata['turn_id'], session._turn_id)

    def test_final_has_priority_over_pending_partial(self):
        session = self.make_session()
        session._in_turn = True
        session._turn_id = 1
        calls = []
        class Stt:
            def transcribe(self, audio, rate):
                calls.append(len(audio))
                return 'final text', {}
        session.stt = Stt()
        session._partial_queue.put((np.zeros(16000), {'turn_id': 1}))
        session._turn_queue.put((np.zeros(32000), {}))
        session.on_transcript = lambda *_: session._stop_event.set()
        session._stt_loop()
        self.assertEqual(calls, [32000])

    def test_partial_worker_emits_without_completing_turn(self):
        session = self.make_session()
        session._in_turn = True
        session._turn_id = 1
        got = []
        class Stt:
            def transcribe(self, audio, rate):
                return 'live preview', {}
        session.stt = Stt()
        def partial(text, metadata):
            got.append(text)
            session._stop_event.set()
        session.on_partial = partial
        session.on_transcript = lambda *_: self.fail('Preview must not finalize a turn')
        session._partial_queue.put((np.zeros(16000), {'turn_id': 1}))
        session._stt_loop()
        self.assertEqual(got, ['live preview'])


if __name__ == '__main__':
    unittest.main()
