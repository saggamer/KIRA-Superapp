import unittest
import numpy as np
from kira_live.playback_echo import PlaybackEchoFilter
from kira_live.duplex import KiraLiveDuplexController
from kira_live.session import KiraLiveSession


class PlaybackEchoTests(unittest.TestCase):
    def test_delayed_scaled_echo_is_removed(self):
        rng = np.random.default_rng(5)
        reference = rng.normal(0, .12, 4000).astype(np.float32)
        guard = PlaybackEchoFilter()
        guard.add_playback(reference, 16000, now=1)
        mic = .4 * reference[1000:1320]
        cleaned = guard.clean(mic, now=1.1)
        self.assertLess(float(np.linalg.norm(cleaned)), float(np.linalg.norm(mic)) * .01)

    def test_near_end_voice_survives_echo_subtraction(self):
        rng = np.random.default_rng(8)
        reference = rng.normal(0, .12, 4000).astype(np.float32)
        user = rng.normal(0, .025, 320).astype(np.float32)
        guard = PlaybackEchoFilter()
        guard.add_playback(reference, 16000, now=1)
        cleaned = guard.clean(.4 * reference[1000:1320] + user, now=1.1)
        self.assertGreater(float(np.linalg.norm(cleaned)), float(np.linalg.norm(user)) * .8)

    def test_room_coloration_is_removed(self):
        rng = np.random.default_rng(9)
        reference = rng.normal(0, .12, 4000).astype(np.float32)
        guard = PlaybackEchoFilter()
        guard.add_playback(reference, 16000, now=1)
        mic = .4 * reference[1000:1320] + .12 * reference[999:1319] - .05 * reference[998:1318]
        cleaned = guard.clean(mic, now=1.1)
        self.assertLess(float(np.linalg.norm(cleaned)), float(np.linalg.norm(mic)) * .01)

    def test_talker_sample_rate_is_resampled_for_microphone(self):
        rng = np.random.default_rng(10)
        reference = rng.normal(0, .12, 6000).astype(np.float32)
        positions = np.arange(4000) * 1.5
        mic_reference = np.interp(positions, np.arange(6000), reference).astype(np.float32)
        guard = PlaybackEchoFilter()
        guard.add_playback(reference, 24000, now=1)
        mic = .4 * mic_reference[1000:1320]
        cleaned = guard.clean(mic, now=1.1)
        self.assertLess(float(np.linalg.norm(cleaned)), float(np.linalg.norm(mic)) * .01)

    def test_reference_expires_and_unrelated_speech_is_unchanged(self):
        guard = PlaybackEchoFilter()
        rng = np.random.default_rng(2)
        guard.add_playback(rng.normal(0, .1, 4000), 16000, now=1)
        user = rng.normal(0, .1, 320).astype(np.float32)
        np.testing.assert_array_equal(guard.clean(user, now=1.1), user)
        np.testing.assert_array_equal(guard.clean(user, now=3), user)

    def test_echo_does_not_cancel_but_user_can_interrupt(self):
        session = KiraLiveSession()
        session.start()
        session.response_started()
        duplex = KiraLiveDuplexController(session)
        ticket = duplex.begin_generation()
        guard = PlaybackEchoFilter()
        rng = np.random.default_rng(4)
        reference = rng.normal(0, .12, 4000).astype(np.float32)
        guard.add_playback(reference, 16000, now=1)
        for _ in range(12):
            cleaned = guard.clean(reference[1000:1320] * .4, now=1.1)
            duplex.ingest_frame(cleaned, 1.0 if np.sqrt(np.mean(cleaned ** 2)) > .009 else 0.0)
        self.assertFalse(ticket.cancelled.is_set())
        for _ in range(6):
            user = rng.normal(0, .08, 320).astype(np.float32)
            cleaned = guard.clean(user, now=1.1)
            duplex.ingest_frame(cleaned, 1.0)
        self.assertTrue(ticket.cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
