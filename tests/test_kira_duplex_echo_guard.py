import unittest
import numpy as np

from kira_live.playback_echo import DuplexEchoGuard
from kira_live.turn_guard import LiveTurnGuard


class Processor:
    def __init__(self): self.far = []
    def process(self, near, far):
        self.far.append(far.copy())
        return near.copy()


class DuplexEchoGuardTests(unittest.TestCase):
    def test_render_fifo_consumed_once_then_silence(self):
        processor=Processor(); guard=DuplexEchoGuard(processor=processor)
        far=np.linspace(-.1,.1,960,dtype=np.float32)
        guard.add_playback(far,24000,now=1)
        for _ in range(3): guard.clean(np.zeros(320,dtype=np.float32),now=1.1)
        expected=np.interp(np.arange(640)*1.5,np.arange(960),far).astype(np.float32)
        np.testing.assert_array_equal(np.concatenate(processor.far[:2]),expected)
        np.testing.assert_array_equal(processor.far[2],np.zeros(320))

    def test_partial_render_block_is_not_lost(self):
        processor=Processor(); guard=DuplexEchoGuard(processor=processor)
        far=np.linspace(-.1,.1,480,dtype=np.float32)
        guard.add_playback(far,16000,now=1)
        for _ in range(2): guard.clean(np.zeros(320,dtype=np.float32),now=1.1)
        np.testing.assert_array_equal(np.concatenate(processor.far)[:480],far)
        np.testing.assert_array_equal(processor.far[1][160:],np.zeros(160))

    def test_tail_survives_stream_end_without_permanently_muting_mic(self):
        guard=DuplexEchoGuard(processor=Processor())
        guard.add_playback(np.ones(640)*.1,16000,now=1)
        guard.finish_playback(now=1.05)
        self.assertTrue(guard.recent_playback(now=1.3))
        self.assertFalse(guard.recent_playback(now=1.6))

    def test_abort_drops_unplayed_reference_and_keeps_short_room_tail(self):
        processor=Processor(); guard=DuplexEchoGuard(processor=processor)
        guard.add_playback(np.ones(640)*.1,16000,now=1)
        guard.finish_playback(aborted=True,now=1.01)
        guard.clean(np.zeros(320,dtype=np.float32),now=1.03)
        np.testing.assert_array_equal(processor.far[0],np.zeros(320))
        self.assertTrue(guard.recent_playback(now=1.3))
        self.assertFalse(guard.recent_playback(now=1.5))

    def test_echo_text_rejected_only_during_playback(self):
        guard=LiveTurnGuard()
        guard.note_playback('Let us choose one thing to tackle first.',now=1)
        self.assertEqual(guard.accept('one thing to tackle first',now=2,during_playback=True),(False,'speaker_echo'))
        self.assertEqual(guard.accept('one thing to tackle first',now=3),(True,'accepted'))

    def test_real_interruption_and_old_text_are_not_echo(self):
        guard=LiveTurnGuard()
        guard.note_playback('Let us choose one thing to tackle first.',now=1)
        self.assertEqual(guard.accept('Wait, I meant tomorrow.',now=2,during_playback=True),(True,'accepted'))
        self.assertEqual(guard.accept('one thing to tackle first',now=50,during_playback=True),(True,'accepted'))

    def test_webrtc_native_wheel_preserves_capture_shape_and_finite_audio(self):
        guard=DuplexEchoGuard()
        rng=np.random.default_rng(91)
        for _ in range(20):
            far=rng.normal(0,.03,320).astype(np.float32)
            near=rng.normal(0,.02,320).astype(np.float32)
            guard.add_playback(far,16000)
            clean=guard.clean(near)
            self.assertEqual(clean.shape,(320,))
            self.assertTrue(np.isfinite(clean).all())
