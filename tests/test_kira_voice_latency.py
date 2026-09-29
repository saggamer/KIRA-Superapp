import threading
import unittest
from kira_live.audio_devices import pick_audio_device
from kira_live.event_dispatch import LiveEventDispatcher


class VoiceLatencyTests(unittest.TestCase):
    def test_quantization_only_changes_talker_linears_not_decoder_or_files(self):
        import mlx.core as mx
        import mlx.nn as nn
        from types import SimpleNamespace
        from kira_live.mlx_voice import optimize_talker
        class Talker(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
                self.small = nn.Linear(15, 10)
        codec = nn.Linear(64, 64)
        saved = codec.weight
        model = SimpleNamespace(talker=Talker(), speech_tokenizer=SimpleNamespace(decoder=codec))
        self.assertEqual(optimize_talker(model, 0)['quantized_linear_layers'], 0)
        result = optimize_talker(model, 8)
        self.assertEqual(result['quantized_linear_layers'], 1)
        self.assertIsInstance(model.talker.linear, nn.QuantizedLinear)
        self.assertIsInstance(model.talker.small, nn.Linear)
        self.assertIs(codec.weight, saved)
        self.assertTrue(bool(mx.all(mx.isfinite(model.talker.linear(mx.ones((1, 64))))).item()))
        with self.assertRaises(ValueError):
            optimize_talker(model, 4)

    def test_missing_portaudio_default_uses_real_device(self):
        devices = [{'max_input_channels': None, 'max_output_channels': 2},
                   {'max_input_channels': 1, 'max_output_channels': 0}]
        for missing in (None, -1, 'invalid'):
            self.assertEqual(pick_audio_device(devices, missing, 'max_input_channels'), 1)
        self.assertEqual(pick_audio_device(devices, 0, 'max_output_channels'), 0)
        with self.assertRaises(RuntimeError):
            pick_audio_device([], None, 'max_input_channels')

    def test_slow_ui_does_not_block_capture_and_meter_events_coalesce(self):
        entered, release = threading.Event(), threading.Event()
        received = []
        def callback(kind, payload):
            if kind == 'TRANSCRIPT':
                entered.set()
                release.wait(1)
            received.append((kind, payload))
        dispatcher = LiveEventDispatcher(callback)
        try:
            dispatcher.submit('TRANSCRIPT', {'text': 'hello'})
            self.assertTrue(entered.wait(1))
            # Caller completes all submissions while the callback is blocked.
            for level in range(1000):
                dispatcher.submit('AUDIO_LEVEL', {'level': level})
            dispatcher.submit('RESPONSE_TEXT', {'text': 'Hi.'})
            dispatcher.submit('FIRST_AUDIO', {})
            self.assertEqual(len(dispatcher.pending), 2)
            self.assertEqual(dispatcher.level[1]['level'], 999)
        finally:
            release.set()
            dispatcher.close()
        self.assertEqual([kind for kind, _ in received if kind != 'AUDIO_LEVEL'],
                         ['TRANSCRIPT', 'RESPONSE_TEXT', 'FIRST_AUDIO'])

    def test_callback_failure_does_not_kill_dispatcher(self):
        done = threading.Event()
        def callback(kind, payload):
            if kind == 'broken':
                raise RuntimeError('UI fixture error')
            done.set()
        dispatcher = LiveEventDispatcher(callback)
        dispatcher.submit('broken', {})
        dispatcher.submit('LISTENING_STARTED', {})
        self.assertTrue(done.wait(1))
        dispatcher.close()
        self.assertEqual(dispatcher.last_error, 'UI fixture error')
        self.assertFalse(dispatcher.worker.is_alive())


if __name__ == '__main__':
    unittest.main()
