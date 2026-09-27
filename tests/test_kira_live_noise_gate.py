import unittest
import numpy as np
from dataclasses import replace
from kira_live.noise_gate import LiveNoiseGate
from kira_live.config import AudioConfig
from kira_live.audio import VadSegmenter
from kira_live.speech_detector import LiveSpeechDetector


class LiveNoiseTests(unittest.TestCase):
    def test_minor_noise_cannot_start_speech(self):
        gate = LiveNoiseGate()
        self.assertEqual(gate.probability(np.full(320, .015), .2), 0)
        self.assertEqual(gate.probability(np.full(320, .04), 1.0), 1)

    def test_short_loud_spike_does_not_interrupt(self):
        segmenter = VadSegmenter(replace(AudioConfig(), start_speech_ms=240))
        for _ in range(5):
            self.assertEqual(segmenter.process(np.full(320, .1), 1), ())
        segmenter.process(np.zeros(320), 0)
        for _ in range(11):
            self.assertEqual(segmenter.process(np.full(320, .04), 1), ())
        self.assertEqual(segmenter.process(np.full(320, .04), 1)[0].kind, "speech_start")

    def test_end_hysteresis_and_invalid_config(self):
        self.assertEqual(LiveNoiseGate().probability(np.full(320, .01), .5, speaking=True), .5)
        with self.assertRaises(ValueError):
            LiveNoiseGate(-100)

    def test_loud_noise_rejected_without_speech_confidence(self):
        gate = LiveNoiseGate()
        self.assertEqual(gate.probability(np.full(320, .2), .2), 0)

    def test_adapts_to_background_without_learning_speech(self):
        gate = LiveNoiseGate()
        for _ in range(100):
            gate.probability(np.full(320, .02), .1)
        self.assertEqual(gate.probability(np.full(320, .03), .9), 0)
        self.assertEqual(gate.probability(np.full(320, .12), .9), .9)

    def test_short_quiet_greeting_is_accepted_when_not_playing(self):
        gate = LiveNoiseGate()
        segmenter = VadSegmenter(replace(AudioConfig(), start_speech_ms=100, vad_start_threshold=.65))
        events = []
        for _ in range(5):
            frame = np.full(320, .012)
            events.extend(segmenter.process(frame, gate.probability(frame, .6)))
        self.assertEqual([e.kind for e in events], ["speech_start"])

    def test_playback_still_requires_sustained_strong_speech(self):
        gate = LiveNoiseGate()
        for _ in range(9):
            self.assertEqual(gate.probability(np.full(320, .04), .9, playback=True), 0)
        self.assertEqual(gate.probability(np.full(320, .04), .9, playback=True), .9)
        self.assertEqual(gate.probability(np.full(320, .01), .9, playback=True), 0)

    def test_detector_uses_exact_hops_without_losing_samples(self):
        class Detector:
            def __init__(self):
                self.blocks = []
            def process(self, pcm):
                self.blocks.append(pcm.copy())
                return .9, True
        detector = Detector()
        vad = LiveSpeechDetector(detector)
        for _ in range(4):
            self.assertTrue(vad.process(np.full(320, .1))[0])
        self.assertEqual(len(detector.blocks), 5)
        self.assertTrue(all(len(block) == 256 and block.dtype == np.int16 for block in detector.blocks))
        self.assertEqual(len(vad.pending), 0)
