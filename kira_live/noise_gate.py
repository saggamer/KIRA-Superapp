"""Native Live onset gate; dBFS is digital level, not calibrated room dB SPL."""
import math
import numpy as np


class LiveNoiseGate:
    def __init__(self, minimum_dbfs=-48.0):
        if not -60 <= minimum_dbfs <= -10:
            raise ValueError("Live microphone threshold must be between -60 and -10 dBFS")
        self.minimum_rms = 10 ** (minimum_dbfs / 20)
        self.noise_rms = .0005
        self.playback_speech_frames = 0

    def probability(self, frame, probability, *, speaking=False, playback=False):
        rms = math.sqrt(float(np.mean(np.square(frame, dtype=np.float64))))
        # Learn background only when the speech model rejects it. Unlike an
        # amplitude-only detector, steady fan noise cannot become speech.
        if not speaking and probability < .3:
            self.noise_rms = .97 * self.noise_rms + .03 * rms
        # Keep end-of-utterance hysteresis once speech has actually started.
        if speaking:
            return probability
        # Ordinary listening must accept short/quiet greetings. Only speaking
        # over KIRA needs the stricter sustained barge-in test.
        threshold = max(self.minimum_rms, self.noise_rms * 2.0)
        if playback:
            threshold = max(threshold, .02, self.noise_rms * 3.5)
        accepted = rms >= threshold and probability >= (.8 if playback else .55)
        if playback:
            self.playback_speech_frames = self.playback_speech_frames + 1 if accepted else 0
            accepted = accepted and self.playback_speech_frames >= 10
        else:
            self.playback_speech_frames = 0
        return max(.7, probability) if accepted else 0.0
