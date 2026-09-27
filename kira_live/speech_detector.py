"""Speech-model gate with exact 256-sample hops for native 320-sample input."""
import numpy as np


class LiveSpeechDetector:
    name = "ten_vad_buffered_256"

    def __init__(self, detector=None):
        if detector is None:
            from ten_vad import TenVad
            detector = TenVad(256, .8)
        self.detector = detector
        self.pending = np.zeros(0, dtype=np.float32)

    def process(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        frame = frame - frame.mean()
        rms = float(np.sqrt(np.mean(frame ** 2)))
        # Built-in microphones can be much quieter than headset inputs.
        # Normalize classification only; the noise/SNR gate sees original audio.
        gain = min(8.0, max(1.0, .035 / max(rms, 1e-6)))
        self.pending = np.concatenate((self.pending, frame * gain))
        probabilities = []
        while len(self.pending) >= 256:
            block, self.pending = self.pending[:256], self.pending[256:]
            pcm = (np.clip(block, -1, 1) * 32767).astype(np.int16)
            probability, _ = self.detector.process(pcm)
            probabilities.append(max(0.0, min(1.0, float(probability))))
        # Average actual model probabilities; demanding the minimum of two
        # hops dropped quiet syllables and reset short-greeting onset.
        confidence = float(np.mean(probabilities)) if probabilities else 0.0
        return confidence >= .8, confidence

    def close(self):
        destroy = getattr(self.detector, "destroy", None)
        if destroy:
            destroy()
