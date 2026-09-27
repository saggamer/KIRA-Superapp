from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import AudioConfig


@dataclass(frozen=True)
class AudioSegmentEvent:
    kind: str
    probability: float
    samples: tuple[float, ...] = ()


class VadSegmenter:
    """Turn frame-level VAD probabilities into complete utterances.

    The component is model-independent: Core ML, TEN VAD, or a deterministic
    test source can provide probabilities while this class owns pre-roll,
    threshold hysteresis, end silence, and maximum utterance length.
    """

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.frame_samples = self.config.samples_per_frame
        self.pre_roll_frames = max(1, self.config.pre_roll_ms // self.config.frame_ms)
        self.start_speech_frames = max(
            1, self.config.start_speech_ms // self.config.frame_ms
        )
        self.end_silence_frames = max(1, self.config.end_silence_ms // self.config.frame_ms)
        self.max_frames = max(
            1, int(self.config.max_utterance_seconds * 1_000 / self.config.frame_ms)
        )
        self._pre_roll: deque[tuple[float, ...]] = deque(maxlen=self.pre_roll_frames)
        self._utterance: list[tuple[float, ...]] = []
        self._silence_frames = 0
        self._speech_frames = 0
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        self._pre_roll.clear()
        self._utterance.clear()
        self._silence_frames = 0
        self._speech_frames = 0
        self._speaking = False

    def process(
        self, frame: Sequence[float] | Iterable[float], probability: float
    ) -> tuple[AudioSegmentEvent, ...]:
        samples = tuple(float(sample) for sample in frame)
        if len(samples) != self.frame_samples:
            raise ValueError(
                f"Expected {self.frame_samples} samples, received {len(samples)}."
            )
        probability = max(0.0, min(1.0, float(probability)))
        events: list[AudioSegmentEvent] = []

        if not self._speaking:
            self._pre_roll.append(samples)
            if probability < self.config.vad_start_threshold:
                self._speech_frames = 0
                return ()
            self._speech_frames += 1
            if self._speech_frames < self.start_speech_frames:
                return ()
            self._speaking = True
            self._utterance = list(self._pre_roll)
            self._pre_roll.clear()
            self._silence_frames = 0
            events.append(AudioSegmentEvent("speech_start", probability))
            return tuple(events)

        self._utterance.append(samples)
        if probability <= self.config.vad_end_threshold:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        timed_out = len(self._utterance) >= self.max_frames
        ended = self._silence_frames >= self.end_silence_frames
        if ended or timed_out:
            flattened = tuple(sample for item in self._utterance for sample in item)
            kind = "speech_timeout" if timed_out else "speech_end"
            events.append(AudioSegmentEvent(kind, probability, flattened))
            self.reset()
        return tuple(events)
