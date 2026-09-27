from __future__ import annotations

"""Full-duplex control plane for always-on acoustic n-grams and barge-in."""

from collections import Counter, deque
from dataclasses import dataclass
import math
import threading
from typing import Callable, Iterable, Sequence

from .audio import AudioSegmentEvent, VadSegmenter
from .config import AudioConfig
from .session import KiraLiveSession


@dataclass(frozen=True)
class GenerationTicket:
    epoch: int
    cancelled: threading.Event


class GenerationGate:
    """Owns cancellation for Thinker token generation and Talker playback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._epoch = 0
        self._active: threading.Event | None = None

    def begin(self) -> GenerationTicket:
        with self._lock:
            self._epoch += 1
            self._active = threading.Event()
            return GenerationTicket(self._epoch, self._active)

    def interrupt(self) -> int:
        with self._lock:
            self._epoch += 1
            if self._active is not None:
                self._active.set()
            return self._epoch


class ContinuousAcousticNgrams:
    """Keep bounded 2/3/4-gram acoustic state for every microphone frame.

    IDs summarize energy, zero-crossing rate, and peak level. They are acoustic
    units, never transcript tokens, and are cleared from raw audio immediately.
    """

    def __init__(self, history_frames: int = 512):
        self.units: deque[int] = deque(maxlen=history_frames)
        self.counts = {order: Counter() for order in (2, 3, 4)}
        self.frames_seen = 0

    @staticmethod
    def quantize(frame: Sequence[float] | Iterable[float]) -> int:
        values = tuple(float(value) for value in frame)
        if not values:
            return 0
        energy = math.sqrt(sum(value * value for value in values) / len(values))
        crossings = sum((left < 0) != (right < 0) for left, right in zip(values, values[1:]))
        zcr = crossings / max(len(values) - 1, 1)
        peak = max(abs(value) for value in values)
        energy_bin = min(63, int(energy * 256))
        zcr_bin = min(31, int(zcr * 64))
        peak_bin = min(31, int(peak * 32))
        return (energy_bin << 10) | (zcr_bin << 5) | peak_bin

    def ingest(self, frame: Sequence[float] | Iterable[float]) -> int:
        unit = self.quantize(frame)
        self.units.append(unit)
        self.frames_seen += 1
        values = tuple(self.units)
        for order in (2, 3, 4):
            if len(values) >= order:
                self.counts[order][values[-order:]] += 1
        return unit

    def snapshot(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "recent_unit_ids": tuple(self.units),
            "unique_ngrams": {
                order: len(counts) for order, counts in self.counts.items()
            },
        }


class KiraLiveDuplexController:
    """Always listen; interrupt generation/playback as soon as speech starts."""

    def __init__(
        self,
        session: KiraLiveSession,
        *,
        audio_config: AudioConfig | None = None,
        stop_playback: Callable[[], None] | None = None,
        on_utterance: Callable[[tuple[float, ...], dict], None] | None = None,
    ):
        self.session = session
        self.segmenter = VadSegmenter(audio_config or session.config.audio)
        self.ngrams = ContinuousAcousticNgrams()
        self.generation = GenerationGate()
        self.stop_playback = stop_playback
        self.on_utterance = on_utterance

    def begin_generation(self) -> GenerationTicket:
        return self.generation.begin()

    def ingest_frame(
        self, frame: Sequence[float] | Iterable[float], speech_probability: float
    ) -> tuple[AudioSegmentEvent, ...]:
        # This executes for every frame, including while KIRA is speaking.
        self.ngrams.ingest(frame)
        events = self.segmenter.process(frame, speech_probability)
        for event in events:
            if event.kind == "speech_start":
                self.generation.interrupt()
                if self.stop_playback is not None:
                    self.stop_playback()
                self.session.voice_started(event.probability)
            elif event.kind in {"speech_end", "speech_timeout"}:
                self.session.voice_ended()
                if self.on_utterance is not None:
                    self.on_utterance(event.samples, self.ngrams.snapshot())
        return events
