from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping


EMOTIONS = ("neutral", "joy", "sadness", "anger", "fear", "surprise")


def _normalized(values: Mapping[str, float]) -> dict[str, float]:
    cleaned = {name: max(0.0, float(values.get(name, 0.0))) for name in EMOTIONS}
    total = sum(cleaned.values())
    if total <= 1e-9:
        return {name: 1.0 if name == "neutral" else 0.0 for name in EMOTIONS}
    return {name: value / total for name, value in cleaned.items()}


@dataclass(frozen=True)
class EmotionState:
    probabilities: dict[str, float]
    valence: float
    arousal: float
    confidence: float
    observed_at: float

    @property
    def label(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    def prompt_context(self) -> str:
        """Expose acoustic cues as uncertainty-aware context, never as a fact."""
        if self.confidence < 0.35:
            return "Voice cue: uncertain; do not infer an emotion."
        return (
            f"Voice cue: possible {self.label} "
            f"(confidence {self.confidence:.2f}, valence {self.valence:.2f}, "
            f"arousal {self.arousal:.2f}). Treat this as a soft cue, not a fact."
        )


class EmotionTracker:
    """Smooth noisy acoustic predictions and decay them toward neutral."""

    def __init__(self, smoothing: float = 0.72, neutral_half_life: float = 8.0):
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1).")
        self.smoothing = float(smoothing)
        self.neutral_half_life = max(0.5, float(neutral_half_life))
        self._probabilities = _normalized({"neutral": 1.0})
        self._valence = 0.0
        self._arousal = 0.0
        self._confidence = 0.0
        self._observed_at = time.monotonic()

    def update(
        self,
        probabilities: Mapping[str, float],
        *,
        valence: float,
        arousal: float,
        confidence: float,
        observed_at: float | None = None,
    ) -> EmotionState:
        incoming = _normalized(probabilities)
        keep = self.smoothing
        take = 1.0 - keep
        self._probabilities = _normalized(
            {
                name: keep * self._probabilities[name] + take * incoming[name]
                for name in EMOTIONS
            }
        )
        self._valence = keep * self._valence + take * max(-1.0, min(1.0, float(valence)))
        self._arousal = keep * self._arousal + take * max(0.0, min(1.0, float(arousal)))
        self._confidence = keep * self._confidence + take * max(0.0, min(1.0, float(confidence)))
        self._observed_at = time.monotonic() if observed_at is None else float(observed_at)
        return self.state()

    def state(self, now: float | None = None) -> EmotionState:
        timestamp = time.monotonic() if now is None else float(now)
        age = max(0.0, timestamp - self._observed_at)
        decay = math.exp(-math.log(2.0) * age / self.neutral_half_life)
        probabilities = {
            name: value * decay for name, value in self._probabilities.items()
        }
        probabilities["neutral"] += 1.0 - decay
        return EmotionState(
            probabilities=_normalized(probabilities),
            valence=self._valence * decay,
            arousal=self._arousal * decay,
            confidence=self._confidence * decay,
            observed_at=self._observed_at,
        )
