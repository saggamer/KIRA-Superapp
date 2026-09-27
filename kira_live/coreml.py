from __future__ import annotations

from pathlib import Path
import time
from typing import Sequence

from .emotion import EMOTIONS, EmotionState


class CoreMLAcousticRunner:
    """Runtime adapter for the KIRA streaming acoustic Core ML contract.

    The accepted model takes a mono float32 `audio` tensor and returns
    `vad_probability`, `emotion_logits`, `valence`, `arousal`, and `confidence`.
    Imports are lazy so source tests do not require Core ML Tools.
    """

    def __init__(self, model_path: str | Path):
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            import coremltools as ct
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "Core ML Tools is required to load the acoustic model."
            ) from error
        self._np = np
        self.model = ct.models.MLModel(str(path))
        self.state = self.model.make_state() if hasattr(self.model, "make_state") else None

    @staticmethod
    def _scalar(value) -> float:
        try:
            return float(value.item())
        except AttributeError:
            return float(value)

    def predict(self, audio: Sequence[float]) -> tuple[float, EmotionState]:
        array = self._np.asarray(audio, dtype=self._np.float32)[None, :]
        kwargs = {"state": self.state} if self.state is not None else {}
        output = self.model.predict({"audio": array}, **kwargs)
        missing = {
            "vad_probability", "emotion_logits", "valence", "arousal", "confidence"
        } - set(output)
        if missing:
            raise RuntimeError("Core ML acoustic output is missing: " + ", ".join(sorted(missing)))

        logits = self._np.asarray(output["emotion_logits"], dtype=self._np.float64).reshape(-1)
        if logits.size != len(EMOTIONS):
            raise RuntimeError(f"Expected {len(EMOTIONS)} emotion logits, received {logits.size}.")
        logits -= logits.max()
        probabilities = self._np.exp(logits)
        probabilities /= max(float(probabilities.sum()), 1e-9)
        state = EmotionState(
            probabilities={name: float(probabilities[index]) for index, name in enumerate(EMOTIONS)},
            valence=max(-1.0, min(1.0, self._scalar(output["valence"]))),
            arousal=max(0.0, min(1.0, self._scalar(output["arousal"]))),
            confidence=max(0.0, min(1.0, self._scalar(output["confidence"]))),
            observed_at=time.monotonic(),
        )
        return max(0.0, min(1.0, self._scalar(output["vad_probability"]))), state


class CoreMLEmotionHeadRunner:
    """Run the accepted 8-class n-gram emotion head on Apple accelerators."""

    def __init__(self, model_path: str | Path):
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            import coremltools as ct
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Core ML Tools is required for the emotion head.") from error
        self._np = np
        self.model = ct.models.MLModel(str(path))

    def predict(self, pooled_listener: Sequence[float]):
        features = self._np.asarray(pooled_listener, dtype=self._np.float32).reshape(1, -1)
        if features.shape != (1, 8192):
            raise ValueError(f"Expected pooled Listener shape (1, 8192), got {features.shape}.")
        output = self.model.predict({"pooled_listener": features})
        return {
            "emotion_logits": self._np.asarray(output["emotion_logits"]).reshape(8),
            "prosody": self._np.asarray(output["prosody"]).reshape(3),
        }
