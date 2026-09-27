from __future__ import annotations

"""Executable top-2 FFN for the separate sparse-student experiment.

The Listener's acoustic/prosody features bias expert selection. They never
replace the text token embeddings, so an uncertain emotion estimate cannot
silently alter the recognized words. This module is not loaded by the current
KIRA Live 1 app model.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3_next import Qwen3NextMLP
from mlx_lm.models.switch_layers import SwitchGLU


class EmotionConditionedSparseFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        expert_count: int = 24,
        active_experts: int = 2,
        expert_intermediate: int = 2_016,
        shared_intermediate: int = 64,
        emotion_dim: int = 8,
    ):
        super().__init__()
        if not 0 < active_experts <= expert_count:
            raise ValueError("Active expert count must be within the expert pool.")
        self.expert_count = expert_count
        self.active_experts = active_experts
        self.emotion_dim = emotion_dim
        self.gate = nn.Linear(hidden_size, expert_count, bias=False)
        self.emotion_bias = nn.Linear(emotion_dim, expert_count, bias=False)
        self.emotion_bias.weight = mx.zeros_like(self.emotion_bias.weight)
        self.switch_mlp = SwitchGLU(hidden_size, expert_intermediate, expert_count)
        self.shared_expert = Qwen3NextMLP(hidden_size, shared_intermediate)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        self._last_expert_indices = None

    @property
    def last_expert_indices(self) -> mx.array | None:
        return self._last_expert_indices

    def __call__(self, hidden: mx.array, emotion_features: mx.array | None = None) -> mx.array:
        if hidden.ndim != 3:
            raise ValueError("Hidden tokens must be [batch, time, width].")
        scores = self.gate(hidden)
        if emotion_features is not None:
            if emotion_features.shape != (hidden.shape[0], self.emotion_dim):
                raise ValueError("Emotion features must be [batch, emotion_dim].")
            scores = scores + self.emotion_bias(emotion_features.astype(hidden.dtype))[:, None, :]
        probabilities = mx.softmax(scores, axis=-1, precise=True)
        indices = mx.argpartition(probabilities, kth=-self.active_experts, axis=-1)[..., -self.active_experts:]
        # Routing decisions are discrete; only the selected probabilities and
        # expert weights receive gradients, never the integer gather indices.
        indices = mx.stop_gradient(indices)
        weights = mx.take_along_axis(probabilities, indices, axis=-1)
        weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-9)
        sparse = self.switch_mlp(hidden, indices)
        sparse = (sparse * weights[..., None]).sum(axis=-2)
        shared = mx.sigmoid(self.shared_expert_gate(hidden)) * self.shared_expert(hidden)
        self._last_expert_indices = indices
        return sparse + shared
