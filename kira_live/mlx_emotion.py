from __future__ import annotations

"""Small trainable emotion/prosody head over frozen Listener states."""

import mlx.core as mx
import mlx.nn as nn


EMOTIONS = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
)


class ListenerEmotionHead(nn.Module):
    """Predict categorical emotion and continuous V/A/D from pooled audio."""

    def __init__(self, input_size: int = 2048, hidden_size: int = 128):
        super().__init__()
        self.trunk = nn.Linear(input_size, hidden_size, bias=True)
        self.emotion = nn.Linear(hidden_size, len(EMOTIONS), bias=True)
        self.prosody = nn.Linear(hidden_size, 3, bias=True)

    def __call__(self, pooled_listener: mx.array) -> tuple[mx.array, mx.array]:
        hidden = nn.silu(self.trunk(pooled_listener))
        return self.emotion(hidden), mx.sigmoid(self.prosody(hidden))


class NgramListenerEmotionHead(nn.Module):
    """Compact classifier over multi-order temporal Listener statistics.

    The input contains mean/std statistics for the Listener stream and its
    causal 1/2/4-frame deltas.  Those deltas are the continuous-audio analogue
    of 2/3/4-gram transitions: they retain changes in pace, energy and spectral
    state without adding an MoE to the Listener island.
    """

    def __init__(self, input_size: int = 8192, hidden_size: int = 256):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.trunk = nn.Linear(input_size, hidden_size, bias=True)
        self.refine = nn.Linear(hidden_size, hidden_size // 2, bias=True)
        self.emotion = nn.Linear(hidden_size // 2, len(EMOTIONS), bias=True)
        self.prosody = nn.Linear(hidden_size // 2, 3, bias=True)

    def __call__(self, pooled_listener: mx.array) -> tuple[mx.array, mx.array]:
        hidden = nn.silu(self.trunk(self.input_norm(pooled_listener)))
        hidden = nn.silu(self.refine(hidden))
        return self.emotion(hidden), mx.sigmoid(self.prosody(hidden))


def pool_listener_states(hidden: mx.array) -> mx.array:
    """Mean/std pooling preserves level and temporal variation per channel."""
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != 1024:
        raise ValueError("Listener states must be [1, time, 1024].")
    values = hidden[0].astype(mx.float32)
    mean = mx.mean(values, axis=0)
    variance = mx.mean(mx.square(values - mean), axis=0)
    return mx.concatenate((mean, mx.sqrt(variance + 1e-6)), axis=-1)


def pool_listener_ngram_states(hidden: mx.array) -> mx.array:
    """Pool base state plus causal 2/3/4-step transition statistics."""
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != 1024:
        raise ValueError("Listener states must be [1, time, 1024].")
    values = hidden[0].astype(mx.float32)

    def mean_std(items: mx.array) -> tuple[mx.array, mx.array]:
        mean = mx.mean(items, axis=0)
        variance = mx.mean(mx.square(items - mean), axis=0)
        return mean, mx.sqrt(variance + 1e-6)

    features = list(mean_std(values))
    for lag in (1, 2, 4):
        if values.shape[0] > lag:
            delta = values[lag:] - values[:-lag]
        else:
            delta = mx.zeros((1, values.shape[-1]), dtype=values.dtype)
        features.extend(mean_std(delta))
    return mx.concatenate(features, axis=-1)
