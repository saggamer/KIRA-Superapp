from __future__ import annotations

"""Trainable MLX building blocks for the KIRA Qwen4-micro conversion.

This module intentionally imports MLX directly and should only be loaded by the
training/model process. The UI and dependency-free control plane never import it.
"""

import mlx.core as mx
import mlx.nn as nn


class GatedResidualStreams(nn.Module):
    """Two or more low-rank mixed residual streams with gated block updates."""

    def __init__(self, hidden_size: int, streams: int = 2, mixer_rank: int = 64):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.stream_count = int(streams)
        width = self.hidden_size * self.stream_count
        self.mix_down = nn.Linear(width, mixer_rank, bias=False)
        self.mix_up = nn.Linear(mixer_rank, width, bias=False)
        self.output_gate = nn.Linear(self.hidden_size, self.stream_count, bias=True)

    def __call__(self, streams: mx.array, block_output: mx.array) -> mx.array:
        if streams.shape[-2:] != (self.stream_count, self.hidden_size):
            raise ValueError("Unexpected residual-stream shape.")
        if block_output.shape != streams.shape[:-2] + (self.hidden_size,):
            raise ValueError("Block output does not match residual streams.")
        flattened = streams.reshape(streams.shape[:-2] + (-1,))
        mixed = flattened + self.mix_up(nn.silu(self.mix_down(flattened)))
        mixed = mixed.reshape(streams.shape)
        gates = mx.sigmoid(self.output_gate(block_output))[..., :, None]
        return mixed + gates * block_output[..., None, :]


class HashedNgramPLE(nn.Module):
    """Compact per-layer hashed n-gram embeddings with causal dilated mixing."""

    def __init__(
        self,
        bucket_count: int,
        embedding_dim: int,
        hidden_size: int,
        *,
        ngram_size: int = 3,
        ngram_sizes: tuple[int, ...] | None = None,
        dilation: int = 2,
        modality_salt: int = 0,
    ):
        super().__init__()
        orders = tuple(int(value) for value in (ngram_sizes or (ngram_size,)))
        if not orders or any(value < 2 for value in orders):
            raise ValueError("At least one n-gram order of two or greater is required.")
        self.bucket_count = int(bucket_count)
        self.embedding_dim = int(embedding_dim)
        self.ngram_sizes = orders
        self.dilation = int(dilation)
        self.modality_salt = int(modality_salt)
        self.embedding = nn.Embedding(self.bucket_count, self.embedding_dim)
        self.order_logits = mx.zeros((len(self.ngram_sizes),))
        self.depthwise_kernel = mx.ones((3, self.embedding_dim)) / 3.0
        self.project = nn.Linear(self.embedding_dim, hidden_size, bias=False)
        # Checkpoint surgery starts as an exact identity; PLE becomes active
        # only after its output projection has received training signal.
        self.project.weight = mx.zeros_like(self.project.weight)

    def _shift(self, values: mx.array, amount: int) -> mx.array:
        if amount <= 0:
            return values
        if amount >= values.shape[-1]:
            return mx.zeros_like(values)
        prefix = mx.zeros(values.shape[:-1] + (amount,), dtype=values.dtype)
        return mx.concatenate([prefix, values[..., :-amount]], axis=-1)

    def __call__(self, token_ids: mx.array) -> mx.array:
        current = token_ids.astype(mx.int64)
        primes = (73_856_093, 19_349_663, 83_492_791, 49_979_687, 67_867_967)
        weights = mx.softmax(self.order_logits, axis=-1)
        features = mx.zeros(token_ids.shape + (self.embedding_dim,))
        for order_index, order in enumerate(self.ngram_sizes):
            hashed = mx.zeros_like(current) + self.modality_salt + order * 1_000_003
            for offset in range(order):
                hashed = hashed + self._shift(current, offset) * primes[offset % len(primes)]
            hashed = hashed % self.bucket_count
            features = features + weights[order_index] * self.embedding(hashed)
        mixed = mx.zeros_like(features)
        for tap in range(3):
            shifted = self._shift_features(features, tap * self.dilation)
            mixed = mixed + shifted * self.depthwise_kernel[tap]
        return self.project(mixed)

    @staticmethod
    def _shift_features(values: mx.array, amount: int) -> mx.array:
        if amount <= 0:
            return values
        if amount >= values.shape[-2]:
            return mx.zeros_like(values)
        prefix = mx.zeros(values.shape[:-2] + (amount, values.shape[-1]), dtype=values.dtype)
        return mx.concatenate([prefix, values[..., :-amount, :]], axis=-2)


class ResidualExpertMoE(nn.Module):
    """Top-k low-rank residual experts layered over the frozen dense FFN."""

    def __init__(
        self,
        hidden_size: int,
        expert_count: int = 8,
        rank: int = 64,
        top_k: int = 2,
    ):
        super().__init__()
        if not 0 < top_k <= expert_count:
            raise ValueError("top_k must be within the expert count.")
        self.expert_count = int(expert_count)
        self.top_k = int(top_k)
        self.router = nn.Linear(hidden_size, self.expert_count, bias=False)
        self.down = [nn.Linear(hidden_size, rank, bias=False) for _ in range(self.expert_count)]
        self.up = [nn.Linear(rank, hidden_size, bias=False) for _ in range(self.expert_count)]

    def __call__(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.router(hidden)
        probabilities = mx.softmax(logits, axis=-1)
        threshold_index = self.expert_count - self.top_k
        threshold = mx.sort(probabilities, axis=-1)[
            ..., threshold_index : threshold_index + 1
        ]
        mask = probabilities >= threshold
        selected = mx.where(mask, probabilities, mx.zeros_like(probabilities))
        selected = selected / mx.maximum(selected.sum(axis=-1, keepdims=True), 1e-9)
        delta = mx.zeros_like(hidden)
        for index in range(self.expert_count):
            expert_delta = self.up[index](nn.silu(self.down[index](hidden)))
            delta = delta + selected[..., index : index + 1] * expert_delta
        return hidden + delta, logits


class ThinkerOnlyExpertMoE(nn.Module):
    """Family-budgeted MoE used only inside the Qwen3.5 Thinker layers.

    Every token activates top-2 experts within each family. Fixed residual
    budgets allocate 25% to emotion/social reasoning, 40% to agentic/action
    reasoning, and 35% to general token reasoning. The pretrained dense FFN is
    still the shared path and is never scaled down.
    """

    FAMILY_INDICES = ((1, 8), (2, 3, 5, 6, 7, 9), (0, 4))
    FAMILY_BUDGETS = (0.25, 0.40, 0.35)

    def __init__(self, hidden_size: int = 1024, rank: int = 128, emotion_dim: int = 8, profile: str = "legacy"):
        super().__init__()
        from .expert_profiles import expert_layout
        self.expert_ranks, self.family_budgets = expert_layout(rank, profile)
        self.router = nn.Linear(hidden_size, 10, bias=False)
        self.emotion_context = nn.Linear(emotion_dim, hidden_size, bias=False)
        self.down = [nn.Linear(hidden_size, r, bias=False) for r in self.expert_ranks]
        self.up = [nn.Linear(r, hidden_size, bias=False) for r in self.expert_ranks]
        # Identity surgery: adapters contribute exactly zero before training.
        for projection in self.up:
            projection.weight = mx.zeros_like(projection.weight)

    def __call__(
        self, hidden: mx.array, emotion_features: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        routed = hidden
        if emotion_features is not None:
            if emotion_features.ndim != 2 or emotion_features.shape[0] != hidden.shape[0]:
                raise ValueError("Emotion features must be [batch, 8].")
            routed = routed + self.emotion_context(emotion_features)[:, None, :]
        logits = self.router(routed)
        delta = mx.zeros_like(hidden)
        for indices, budget in zip(self.FAMILY_INDICES, self.family_budgets):
            family_logits = mx.stack([logits[..., index] for index in indices], axis=-1)
            probabilities = mx.softmax(family_logits, axis=-1)
            top_k = min(2, len(indices))
            threshold = mx.sort(probabilities, axis=-1)[..., -top_k : -top_k + 1]
            selected = mx.where(
                probabilities >= threshold, probabilities, mx.zeros_like(probabilities)
            )
            selected = selected / mx.maximum(selected.sum(axis=-1, keepdims=True), 1e-9)
            family_delta = mx.zeros_like(hidden)
            for local_index, expert_index in enumerate(indices):
                expert = self.up[expert_index](nn.silu(self.down[expert_index](hidden)))
                family_delta = family_delta + selected[..., local_index : local_index + 1] * expert
            delta = delta + budget * family_delta
        return hidden + delta, logits


class GatedAudioFusion(nn.Module):
    """Project aligned acoustic states and gate them into a decoder layer."""

    def __init__(self, audio_dim: int, hidden_size: int):
        super().__init__()
        self.project = nn.Linear(audio_dim, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size * 2, hidden_size, bias=True)

    def __call__(self, hidden: mx.array, aligned_audio: mx.array) -> mx.array:
        projected = self.project(aligned_audio)
        if projected.shape != hidden.shape:
            raise ValueError("Audio states must be aligned to the decoder sequence.")
        gate = mx.sigmoid(self.gate(mx.concatenate([hidden, projected], axis=-1)))
        return hidden + gate * projected


class GatedMemoryFusion(nn.Module):
    """Low-rank retrieval attention over externally persisted memory slots."""

    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        self.rank = int(rank)
        self.query = nn.Linear(hidden_size, self.rank, bias=False)
        self.key = nn.Linear(hidden_size, self.rank, bias=False)
        self.value_down = nn.Linear(hidden_size, self.rank, bias=False)
        self.value_up = nn.Linear(self.rank, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1, bias=True)

    def __call__(self, hidden: mx.array, memory_slots: mx.array) -> mx.array:
        if hidden.ndim != 3 or memory_slots.ndim != 3:
            raise ValueError("Hidden states and memory slots must be [batch, tokens, width].")
        if hidden.shape[0] != memory_slots.shape[0] or hidden.shape[-1] != memory_slots.shape[-1]:
            raise ValueError("Memory batch and width must match hidden states.")
        queries = self.query(hidden)
        keys = self.key(memory_slots)
        scores = (queries @ mx.swapaxes(keys, -1, -2)) / (self.rank ** 0.5)
        weights = mx.softmax(scores, axis=-1)
        values = self.value_down(memory_slots)
        retrieved = self.value_up(weights @ values)
        return hidden + mx.sigmoid(self.gate(hidden)) * retrieved


class SharedMultimodalNgramPLE(nn.Module):
    """One 2/3/4-gram parameter table shared by listener, thinker, and talker IDs."""

    DOMAIN_SALTS = (17_171, 43_117, 79_919)

    def __init__(
        self,
        bucket_count: int,
        embedding_dim: int,
        hidden_size: int,
        ngram_sizes: tuple[int, ...] = (2, 3, 4),
        dilation: int = 2,
    ):
        super().__init__()
        self.bucket_count = int(bucket_count)
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.ngram_sizes = tuple(int(value) for value in ngram_sizes)
        self.dilation = int(dilation)
        self.embedding = nn.Embedding(self.bucket_count, self.embedding_dim)
        self.order_logits = mx.zeros((len(self.ngram_sizes),))
        self.depthwise_kernel = mx.ones((3, self.embedding_dim)) / 3.0
        self.projectors = [
            nn.Linear(self.embedding_dim, self.hidden_size, bias=False)
            for _ in self.DOMAIN_SALTS
        ]

    @staticmethod
    def _shift_ids(values: mx.array, amount: int) -> mx.array:
        if amount <= 0:
            return values
        if amount >= values.shape[-1]:
            return mx.zeros_like(values)
        prefix = mx.zeros(values.shape[:-1] + (amount,), dtype=values.dtype)
        return mx.concatenate([prefix, values[..., :-amount]], axis=-1)

    @staticmethod
    def _shift_features(values: mx.array, amount: int) -> mx.array:
        if amount <= 0:
            return values
        if amount >= values.shape[-2]:
            return mx.zeros_like(values)
        prefix = mx.zeros(values.shape[:-2] + (amount, values.shape[-1]), dtype=values.dtype)
        return mx.concatenate([prefix, values[..., :-amount, :]], axis=-2)

    def __call__(self, unit_ids: mx.array, domain: int) -> mx.array:
        domain = int(domain)
        if not 0 <= domain < len(self.DOMAIN_SALTS):
            raise ValueError("Unknown multimodal n-gram domain.")
        current = unit_ids.astype(mx.int64)
        primes = (73_856_093, 19_349_663, 83_492_791, 49_979_687)
        weights = mx.softmax(self.order_logits, axis=-1)
        features = mx.zeros(unit_ids.shape + (self.embedding_dim,))
        for order_index, order in enumerate(self.ngram_sizes):
            hashed = mx.zeros_like(current) + self.DOMAIN_SALTS[domain] + order * 1_000_003
            for offset in range(order):
                hashed = hashed + self._shift_ids(current, offset) * primes[offset % len(primes)]
            features = features + weights[order_index] * self.embedding(hashed % self.bucket_count)
        mixed = mx.zeros_like(features)
        for tap in range(3):
            mixed = mixed + self._shift_features(features, tap * self.dilation) * self.depthwise_kernel[tap]
        return self.projectors[domain](mixed)


class ListenerToThinkerBridge(nn.Module):
    """Temporally resample Listener states into the Thinker semantic space.

    Qwen3-ASR emits roughly three acoustic states per text token.  Projecting
    every acoustic frame independently made the first bridge overfit because
    it was asked to imitate repeated token embeddings without any temporal
    context.  This bridge first resamples, then mixes a local acoustic window,
    while retaining a full-rank path between the two 1,024-wide spaces.
    """

    def __init__(self, listener_dim: int, thinker_dim: int, rank: int = 256):
        super().__init__()
        self.listener_dim = int(listener_dim)
        self.thinker_dim = int(thinker_dim)
        self.input_norm = nn.RMSNorm(self.listener_dim, eps=1e-6)
        self.down = nn.Linear(self.listener_dim, rank, bias=True)
        self.temporal = nn.Conv1d(rank, rank, kernel_size=5, padding=2, bias=True)
        self.up = nn.Linear(rank, self.thinker_dim, bias=True)
        self.skip = nn.Linear(self.listener_dim, self.thinker_dim, bias=False)
        self.output_norm = nn.RMSNorm(self.thinker_dim, eps=1e-6)
        self.gate = nn.Linear(self.thinker_dim * 2, self.thinker_dim, bias=True)

    @staticmethod
    def align(states: mx.array, target_length: int) -> mx.array:
        if states.ndim != 3 or states.shape[-2] <= 0 or target_length <= 0:
            raise ValueError("Bridge states must be [batch, positive_time, width].")
        indices = (mx.arange(target_length) * states.shape[-2] // target_length).astype(mx.int32)
        return mx.take(states, indices, axis=-2)

    def __call__(self, thinker_hidden: mx.array, listener_hidden: mx.array) -> tuple[mx.array, mx.array]:
        aligned = self.project_sequence(listener_hidden, thinker_hidden.shape[-2])
        gate = mx.sigmoid(self.gate(mx.concatenate([thinker_hidden, aligned], axis=-1)))
        return thinker_hidden + gate * aligned, aligned

    def project_sequence(
        self, listener_hidden: mx.array, target_length: int | None = None
    ) -> mx.array:
        if listener_hidden.ndim != 3 or listener_hidden.shape[-1] != self.listener_dim:
            raise ValueError("Listener states must be [batch, time, listener_dim].")
        if target_length is None:
            target_length = max(1, round(listener_hidden.shape[-2] / 3))
        resampled = self.align(listener_hidden, target_length)
        normalized = self.input_norm(resampled)
        local = nn.silu(self.down(normalized))
        local = nn.silu(self.temporal(local))
        return self.output_norm(self.skip(normalized) + self.up(local))


class ThinkerToTalkerBridge(nn.Module):
    """Convert semantic token states directly into streaming Talker conditioning."""

    def __init__(self, thinker_dim: int, talker_dim: int, rank: int = 64, emotion_dim: int = 8):
        super().__init__()
        self.down = nn.Linear(thinker_dim, rank, bias=False)
        self.up = nn.Linear(rank, talker_dim, bias=False)
        self.style = nn.Linear(emotion_dim, talker_dim, bias=True)
        self.style_gate = nn.Linear(talker_dim, talker_dim, bias=True)
        # Untrained emotion injection must not corrupt speech conditioning.
        # Emotional information still reaches the Talker through the Thinker's
        # trained hidden states; this explicit style branch stays neutral until
        # a paired expressive-speech stage has real supervision.
        self.style.weight = mx.zeros_like(self.style.weight)
        self.style.bias = mx.zeros_like(self.style.bias)

    def __call__(self, thinker_hidden: mx.array, emotion_features: mx.array) -> mx.array:
        if emotion_features.ndim != 2 or emotion_features.shape[0] != thinker_hidden.shape[0]:
            raise ValueError("Emotion features must be [batch, emotion_width].")
        semantic = self.up(nn.silu(self.down(thinker_hidden)))
        style = mx.tanh(self.style(emotion_features))[:, None, :]
        return semantic + mx.sigmoid(self.style_gate(semantic)) * style


class KiraLiveCompositeBridge(nn.Module):
    """Executable joint bridge: one graph, three weight islands, zero prompt handoffs."""

    def __init__(
        self,
        listener_dim: int = 1024,
        thinker_dim: int = 1024,
        talker_dim: int = 1024,
        bucket_count: int = 262_144,
        embedding_dim: int = 64,
        expert_count: int = 10,
        expert_rank: int = 64,
        memory_rank: int = 64,
        bridge_rank: int = 256,
        emotion_dim: int = 8,
    ):
        super().__init__()
        self.expert_count = int(expert_count)
        self.ngram = SharedMultimodalNgramPLE(
            bucket_count, embedding_dim, thinker_dim, ngram_sizes=(2, 3, 4)
        )
        self.listener_bridge = ListenerToThinkerBridge(listener_dim, thinker_dim, bridge_rank)
        self.residual = GatedResidualStreams(thinker_dim, streams=2, mixer_rank=bridge_rank)
        self.memory = GatedMemoryFusion(thinker_dim, memory_rank)
        self.talker_bridge = ThinkerToTalkerBridge(
            thinker_dim, talker_dim, bridge_rank, emotion_dim
        )
        self.codec_project = nn.Linear(thinker_dim, talker_dim, bias=False)

    @staticmethod
    def _align(states: mx.array, target_length: int) -> mx.array:
        return ListenerToThinkerBridge.align(states, target_length)

    def __call__(
        self,
        *,
        thinker_hidden: mx.array,
        thinker_token_ids: mx.array,
        listener_hidden: mx.array,
        listener_unit_ids: mx.array,
        talker_code_ids: mx.array,
        memory_slots: mx.array,
        emotion_features: mx.array,
    ) -> dict[str, mx.array]:
        text_ngram = self.ngram(thinker_token_ids, 1)
        listener_ngram = self._align(
            self.ngram(listener_unit_ids, 0), thinker_hidden.shape[-2]
        )
        semantic = thinker_hidden + text_ngram
        live = thinker_hidden + listener_ngram
        semantic, aligned_listener = self.listener_bridge(semantic, listener_hidden)
        streams = mx.stack([semantic, live + aligned_listener], axis=-2)
        streams = self.residual(streams, (semantic + live) * 0.5)
        combined = streams.mean(axis=-2)
        combined = self.memory(combined, memory_slots)

        talker_conditioning = self.talker_bridge(combined, emotion_features)
        codec_ngram = self.ngram(talker_code_ids, 2)
        codec_conditioning = self.codec_project(codec_ngram)
        return {
            "thinker_hidden": combined,
            "talker_conditioning": talker_conditioning,
            "codec_conditioning": codec_conditioning,
            # Expert routing occurs inside Thinker layers 9-24, not in this
            # cross-modal bridge or either voice island.
            "router_logits": mx.zeros(combined.shape[:-1] + (self.expert_count,)),
            "residual_streams": streams,
            "aligned_listener": aligned_listener,
        }

    def audio_prefix(
        self,
        *,
        listener_hidden: mx.array,
        listener_unit_ids: mx.array,
        memory_slots: mx.array,
        target_length: int | None = None,
    ) -> mx.array:
        """Create a full-rate Thinker prefix directly from acoustic states."""
        semantic = self.listener_bridge.project_sequence(listener_hidden, target_length)
        listener_ngram = self._align(
            self.ngram(listener_unit_ids, 0), semantic.shape[-2]
        )
        live = semantic + listener_ngram
        streams = mx.stack((semantic, live), axis=-2)
        streams = self.residual(streams, (semantic + live) * 0.5)
        return self.memory(streams.mean(axis=-2), memory_slots)
