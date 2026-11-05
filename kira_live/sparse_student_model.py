from __future__ import annotations

"""Executable Experimental Live 1 graph, separate from the deployed model.

It still requires teacher-to-student initialization and training. The graph
preserves lexical token embeddings while carrying Listener acoustics in a
separate gated social stream; prosody also biases sparse expert routing.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextMLP

from .mlx_layers import HashedNgramPLE
from .sparse_emotion_moe import EmotionConditionedSparseFFN
from .sparse_student_architecture import DEFAULT_SPARSE_STUDENT, SparseStudentSpec


class IdentityDualStreamMixer(nn.Module):
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.mix_down = nn.Linear(2 * hidden_size, rank, bias=False)
        self.mix_up = nn.Linear(rank, 2 * hidden_size, bias=False)
        self.mix_up.weight = mx.zeros_like(self.mix_up.weight)
        self.output_gate = nn.Linear(hidden_size, 2, bias=True)

    def __call__(self, semantic: mx.array, social: mx.array) -> tuple[mx.array, mx.array]:
        if semantic.shape != social.shape:
            raise ValueError("Semantic and social streams must have the same shape.")
        hidden_size = semantic.shape[-1]
        delta = self.mix_up(nn.silu(self.mix_down(mx.concatenate((semantic, social), axis=-1))))
        semantic_delta, social_delta = mx.split(delta, 2, axis=-1)
        gates = mx.sigmoid(self.output_gate(semantic))
        return (
            semantic + gates[..., 0:1] * semantic_delta,
            social + gates[..., 1:2] * social_delta,
        )


class SharedListenerAdapter(nn.Module):
    def __init__(self, listener_width: int, hidden_size: int, rank: int):
        super().__init__()
        self.down = nn.Linear(listener_width, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)
        self.gate = nn.Linear(2 * hidden_size, 1, bias=False)

    def __call__(self, semantic: mx.array, listener_states: mx.array) -> mx.array:
        projected = self.up(nn.silu(self.down(listener_states)))
        if projected.shape != semantic.shape:
            raise ValueError("Listener states must be aligned to the token sequence.")
        gate = mx.sigmoid(self.gate(mx.concatenate((semantic, projected), axis=-1)))
        return gate * projected


class KiraSparseStudent(nn.Module):
    def __init__(
        self,
        args: TextModelArgs,
        *,
        ngram_indices: tuple[int, ...],
        residual_indices: tuple[int, ...],
        audio_indices: tuple[int, ...],
        ngram_buckets: int,
        ngram_width: int,
        listener_width: int,
        audio_bridge_rank: int,
        residual_rank: int,
        dense_prefix_layers: int = 0,
        emotion_dim: int = 8,
    ):
        super().__init__()
        if args.num_experts <= 0 or args.num_experts_per_tok <= 0:
            raise ValueError("KIRA sparse student requires token-routed MoE experts.")
        if not args.tie_word_embeddings:
            raise ValueError("The sparse parameter budget requires tied embeddings.")
        self.language = TextModel(args)
        self.decoder = self.language.model
        self.ngram_indices = ngram_indices
        self.residual_indices = residual_indices
        self.audio_indices = audio_indices
        self.dense_prefix_layers = dense_prefix_layers
        for index, layer in enumerate(self.decoder.layers):
            if index < dense_prefix_layers:
                layer.mlp = Qwen3NextMLP(args.hidden_size, args.moe_intermediate_size)
            else:
                layer.mlp = EmotionConditionedSparseFFN(
                    hidden_size=args.hidden_size,
                    expert_count=args.num_experts,
                    active_experts=args.num_experts_per_tok,
                    expert_intermediate=args.moe_intermediate_size,
                    shared_intermediate=args.shared_expert_intermediate_size,
                    emotion_dim=emotion_dim,
                )
        self.ngram_layers = [
            HashedNgramPLE(
                bucket_count=ngram_buckets,
                embedding_dim=ngram_width,
                hidden_size=args.hidden_size,
                ngram_sizes=(2, 3, 4),
                dilation=2,
                modality_salt=43_117 + index * 1_009,
            )
            for index in ngram_indices
        ]
        self.residual_mixers = [
            IdentityDualStreamMixer(args.hidden_size, residual_rank)
            for _ in residual_indices
        ]
        self.listener_adapter = SharedListenerAdapter(
            listener_width, args.hidden_size, audio_bridge_rank
        )

    @classmethod
    def from_spec(cls, spec: SparseStudentSpec = DEFAULT_SPARSE_STUDENT):
        spec.validate_budget()
        return cls(
            spec.mlx_text_args(),
            ngram_indices=(2, 6, 10, 14, 18, 22),
            residual_indices=(3, 7, 11, 15, 19, 23),
            audio_indices=(0, 3, 7, 11, 15, 19, 23),
            ngram_buckets=spec.ngram_buckets,
            ngram_width=spec.ngram_width,
            listener_width=spec.listener_width,
            audio_bridge_rank=spec.audio_bridge_rank,
            residual_rank=spec.gated_residual_rank,
            dense_prefix_layers=spec.dense_prefix_layers,
            emotion_dim=spec.emotion_feature_dim,
        )

    def load_sparse_layer(self, index: int, checkpoint: Path) -> None:
        """Insert an independently distilled FFN checkpoint at one layer."""
        if not 0 <= index < len(self.decoder.layers):
            raise IndexError(index)
        if index < self.dense_prefix_layers:
            raise ValueError("Dense prefix layers do not accept sparse expert checkpoints.")
        self.decoder.layers[index].mlp.load_weights(str(checkpoint), strict=True)

    def __call__(
        self,
        token_ids: mx.array,
        *,
        emotion_features: mx.array | None = None,
        listener_states: mx.array | None = None,
        cache: list | None = None,
        ngram_history: mx.array | None = None,
        logits_from: int = 0,
    ) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError("Token IDs must be [batch, time].")
        hidden = self.decoder.embed_tokens(token_ids)
        social = mx.zeros_like(hidden)
        audio = (
            self.listener_adapter(hidden, listener_states)
            if listener_states is not None else None
        )
        if cache is None:
            cache = [None] * len(self.decoder.layers)
        if len(cache) != len(self.decoder.layers):
            raise ValueError("One cache entry is required per decoder layer.")
        ple_ids = (
            mx.concatenate((ngram_history, token_ids), axis=1)
            if ngram_history is not None else token_ids
        )
        fa_mask = create_attention_mask(hidden, cache[self.decoder.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[self.decoder.ssm_idx])
        for index, (layer, layer_cache) in enumerate(zip(self.decoder.layers, cache)):
            if index in self.ngram_indices:
                ple = self.ngram_layers[self.ngram_indices.index(index)](ple_ids)
                hidden = hidden + ple[:, -hidden.shape[1]:]
            if audio is not None and index in self.audio_indices:
                social = social + audio
            attention = layer.linear_attn if layer.is_linear else layer.self_attn
            mask = ssm_mask if layer.is_linear else fa_mask
            attended = hidden + attention(layer.input_layernorm(hidden), mask, layer_cache)
            feed_forward_input = layer.post_attention_layernorm(attended)
            feed_forward = (
                layer.mlp(feed_forward_input)
                if index < self.dense_prefix_layers
                else layer.mlp(feed_forward_input, emotion_features)
            )
            hidden = attended + feed_forward
            if index in self.residual_indices:
                mixer = self.residual_mixers[self.residual_indices.index(index)]
                hidden, social = mixer(hidden, social)
        hidden = self.decoder.norm(hidden)
        return self.decoder.embed_tokens.as_linear(hidden[:, logits_from:])
