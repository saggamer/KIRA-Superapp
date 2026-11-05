from __future__ import annotations

"""CUDA-trainable Experimental Live 1 graph; weights still need distillation.

The backbone uses Transformers' Qwen3.5 MoE DeltaNet/attention implementation.
KIRA adds causal hashed n-grams, a separate social/audio stream, and an
acoustic bias on token-level expert routing. This is a training graph, not a
checkpoint or a drop-in replacement for the existing MLX live runtime.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP

from .sparse_student_architecture import DEFAULT_SPARSE_STUDENT, SparseStudentSpec


class TorchHashedNgramPLE(nn.Module):
    def __init__(self, buckets: int, width: int, hidden: int, salt: int):
        super().__init__()
        self.buckets = buckets
        self.salt = salt
        self.table = nn.Embedding(buckets, width)
        self.order_logits = nn.Parameter(torch.zeros(3))
        self.mixer = nn.Conv1d(width, width, 3, dilation=2, groups=width, bias=False)
        self.project = nn.Linear(width, hidden, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("Token IDs must be [batch, time].")
        batch, length = token_ids.shape
        orders = []
        for n in (2, 3, 4):
            padded = F.pad(token_ids.long(), (n - 1, 0))
            hashed = torch.full((batch, length), self.salt, dtype=torch.long, device=token_ids.device)
            for offset in range(n):
                hashed = torch.remainder(hashed * 131_071 + padded[:, offset:offset + length], self.buckets)
            orders.append(self.table(hashed))
        mixed = sum(weight * value for weight, value in zip(self.order_logits.softmax(0), orders))
        causal = self.mixer(F.pad(mixed.transpose(1, 2), (4, 0))).transpose(1, 2)
        return self.project(causal)


class TorchEmotionRouter(nn.Module):
    def __init__(self, hidden: int, experts: int, top_k: int, emotion_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(experts, hidden))
        self.emotion_bias = nn.Linear(emotion_dim, experts, bias=False)
        nn.init.zeros_(self.emotion_bias.weight)
        self.top_k = top_k
        self.emotion_features: torch.Tensor | None = None
        self.batch_size = 0
        self.last_indices: torch.Tensor | None = None

    def set_context(self, features: torch.Tensor | None, batch_size: int) -> None:
        self.emotion_features = features
        self.batch_size = batch_size

    def forward(self, hidden_states: torch.Tensor):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = F.linear(flat, self.weight)
        if self.emotion_features is not None:
            if self.batch_size <= 0 or flat.shape[0] % self.batch_size:
                raise ValueError("Emotion features cannot align with routed tokens.")
            time = flat.shape[0] // self.batch_size
            bias = self.emotion_bias(self.emotion_features.to(flat.dtype))
            logits = logits + bias.repeat_interleave(time, dim=0)
        probabilities = logits.float().softmax(-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(logits.dtype)
        self.last_indices = indices
        return logits, weights, indices


class TorchDualStreamMixer(nn.Module):
    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.down = nn.Linear(2 * hidden, rank, bias=False)
        self.up = nn.Linear(rank, 2 * hidden, bias=False)
        nn.init.zeros_(self.up.weight)
        self.gate = nn.Linear(hidden, 2)

    def forward(self, semantic: torch.Tensor, social: torch.Tensor):
        delta = self.up(F.silu(self.down(torch.cat((semantic, social), dim=-1))))
        semantic_delta, social_delta = delta.chunk(2, dim=-1)
        gates = self.gate(semantic).sigmoid()
        return semantic + gates[..., :1] * semantic_delta, social + gates[..., 1:] * social_delta


class TorchListenerAdapter(nn.Module):
    def __init__(self, listener_width: int, hidden: int, rank: int):
        super().__init__()
        self.down = nn.Linear(listener_width, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.up.weight)
        self.gate = nn.Linear(2 * hidden, 1, bias=False)

    def forward(self, semantic: torch.Tensor, listener: torch.Tensor):
        projected = self.up(F.silu(self.down(listener)))
        if projected.shape != semantic.shape:
            raise ValueError("Listener states must align with the full token sequence.")
        return self.gate(torch.cat((semantic, projected), dim=-1)).sigmoid() * projected


@dataclass(frozen=True)
class TorchExperimentalLayout:
    ngram_indices: tuple[int, ...] = (2, 6, 10, 14, 18, 22)
    residual_indices: tuple[int, ...] = (3, 7, 11, 15, 19, 23)
    audio_indices: tuple[int, ...] = (0, 3, 7, 11, 15, 19, 23)


class ExperimentalLive1Torch(nn.Module):
    """Full training graph with hooks at KIRA's six insertion points.

    Only full-sequence training is supported here. Streaming cache and MLX
    checkpoint parity must be verified before a trained model can enter the app.
    """

    def __init__(self, config: Qwen3_5MoeTextConfig, *, dense_prefix: int, ngram_buckets: int,
                 ngram_width: int, listener_width: int, audio_rank: int, residual_rank: int,
                 emotion_dim: int, layout: TorchExperimentalLayout = TorchExperimentalLayout()):
        super().__init__()
        self.language = Qwen3_5MoeForCausalLM(config)
        self.layout = layout
        self.dense_prefix = dense_prefix
        for index, layer in enumerate(self.language.model.layers):
            if index < dense_prefix:
                layer.mlp = Qwen3_5MoeMLP(config, intermediate_size=config.moe_intermediate_size)
            else:
                layer.mlp.gate = TorchEmotionRouter(config.hidden_size, config.num_experts,
                                                    config.num_experts_per_tok, emotion_dim)
        self.ngrams = nn.ModuleDict({str(i): TorchHashedNgramPLE(
            ngram_buckets, ngram_width, config.hidden_size, 43_117 + i * 1_009
        ) for i in layout.ngram_indices if i < config.num_hidden_layers})
        self.mixers = nn.ModuleDict({str(i): TorchDualStreamMixer(config.hidden_size, residual_rank)
                                     for i in layout.residual_indices if i < config.num_hidden_layers})
        self.listener_adapter = TorchListenerAdapter(listener_width, config.hidden_size, audio_rank)
        self._token_ids: torch.Tensor | None = None
        self._social: torch.Tensor | None = None
        self._audio: torch.Tensor | None = None
        for index, layer in enumerate(self.language.model.layers):
            layer.register_forward_pre_hook(self._make_pre_hook(index), with_kwargs=True)
            layer.register_forward_hook(self._make_post_hook(index), with_kwargs=True)

    @classmethod
    def from_spec(cls, spec: SparseStudentSpec = DEFAULT_SPARSE_STUDENT):
        spec.validate_budget()
        layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                       for i in range(spec.layers)]
        config = Qwen3_5MoeTextConfig(
            vocab_size=spec.vocab_size, hidden_size=spec.hidden_size,
            num_hidden_layers=spec.layers, num_attention_heads=spec.attention_heads,
            num_key_value_heads=spec.key_value_heads, head_dim=spec.head_dim,
            linear_num_key_heads=spec.linear_heads, linear_num_value_heads=spec.linear_heads,
            linear_key_head_dim=spec.linear_head_dim, linear_value_head_dim=spec.linear_head_dim,
            moe_intermediate_size=spec.expert_intermediate_size,
            shared_expert_intermediate_size=spec.shared_expert_intermediate_size,
            num_experts=spec.experts_per_layer, num_experts_per_tok=spec.active_experts_per_token,
            layer_types=layer_types, tie_word_embeddings=True,
            max_position_embeddings=262_144,
            rope_parameters={
                "rope_type": "default", "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True, "mrope_section": [11, 11, 10],
            },
        )
        return cls(config, dense_prefix=spec.dense_prefix_layers, ngram_buckets=spec.ngram_buckets,
                   ngram_width=spec.ngram_width, listener_width=spec.listener_width,
                   audio_rank=spec.audio_bridge_rank, residual_rank=spec.gated_residual_rank,
                   emotion_dim=spec.emotion_feature_dim)

    def _make_pre_hook(self, index: int):
        def hook(_module, args, kwargs):
            hidden = args[0]
            if str(index) in self.ngrams:
                if self._token_ids is None:
                    raise RuntimeError("N-gram hook requires token IDs.")
                hidden = hidden + self.ngrams[str(index)](self._token_ids)
            if self._audio is not None and index in self.layout.audio_indices:
                self._social = self._social + self._audio
            return (hidden, *args[1:]), kwargs
        return hook

    def _make_post_hook(self, index: int):
        def hook(_module, _args, _kwargs, output):
            if str(index) in self.mixers:
                output, self._social = self.mixers[str(index)](output, self._social)
            return output
        return hook

    def forward(self, token_ids: torch.Tensor, *, emotion_features: torch.Tensor | None = None,
                listener_states: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        if token_ids.ndim != 2:
            raise ValueError("Token IDs must be [batch, time].")
        if emotion_features is not None and emotion_features.shape != (token_ids.shape[0], 8):
            raise ValueError("Emotion features must be [batch, 8].")
        if listener_states is not None and listener_states.shape[:2] != token_ids.shape:
            raise ValueError("Listener states must align with token IDs.")
        embeddings = self.language.model.embed_tokens(token_ids)
        self._token_ids = token_ids
        self._social = torch.zeros_like(embeddings)
        self._audio = self.listener_adapter(embeddings, listener_states) if listener_states is not None else None
        for layer in self.language.model.layers[self.dense_prefix:]:
            layer.mlp.gate.set_context(emotion_features, token_ids.shape[0])
        try:
            return self.language(inputs_embeds=embeddings, labels=labels, use_cache=False)
        finally:
            self._token_ids = None
            self._social = None
            self._audio = None
            for layer in self.language.model.layers[self.dense_prefix:]:
                layer.mlp.gate.set_context(None, 0)
