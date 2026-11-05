from __future__ import annotations

"""Experimental Live 1 sparse-student design distilled from Qwen3.5-2B.

This is a budgeted architecture, not a trained checkpoint. The dense 2B donor
is a teacher: its 2,048-wide weights cannot be loaded unchanged into this
768-wide student, and adding experts to the donor would not reduce its active
parameter count.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseStudentSpec:
    name: str = "Experimental Live 1"
    vocab_size: int = 248_320
    hidden_size: int = 768
    layers: int = 24
    dense_prefix_layers: int = 4
    full_attention_layers: int = 6
    head_dim: int = 256
    attention_heads: int = 3
    key_value_heads: int = 1
    linear_heads: int = 6
    linear_head_dim: int = 128
    experts_per_layer: int = 24
    active_experts_per_token: int = 2
    expert_intermediate_size: int = 2_016
    shared_expert_intermediate_size: int = 64
    ngram_injections: int = 6
    ngram_buckets: int = 65_536
    ngram_width: int = 32
    gated_residual_points: int = 6
    gated_residual_rank: int = 16
    listener_width: int = 1_024
    audio_bridge_rank: int = 256
    emotion_feature_dim: int = 8

    @property
    def expert_families(self) -> dict[str, int]:
        # 25% emotion, approximately 40% agentic, approximately 35% general.
        return {"emotion_social": 6, "agentic_action": 10, "general_language": 8}

    def parameter_budget(self) -> dict[str, int]:
        h = self.hidden_size
        if self.layers != 24 or self.full_attention_layers != 6:
            raise ValueError("This candidate preserves the 24-layer, every-fourth-attention layout.")
        if not 0 <= self.dense_prefix_layers < self.layers:
            raise ValueError("The sparse student must retain at least one sparse layer.")
        if self.attention_heads * self.head_dim != h:
            raise ValueError("Attention heads must fill the student hidden width.")
        if self.linear_heads * self.linear_head_dim != h:
            raise ValueError("DeltaNet key/value heads must fill the student hidden width.")
        if sum(self.expert_families.values()) != self.experts_per_layer:
            raise ValueError("Expert families must cover all routed experts.")
        if not 0 < self.active_experts_per_token <= self.experts_per_layer:
            raise ValueError("Invalid sparse top-k routing.")

        embeddings = self.vocab_size * h  # tied input/output weights
        delta_conv_width = 3 * h
        delta_per_layer = (
            delta_conv_width * h  # qkv input projection
            + h * h  # z projection
            + h * h  # output projection
            + 2 * self.linear_heads * h  # a/b projections
            + delta_conv_width * 4  # depthwise convolution, kernel four
            + 2 * self.linear_heads  # A_log and dt_bias
            + self.linear_head_dim  # gated RMS norm
        )
        full_per_layer = (
            2 * self.attention_heads * self.head_dim * h  # gated q projection
            + 2 * self.key_value_heads * self.head_dim * h  # k and v
            + h * h  # output projection
            + 2 * self.head_dim  # q/k norms
        )
        attention = (
            (self.layers - self.full_attention_layers) * delta_per_layer
            + self.full_attention_layers * full_per_layer
            + 2 * self.layers * h  # decoder layer norms
            + h  # final norm
        )
        expert_unit = 3 * h * self.expert_intermediate_size  # SwiGLU
        sparse_layers = self.layers - self.dense_prefix_layers
        dense_prefix_ffn = self.dense_prefix_layers * expert_unit
        experts = sparse_layers * self.experts_per_layer * expert_unit
        active_experts = sparse_layers * self.active_experts_per_token * expert_unit
        shared_experts = sparse_layers * 3 * h * self.shared_expert_intermediate_size
        routers = sparse_layers * (h * self.experts_per_layer + h)  # router + shared gate
        ngrams = self.ngram_injections * (
            self.ngram_buckets * self.ngram_width
            + self.ngram_width * h
            + 3  # order logits
            + 3 * self.ngram_width  # causal depthwise mixer
        )
        residual = self.gated_residual_points * (
            2 * (2 * h) * self.gated_residual_rank
            + h * 2 + 2
        )
        audio_bridge = (
            self.listener_width * self.audio_bridge_rank
            + self.audio_bridge_rank * h
            + 2 * h  # small fusion gate
        )
        # Live Listener prosody adds a bias to the *router*, not the user's
        # semantic token stream. This keeps emotion cues from rewriting words.
        emotion_router = sparse_layers * self.emotion_feature_dim * self.experts_per_layer
        components = {
            "tied_embeddings": embeddings,
            "attention_and_norms": attention,
            "dense_prefix_ffn": dense_prefix_ffn,
            "all_experts": experts,
            "active_experts": active_experts,
            "shared_experts": shared_experts,
            "routers": routers,
            "ngram_ple": ngrams,
            "gated_residual": residual,
            "shared_listener_bridge": audio_bridge,
            "emotion_router": emotion_router,
        }
        common = embeddings + attention + dense_prefix_ffn + shared_experts + routers + ngrams + residual + audio_bridge + emotion_router
        components["total_thinker"] = common + experts
        components["active_thinker"] = common + active_experts
        return components

    def validate_budget(self) -> dict[str, int]:
        budget = self.parameter_budget()
        if not 2_500_000_000 <= budget["total_thinker"] <= 3_000_000_000:
            raise ValueError("Sparse Thinker exceeds the requested 2.5–3B total range.")
        if budget["active_thinker"] > 500_000_000:
            raise ValueError("Sparse Thinker exceeds 0.5B active parameters.")
        return budget

    def mlx_text_args(self):
        """Build the actual MLX-LM sparse text topology; weights need distillation."""
        from mlx_lm.models.qwen3_5 import TextModelArgs

        self.validate_budget()
        return TextModelArgs(
            model_type="qwen3_5_text",
            hidden_size=self.hidden_size,
            intermediate_size=self.expert_intermediate_size,
            num_hidden_layers=self.layers,
            num_attention_heads=self.attention_heads,
            num_key_value_heads=self.key_value_heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            full_attention_interval=4,
            linear_num_key_heads=self.linear_heads,
            linear_num_value_heads=self.linear_heads,
            linear_key_head_dim=self.linear_head_dim,
            linear_value_head_dim=self.linear_head_dim,
            num_experts=self.experts_per_layer,
            num_experts_per_tok=self.active_experts_per_token,
            moe_intermediate_size=self.expert_intermediate_size,
            shared_expert_intermediate_size=self.shared_expert_intermediate_size,
            tie_word_embeddings=True,
            norm_topk_prob=True,
        )


DEFAULT_SPARSE_STUDENT = SparseStudentSpec()
