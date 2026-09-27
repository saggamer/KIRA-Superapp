from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


EXPERTS = (
    "conversation",
    "emotion_prosody",
    "agent_planning",
    "tool_use",
    "episodic_memory",
    "workflow_state",
    "safety_permissions",
    "failure_recovery",
    "speech_style",
    "verification_completion",
)


@dataclass(frozen=True)
class LayerSpec:
    index: int
    token_mixer: str
    residual: str
    ple_ngram: bool
    audio_fusion: bool
    memory_fusion: bool
    expert_moe: bool
    expert_top_k: int = 0


@dataclass(frozen=True)
class Qwen4MicroBlueprint:
    """Trainable KIRA layer plan derived from Qwen4 concepts.

    This is deliberately smaller than Qwen4-Exp: it preserves the 0.8B
    Qwen3.5 backbone and adds QSA, compact PLE, two-stream GatedResidual, audio
    fusion, and residual low-rank experts within an approximately 1B Thinker.
    """

    base_model: str
    hidden_size: int
    layers: tuple[LayerSpec, ...]
    residual_streams: int
    gr_mixer_rank: int
    ple_bucket_count: int
    ple_embedding_dim: int
    ple_ngram_sizes: tuple[int, ...]
    ple_dilation: int
    qsa_block_size: int
    qsa_index_budget: int
    expert_names: tuple[str, ...]
    expert_rank: int
    audio_encoder_dim: int
    audio_tokens_hz: float
    talker_codec_hz: float
    memory_slots: int
    memory_chunk_tokens: int
    memory_adapter_rank: int
    native_context_tokens: int
    deployment_parameter_cap: int

    @classmethod
    def kira_live_1(cls) -> "Qwen4MicroBlueprint":
        # Qwen3.5-0.8B uses three linear-attention layers followed by one full
        # attention layer. KIRA replaces the six full-attention positions with
        # QSA and adds compact PLE immediately before each QSA layer.
        qsa_layers = {4, 8, 12, 16, 20, 24}
        ple_layers = {3, 7, 11, 15, 19, 23}
        audio_layers = {1, 4, 8, 12, 16, 20, 24}
        memory_layers = {4, 8, 12, 16, 20, 24}
        moe_layers = set(range(9, 25))
        layers = tuple(
            LayerSpec(
                index=index,
                token_mixer="qwen_sparse_attention" if index in qsa_layers else "gated_deltanet",
                residual="gated_residual_2stream",
                ple_ngram=index in ple_layers,
                audio_fusion=index in audio_layers,
                memory_fusion=index in memory_layers,
                expert_moe=index in moe_layers,
                expert_top_k=2 if index in moe_layers else 0,
            )
            for index in range(1, 25)
        )
        return cls(
            base_model="Qwen/Qwen3.5-0.8B",
            hidden_size=1024,
            layers=layers,
            residual_streams=2,
            gr_mixer_rank=64,
            ple_bucket_count=262_144,
            ple_embedding_dim=64,
            ple_ngram_sizes=(2, 3, 4),
            ple_dilation=2,
            qsa_block_size=64,
            qsa_index_budget=256,
            expert_names=EXPERTS,
            expert_rank=128,
            # Qwen3-ASR's 896-wide acoustic tower projects into the 1,024-wide
            # decoder stream used as KIRA's fusion boundary.
            audio_encoder_dim=1024,
            audio_tokens_hz=12.5,
            talker_codec_hz=12.0,
            memory_slots=256,
            memory_chunk_tokens=2_048,
            memory_adapter_rank=64,
            native_context_tokens=262_144,
            deployment_parameter_cap=4_000_000_000,
        )

    def validate(self) -> None:
        indices = [layer.index for layer in self.layers]
        if indices != list(range(1, len(self.layers) + 1)):
            raise ValueError("Layer indices must be contiguous and one-based.")
        if len(self.layers) != 24:
            raise ValueError("KIRA Live 1 must match the 24-layer Qwen3.5-0.8B backbone.")
        if not any(layer.token_mixer == "qwen_sparse_attention" for layer in self.layers):
            raise ValueError("At least one QSA layer is required.")
        if not any(layer.ple_ngram for layer in self.layers):
            raise ValueError("At least one PLE layer is required.")
        if tuple(sorted(set(self.ple_ngram_sizes))) != self.ple_ngram_sizes:
            raise ValueError("PLE n-gram orders must be unique and increasing.")
        if any(order < 2 for order in self.ple_ngram_sizes):
            raise ValueError("PLE n-gram orders must be at least two.")
        if any(layer.expert_top_k > len(self.expert_names) for layer in self.layers):
            raise ValueError("Expert top-k exceeds the expert count.")
        if self.total_deployment_parameter_estimate()["total"] >= self.deployment_parameter_cap:
            raise ValueError("KIRA Live exceeds the deployment parameter cap.")

    def added_parameter_estimate(self) -> dict[str, int]:
        ple_layers = sum(layer.ple_ngram for layer in self.layers)
        moe_layers = sum(layer.expert_moe for layer in self.layers)
        # Residual experts are two low-rank matrices per expert and layer.
        residual_experts = (
            moe_layers * len(self.expert_names) * 2 * self.hidden_size * self.expert_rank
        )
        ple = ple_layers * self.ple_bucket_count * self.ple_embedding_dim
        # Each GR input mixer and output gate uses a compact low-rank path.
        gated_residual = (
            len(self.layers)
            * self.residual_streams
            * 2
            * self.hidden_size
            * self.gr_mixer_rank
        )
        audio_projector = self.audio_encoder_dim * self.hidden_size + self.hidden_size**2
        routers = moe_layers * self.hidden_size * len(self.expert_names)
        memory_layers = sum(layer.memory_fusion for layer in self.layers)
        # Query, key, value-down/value-up projections plus a scalar gate.
        memory_fusion = memory_layers * (
            4 * self.hidden_size * self.memory_adapter_rank + self.hidden_size
        )
        total = (
            residual_experts
            + ple
            + gated_residual
            + audio_projector
            + routers
            + memory_fusion
        )
        return {
            "residual_experts": residual_experts,
            "ple": ple,
            "gated_residual": gated_residual,
            "audio_projector": audio_projector,
            "routers": routers,
            "memory_fusion": memory_fusion,
            "total": total,
        }

    def total_deployment_parameter_estimate(self) -> dict[str, int]:
        """Whole-stack design budget based on audited local checkpoint headers."""
        components = {
            "thinker_base": 873_438_784,
            "thinker_additions": self.added_parameter_estimate()["total"],
            "listener": 782_426_112,
            "talker": 1_916_676_352,
            "talker_codec": 170_557_441,
            "emotion_encoder_reserve": 50_000_000,
            "vad_reserve": 5_000_000,
        }
        components["total"] = sum(components.values())
        components["headroom"] = self.deployment_parameter_cap - components["total"]
        return components

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["estimated_added_parameters"] = self.added_parameter_estimate()
        return payload

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target
