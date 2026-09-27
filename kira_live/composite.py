from __future__ import annotations

"""Design blueprint for KIRA Live's three-weight runtime."""

from dataclasses import asdict, dataclass

from .architecture import Qwen4MicroBlueprint
from .config import KiraLiveConfig


@dataclass(frozen=True)
class WeightIsland:
    role: str
    model_id: str
    hidden_width: int
    parameter_count: int
    initially_frozen: bool = True


@dataclass(frozen=True)
class HiddenStateBridge:
    name: str
    source: str
    destination: str
    input_width: int
    output_width: int
    rank: int
    rate_hz: float
    carries: tuple[str, ...]


@dataclass(frozen=True)
class KiraLiveCompositePlan:
    """Target architecture, not a claim that all proposed bridges are deployed.

    The deployed runtime uses Listener semantic token remapping, Thinker
    generation, and public-token/prosody input to Talker. Its three islands are
    coordinated in one live session but are not yet jointly trained.
    """

    name: str
    listener: WeightIsland
    thinker: WeightIsland
    talker: WeightIsland
    listener_to_thinker: HiddenStateBridge
    thinker_to_talker: HiddenStateBridge
    ngram_orders: tuple[int, ...]
    ngram_domains: tuple[str, ...]
    expert_count: int
    expert_top_k: int
    thinker_audio_injection_layers: tuple[int, ...]
    thinker_moe_layers: tuple[int, ...]
    transcript_is_auxiliary: bool
    text_prompt_handoff: bool
    jointly_trainable_bridges: bool
    parameter_cap: int

    @classmethod
    def kira_live_1(cls) -> "KiraLiveCompositePlan":
        config = KiraLiveConfig()
        blueprint = Qwen4MicroBlueprint.kira_live_1()
        return cls(
            name="KIRA Live 1 composite",
            # The ASR acoustic encoder projects its 896-wide states into the
            # 1,024-wide decoder stream.  We bridge from that projected stream,
            # which is the checkpoint's actual cross-modal boundary.
            listener=WeightIsland("listener", config.listener.model_id, 1024, 782_426_112),
            thinker=WeightIsland("thinker", config.thinker.model_id, blueprint.hidden_size, 873_438_784),
            # Includes the Talker and the separate 170.6M speech codec.
            talker=WeightIsland("talker", config.talker.model_id, 2048, 2_087_233_793),
            listener_to_thinker=HiddenStateBridge(
                "acoustic_semantic_bridge",
                "listener",
                "thinker",
                1024,
                blueprint.hidden_size,
                64,
                blueprint.audio_tokens_hz,
                ("phonetics", "prosody", "timing", "emotion_uncertainty"),
            ),
            thinker_to_talker=HiddenStateBridge(
                "semantic_codec_bridge",
                "thinker",
                "talker",
                blueprint.hidden_size,
                2048,
                64,
                blueprint.talker_codec_hz,
                ("semantics", "speech_style", "interruptibility", "codec_prefix"),
            ),
            ngram_orders=blueprint.ple_ngram_sizes,
            ngram_domains=("listener_units", "thinker_tokens", "talker_codes"),
            expert_count=len(blueprint.expert_names),
            expert_top_k=2,
            thinker_audio_injection_layers=tuple(
                layer.index for layer in blueprint.layers if layer.audio_fusion
            ),
            thinker_moe_layers=tuple(
                layer.index for layer in blueprint.layers if layer.expert_moe
            ),
            transcript_is_auxiliary=True,
            text_prompt_handoff=False,
            jointly_trainable_bridges=True,
            parameter_cap=blueprint.deployment_parameter_cap,
        )

    def bridge_parameter_estimate(self) -> dict[str, int]:
        # Low-rank down/up projections, modality gates, style projection, and a
        # single shared tri-modal n-gram table. Biases are conservatively included.
        l2t = (
            self.listener_to_thinker.input_width * self.listener_to_thinker.rank
            + self.listener_to_thinker.rank * self.listener_to_thinker.output_width
            + 2 * self.listener_to_thinker.output_width**2
            + self.listener_to_thinker.output_width
        )
        t2t = (
            self.thinker_to_talker.input_width * self.thinker_to_talker.rank
            + self.thinker_to_talker.rank * self.thinker_to_talker.output_width
            + 8 * self.thinker_to_talker.output_width
            + self.thinker_to_talker.output_width
        )
        shared_ngram = 262_144 * 64 + len(self.ngram_domains) * 64 * self.thinker.hidden_width
        total = l2t + t2t + shared_ngram
        return {
            "listener_to_thinker": l2t,
            "thinker_to_talker": t2t,
            "shared_tri_modal_ngram": shared_ngram,
            "total": total,
        }

    def deployment_parameter_estimate(self) -> dict[str, int]:
        blueprint = Qwen4MicroBlueprint.kira_live_1()
        components = blueprint.total_deployment_parameter_estimate()
        bridge_total = self.bridge_parameter_estimate()["total"]
        total = components["total"] + bridge_total
        return {
            "three_weight_islands": (
                self.thinker.parameter_count
                + self.listener.parameter_count
                + self.talker.parameter_count
            ),
            "thinker_surgery": components["thinker_additions"],
            "composite_bridges": bridge_total,
            "emotion_and_vad_reserve": (
                components["emotion_encoder_reserve"] + components["vad_reserve"]
            ),
            "total": (
                self.thinker.parameter_count
                + self.listener.parameter_count
                + self.talker.parameter_count
                + components["thinker_additions"]
                + bridge_total
                + components["emotion_encoder_reserve"]
                + components["vad_reserve"]
            ),
            "headroom": self.parameter_cap - (
                self.thinker.parameter_count
                + self.listener.parameter_count
                + self.talker.parameter_count
                + components["thinker_additions"]
                + bridge_total
                + components["emotion_encoder_reserve"]
                + components["vad_reserve"]
            ),
        }

    def validate(self) -> None:
        if self.text_prompt_handoff:
            raise ValueError("KIRA Live weight islands must not communicate through prompts.")
        if not self.transcript_is_auxiliary:
            raise ValueError("An observable auxiliary transcript is required for debugging.")
        if self.ngram_domains != ("listener_units", "thinker_tokens", "talker_codes"):
            raise ValueError("All three weight islands must participate in shared n-gram routing.")
        if self.expert_top_k != 2 or self.expert_top_k > self.expert_count:
            raise ValueError("KIRA Live requires valid top-2 expert routing.")
        if self.deployment_parameter_estimate()["total"] >= self.parameter_cap:
            raise ValueError("Composite KIRA Live architecture exceeds its parameter cap.")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["bridge_parameters"] = self.bridge_parameter_estimate()
        payload["deployment_parameters"] = self.deployment_parameter_estimate()
        return payload
