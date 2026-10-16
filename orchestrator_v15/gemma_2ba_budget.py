"""Conservative parameter accounting for Gemma E4B + 2B-active routed FFNs.

The upstream model card reports 8B total parameters.  This module counts
new FFNs at their unquantized logical size, even when stored more compactly.
"""
from dataclasses import asdict, dataclass


CHANNELS = ("browser", "computer", "code", "knowledge", "reasoning", "general")


@dataclass(frozen=True)
class Gemma2BABudget:
    donor_total_bound: int = 8_000_000_000
    hidden_size: int = 2_560
    intermediate_size: int = 10_240
    routed_layers: tuple[int, ...] = tuple(range(29, 42))
    experts_per_layer: int = 3
    top_k: int = 2
    channels: tuple[str, ...] = CHANNELS
    ceiling: int = 12_000_000_000

    @property
    def parameters_per_expert(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def expert_parameters(self) -> int:
        return len(self.routed_layers) * self.experts_per_layer * self.parameters_per_expert

    @property
    def active_expert_parameters_per_token(self) -> int:
        return len(self.routed_layers) * self.top_k * self.parameters_per_expert

    @property
    def router_parameters(self) -> int:
        return len(self.routed_layers) * (
            self.hidden_size * self.experts_per_layer
            + len(self.channels) * self.experts_per_layer
        )

    @property
    def total_logical_bound(self) -> int:
        return self.donor_total_bound + self.expert_parameters + self.router_parameters

    def validate(self) -> None:
        if not self.routed_layers or min(self.routed_layers) < 0 or max(self.routed_layers) >= 42:
            raise ValueError("routed layer index outside Gemma E4B text decoder")
        if len(set(self.routed_layers)) != len(self.routed_layers):
            raise ValueError("duplicate routed layer")
        if not 1 <= self.top_k < self.experts_per_layer:
            raise ValueError("top-k must be sparse")
        if len(set(self.channels)) != len(self.channels) or not self.channels:
            raise ValueError("channels must be nonempty and unique")
        if self.active_expert_parameters_per_token < 2_000_000_000:
            raise ValueError("routed expert path is below 2B active parameters")
        if self.total_logical_bound > self.ceiling:
            raise ValueError("total logical parameter ceiling exceeded")

    def report(self) -> dict:
        self.validate()
        return {**asdict(self),
                "parameters_per_expert": self.parameters_per_expert,
                "expert_parameters": self.expert_parameters,
                "active_expert_parameters_per_token": self.active_expert_parameters_per_token,
                "router_parameters": self.router_parameters,
                "total_logical_bound": self.total_logical_bound,
                "note": "2B active refers to added routed experts, not total active donor compute"}


@dataclass(frozen=True)
class Gemma2BAPreservedBudget:
    """Keep more of the successful old top-layer expert bank under 12B."""
    donor_total_bound: int = 8_000_000_000
    hidden_size: int = 2_560
    intermediate_size: int = 10_240
    routed_layers: tuple[int, ...] = tuple(range(29, 42))
    preserve_layers: tuple[int, ...] = (39, 40, 41)
    lower_experts: int = 3
    preserved_experts: int = 6
    top_k: int = 2
    channels: tuple[str, ...] = CHANNELS
    ceiling: int = 12_000_000_000

    def experts_at_layer(self, layer: int) -> int:
        return self.preserved_experts if layer in self.preserve_layers else self.lower_experts

    @property
    def parameters_per_expert(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def expert_parameters(self) -> int:
        return sum(self.experts_at_layer(layer) for layer in self.routed_layers) * self.parameters_per_expert

    @property
    def active_expert_parameters_per_token(self) -> int:
        return len(self.routed_layers) * self.top_k * self.parameters_per_expert

    @property
    def router_parameters(self) -> int:
        return sum(self.experts_at_layer(layer) *
                   (self.hidden_size + len(self.channels)) for layer in self.routed_layers)

    @property
    def total_logical_bound(self) -> int:
        return self.donor_total_bound + self.expert_parameters + self.router_parameters

    def validate(self) -> None:
        if tuple(sorted(self.routed_layers)) != tuple(range(29, 42)):
            raise ValueError('preserved budget requires text layers 29–41')
        if not set(self.preserve_layers).issubset(self.routed_layers):
            raise ValueError('preserved layer is not routed')
        if not 1 <= self.top_k < min(self.lower_experts, self.preserved_experts):
            raise ValueError('routing must be sparse')
        if self.active_expert_parameters_per_token < 2_000_000_000:
            raise ValueError('active expert path below 2B')
        if self.total_logical_bound > self.ceiling:
            raise ValueError('logical budget exceeded')

    def report(self) -> dict:
        self.validate()
        return {**asdict(self), 'expert_parameters': self.expert_parameters,
                'active_expert_parameters_per_token': self.active_expert_parameters_per_token,
                'router_parameters': self.router_parameters,
                'total_logical_bound': self.total_logical_bound}
