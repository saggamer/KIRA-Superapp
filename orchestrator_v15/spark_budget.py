"""Exact logical parameter ceiling for Spark-X2.5-4B selective MoE upcycling."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SparkBudget:
    base_parameters: int = 4_112_079_360
    hidden_size: int = 2560
    intermediate_size: int = 10240
    selected_layers: tuple[int, ...] = (8, 17, 26, 35)
    added_experts_per_layer: int = 15
    ceiling: int = 9_000_000_000

    @property
    def added_expert_parameters(self) -> int:
        # Spark MLP has gate/up/down matrices and no bias.
        return len(self.selected_layers) * self.added_experts_per_layer * (
            3 * self.hidden_size * self.intermediate_size)

    @property
    def router_parameters(self) -> int:
        # Sixteen choices at each selected layer, including the resident FFN.
        choices = self.added_experts_per_layer + 1
        return len(self.selected_layers) * (self.hidden_size * choices + choices)

    @property
    def total_parameters(self) -> int:
        return self.base_parameters + self.added_expert_parameters + self.router_parameters

    def validate(self) -> None:
        if not self.selected_layers or len(set(self.selected_layers)) != len(self.selected_layers):
            raise ValueError('selected layers must be unique')
        if any(layer < 0 or layer >= 36 for layer in self.selected_layers):
            raise ValueError('Spark has 36 decoder layers')
        if self.total_parameters > self.ceiling:
            raise ValueError('MoE logical parameter ceiling exceeded')
