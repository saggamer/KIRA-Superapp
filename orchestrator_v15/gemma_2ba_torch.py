"""Sparse, channel-gated full FFN experts for Gemma E4B text decoder.

The shared Gemma FFN is retained.  Experts begin as exact copies of it, then
their *full* matrices can be trained one at a time.  The forward executes only
the selected two of three experts for each token.
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F

from orchestrator_v15.gemma_2ba_budget import (
    CHANNELS, Gemma2BABudget, Gemma2BAPreservedBudget,
)


CHANNEL_TARGETS = {
    'browser': (0, 1),
    'computer': (0, 1),
    'code': (1, 2),
    'knowledge': (0, 2),
    'reasoning': (1, 2),
    'general': (0, 2),
}


class ChannelGatedFullFFN(nn.Module):
    def __init__(self, shared: nn.Module, hidden_size: int, experts: int = 3,
                 top_k: int = 2, channels=CHANNELS, delta_scale: float = 1.0):
        super().__init__()
        if not 1 <= top_k < experts:
            raise ValueError('routing must be sparse')
        self.shared = shared
        self.shared.requires_grad_(False)
        self.experts = nn.ModuleList(copy.deepcopy(shared) for _ in range(experts))
        for expert in self.experts:
            expert.requires_grad_(False)
        device = next(shared.parameters()).device
        self.router = nn.Linear(hidden_size, experts, bias=False,
                                device=device, dtype=torch.float32)
        nn.init.zeros_(self.router.weight)
        self.channel_bias = nn.Parameter(torch.zeros(len(channels), experts,
                                                     device=device, dtype=torch.float32))
        self.router.requires_grad_(False)
        self.channel_bias.requires_grad_(False)
        self.channels = tuple(channels)
        self.top_k = top_k
        self.delta_scale = delta_scale
        self.active_channel = 'general'
        self.forced_expert = None
        self._routing_loss = None

    def set_channel(self, channel: str):
        if channel not in self.channels:
            raise ValueError(f'unknown routing channel: {channel}')
        self.active_channel = channel

    def train_expert(self, index: int):
        if not 0 <= index < len(self.experts):
            raise ValueError('expert index out of range')
        for i, expert in enumerate(self.experts):
            expert.requires_grad_(i == index)
        self.router.requires_grad_(False)
        self.channel_bias.requires_grad_(False)
        self.forced_expert = index

    def train_router(self):
        for expert in self.experts:
            expert.requires_grad_(False)
        self.router.requires_grad_(True)
        self.channel_bias.requires_grad_(True)
        self.forced_expert = None

    def freeze_all(self):
        for expert in self.experts:
            expert.requires_grad_(False)
        self.router.requires_grad_(False)
        self.channel_bias.requires_grad_(False)
        self.forced_expert = None

    def routing_loss(self):
        return self._routing_loss

    def forward(self, x):
        if self.forced_expert is not None:
            self._routing_loss = None
            return self.experts[self.forced_expert](x)
        shared = self.shared(x)
        flat = x.reshape(-1, x.shape[-1])
        shared_flat = shared.reshape(-1, shared.shape[-1])
        channel_id = self.channels.index(self.active_channel)
        scores = self.router(flat.float()) + self.channel_bias[channel_id]
        selected_scores, selected_indices = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(selected_scores, dim=-1)
        delta = torch.zeros_like(shared_flat)
        for expert_id, expert in enumerate(self.experts):
            positions, slots = torch.where(selected_indices == expert_id)
            if positions.numel() == 0:
                continue
            outputs = expert(flat.index_select(0, positions))
            weighted = (outputs - shared_flat.index_select(0, positions)) * \
                weights[positions, slots, None].to(outputs.dtype)
            delta.index_add_(0, positions, weighted)
        if self.training and self.router.weight.requires_grad:
            target = torch.zeros_like(scores)
            for expert_id in CHANNEL_TARGETS[self.active_channel]:
                target[:, expert_id] = 1.0
            supervision = F.binary_cross_entropy_with_logits(scores, target)
            balance = torch.mean(F.softmax(scores, dim=-1), dim=0)
            balance_penalty = len(self.experts) * torch.sum(balance.square()) - 1.0
            self._routing_loss = supervision + 0.01 * balance_penalty
        else:
            self._routing_loss = None
        return shared + self.delta_scale * delta.reshape_as(shared)


def attach_gemma_2ba(model, budget: Gemma2BABudget = Gemma2BABudget()):
    """Attach 13 sparse layers to a Gemma4ForConditionalGeneration text model."""
    budget.validate()
    # Transformers Gemma4ForConditionalGeneration owns Gemma4Model at `.model`;
    # its language_model is already Gemma4TextModel, with decoder layers directly.
    decoder = model.model.language_model.layers
    if len(decoder) != 42:
        raise ValueError('expected 42 Gemma E4B text decoder layers')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = {}
    for layer_idx in budget.routed_layers:
        shared = decoder[layer_idx].mlp
        modules[layer_idx] = ChannelGatedFullFFN(
            shared, budget.hidden_size, budget.experts_per_layer,
            budget.top_k, budget.channels)
        decoder[layer_idx].mlp = modules[layer_idx]
    return modules


def attach_gemma_2ba_preserved(model,
                               budget: Gemma2BAPreservedBudget = Gemma2BAPreservedBudget()):
    """Attach sparse full experts, retaining six top-layer transfer slots."""
    budget.validate()
    decoder = model.model.language_model.layers
    if len(decoder) != 42:
        raise ValueError('expected 42 Gemma E4B text decoder layers')
    model.requires_grad_(False)
    modules = {}
    for layer_idx in budget.routed_layers:
        shared = decoder[layer_idx].mlp
        modules[layer_idx] = ChannelGatedFullFFN(
            shared, budget.hidden_size, budget.experts_at_layer(layer_idx),
            budget.top_k, budget.channels,
            delta_scale=0.5 if layer_idx in budget.preserve_layers else 1.0)
        decoder[layer_idx].mlp = modules[layer_idx]
    return modules
