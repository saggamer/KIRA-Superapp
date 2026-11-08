"""Trainable mode-gated LLM-context bridge for the public visual donor.

Inspired by Kira Live's zero-initialized residual paths. This is a conditioning
bridge into a joint text/image diffusion transformer, not BAGEL's full MoT.
"""
from functools import wraps
from pathlib import Path
import json

import torch
from torch import nn
from torch.nn import functional as F


class ModeGatedSemanticBridge(nn.Module):
    def __init__(self, native_width, source_width=None, rank=32, modes=('editing', 'creativity')):
        super().__init__()
        self.native_width = int(native_width)
        self.source_width = int(source_width or native_width)
        self.rank = int(rank)
        self.modes = tuple(modes)
        if self.rank < 1 or not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError('Invalid bridge rank or modes')
        self.query = nn.Linear(self.native_width, self.rank, bias=False)
        self.key = nn.Linear(self.source_width, self.rank, bias=False)
        self.value = nn.Linear(self.source_width, self.rank, bias=False)
        self.output = nn.ModuleDict({mode: nn.Linear(self.rank, self.native_width, bias=False)
                                     for mode in self.modes})
        self.gates = nn.ParameterDict({mode: nn.Parameter(torch.tensor(-4.0)) for mode in self.modes})
        for layer in self.output.values():
            nn.init.zeros_(layer.weight)

    def forward(self, native, semantic=None, *, mode=None, source_mask=None):
        if mode is None:
            return native
        if mode not in self.modes:
            raise ValueError('Unsupported visual bridge mode')
        semantic = native if semantic is None else semantic
        if native.ndim != 3 or semantic.ndim != 3 or native.shape[0] != semantic.shape[0]:
            raise ValueError('Conditioning must be batch/token/feature tensors with matching batches')
        if native.shape[-1] != self.native_width or semantic.shape[-1] != self.source_width:
            raise ValueError('Conditioning feature widths do not match the bridge')
        dtype = self.query.weight.dtype
        query = self.query(native.to(dtype)).unsqueeze(1)
        key = self.key(semantic.to(dtype)).unsqueeze(1)
        value = self.value(semantic.to(dtype)).unsqueeze(1)
        mask = None
        if source_mask is not None:
            if source_mask.shape != semantic.shape[:2] or not source_mask.bool().any(dim=1).all():
                raise ValueError('Each source must contain at least one unmasked semantic token')
            mask = source_mask.bool()[:, None, None, :]
        mixed = F.scaled_dot_product_attention(query, key, value, attn_mask=mask).squeeze(1)
        delta = self.output[mode](mixed)
        return native + (self.gates[mode].sigmoid() * delta).to(native.dtype)


def install_fusion_bridge(transformer, mode, rank=32):
    if hasattr(transformer, 'horizon_bridge'):
        raise ValueError('A Horizon bridge is already installed')
    if mode not in {'editing', 'creativity'}:
        raise ValueError('Use an explicit visual training mode')
    width = transformer.config.joint_attention_dim
    bridge = ModeGatedSemanticBridge(width, rank=rank)
    device = next(transformer.parameters()).device
    transformer.horizon_bridge = bridge.to(device=device)
    transformer.horizon_mode = mode
    transformer.horizon_bridge_gradient_updates = 0
    def record_gradient(gradient):
        if gradient.detach().float().norm().item() > 0:
            transformer.horizon_bridge_gradient_updates += 1
        return gradient
    bridge.output[mode].weight.register_hook(record_gradient)
    original = transformer.forward

    @wraps(original)
    def bridged_forward(*args, **kwargs):
        if 'encoder_hidden_states' not in kwargs:
            raise ValueError('Horizon visual calls must provide encoder_hidden_states explicitly')
        semantic = kwargs.pop('horizon_semantic_tokens', None)
        kwargs['encoder_hidden_states'] = transformer.horizon_bridge(
            kwargs['encoder_hidden_states'], semantic, mode=transformer.horizon_mode)
        return original(*args, **kwargs)

    transformer.forward = bridged_forward
    return bridge


def save_fusion_bridge(transformer, directory):
    from safetensors.torch import save_file
    bridge = transformer.horizon_bridge
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({key:value.detach().cpu().contiguous() for key,value in bridge.state_dict().items()},
              str(directory / 'horizon_bridge.safetensors'))
    config = {'native_width':bridge.native_width, 'source_width':bridge.source_width,
              'rank':bridge.rank, 'modes':bridge.modes, 'trained_mode':transformer.horizon_mode,
              'scope':'Trainable LLM-context conditioning bridge; not a fully shared BAGEL backbone'}
    (directory / 'horizon_bridge.json').write_text(json.dumps(config,indent=2)+'\n')
    evidence = {'backward_calls_with_nonzero_output_gradient':getattr(transformer,'horizon_bridge_gradient_updates',0),
                'active_output_weight_l2':float(bridge.output[transformer.horizon_mode].weight.detach().float().norm()),
                'bridge_parameters':sum(parameter.numel() for parameter in bridge.parameters()),
                'active_mode':transformer.horizon_mode}
    (directory / 'horizon_bridge_training.json').write_text(json.dumps(evidence,indent=2)+'\n')


def load_fusion_bridge(transformer, directory, mode):
    from safetensors.torch import load_file
    directory = Path(directory)
    config = json.loads((directory / 'horizon_bridge.json').read_text())
    if mode != config['trained_mode']:
        raise ValueError('Requested mode does not match the trained bridge')
    bridge = install_fusion_bridge(transformer, mode, rank=config['rank'])
    bridge.load_state_dict(load_file(str(directory / 'horizon_bridge.safetensors')), strict=True)
    return bridge
