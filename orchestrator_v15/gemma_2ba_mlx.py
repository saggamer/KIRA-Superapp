"""MLX inference bridge for the 13-layer BF16 Gemma 2B-active expert bank.

The donor remains the locally cached 4-bit Gemma E4B conversion. Expert
matrices retain the checkpoint's BF16 precision; no expert quantization or
weight conversion is performed. This is inference only.
"""
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

from orchestrator_v15.gemma_2ba_budget import CHANNELS, Gemma2BABudget


class SparseBF16Experts(nn.Module):
    def __init__(self, state, channel='general'):
        super().__init__()
        self.gate = mx.stack([state[f'experts.{i}.gate_proj.weight'] for i in range(3)])
        self.up = mx.stack([state[f'experts.{i}.up_proj.weight'] for i in range(3)])
        self.down = mx.stack([state[f'experts.{i}.down_proj.weight'] for i in range(3)])
        self.router = state['router.weight'].astype(mx.float32)
        self.channel_bias = state['channel_bias'].astype(mx.float32)
        self.channel = channel

    def __call__(self, x):
        scores = x.astype(mx.float32) @ self.router.T + self.channel_bias[CHANNELS.index(self.channel)]
        indices = mx.argsort(scores, axis=-1)[..., -2:]
        weights = mx.softmax(mx.take_along_axis(scores, indices, axis=-1), axis=-1)
        expanded = mx.expand_dims(x, (-2, -3))
        gate = mx.gather_mm(expanded, self.gate.swapaxes(-1, -2), rhs_indices=indices)
        up = mx.gather_mm(expanded, self.up.swapaxes(-1, -2), rhs_indices=indices)
        activated = nn.gelu_approx(gate) * up
        outputs = mx.gather_mm(activated, self.down.swapaxes(-1, -2), rhs_indices=indices).squeeze(-2)
        # With delta_scale=1 in the CUDA model, shared + sum(w*(expert-shared))
        # equals sum(w*expert). This avoids computing a second, 4-bit shared FFN.
        return mx.sum(outputs * weights[..., None].astype(outputs.dtype), axis=-2)


def attach_checkpoint(model, checkpoint: Path, channel='general'):
    if channel not in CHANNELS:
        raise ValueError(f'unknown channel: {channel}')
    budget = Gemma2BABudget()
    budget.validate()
    checkpoint = Path(checkpoint)
    if len(model.layers) != 42:
        raise ValueError('expected 42 Gemma E4B decoder layers')
    modules = {}
    for layer in budget.routed_layers:
        path = checkpoint / f'layer_{layer:02d}.safetensors'
        if not path.is_file():
            raise FileNotFoundError(path)
        state = mx.load(str(path))
        module = SparseBF16Experts(state, channel=channel)
        mx.eval(module.gate, module.up, module.down, module.router, module.channel_bias)
        model.layers[layer].mlp = module
        modules[layer] = module
        del state
    return modules, budget.report()
