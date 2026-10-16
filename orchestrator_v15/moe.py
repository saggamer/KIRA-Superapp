"""Full-FFN sparse upcycling with frozen quantized experts and learned LoRA/router.

Only selected expert matrices are gathered. This is not compute-all-then-mask.
Routed deltas are centered on their frozen FFN output, preserving the original
FFN exactly at zero adapter initialization even with BF16 gathered kernels.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class GatherLinear(nn.Module):
    def __init__(self, base, experts, rank):
        super().__init__()
        if not isinstance(base, nn.QuantizedLinear) or "bias" in base:
            raise TypeError("pilot requires bias-free quantized FFN linears")
        self.bits, self.group_size = base.bits, base.group_size
        self.mode = base.mode
        self.weight = mx.stack([base.weight] * experts)
        self.scales = mx.stack([base.scales] * experts)
        self.biases = mx.stack([base.biases] * experts) if base.get("biases") is not None else None
        inp = base.weight.shape[-1] * 32 // self.bits
        out = base.weight.shape[0]
        self.lora_a = mx.random.uniform(-1 / math.sqrt(inp), 1 / math.sqrt(inp), (experts, rank, inp))
        self.lora_b = mx.zeros((experts, out, rank))
        self.scale = 1.0
        self.freeze()
        self.unfreeze(keys=["lora_a", "lora_b"], recurse=False)

    def base(self, x, indices):
        return mx.gather_qmm(x, self.weight, self.scales, self.biases, rhs_indices=indices,
                          transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode)

    def delta(self, x, indices):
        z = mx.gather_mm(x, self.lora_a.swapaxes(-1, -2), rhs_indices=indices)
        z = mx.gather_mm(z, self.lora_b.swapaxes(-1, -2), rhs_indices=indices)
        return (self.scale * z).astype(x.dtype)

    def __call__(self, x, indices):
        return self.base(x, indices) + self.delta(x, indices)


class SparseUpcycledFFN(nn.Module):
    def __init__(self, base, experts=8, rank=16, top_k=2, preserve_top2=False,
                 legacy_experts=0):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError("invalid expert selection")
        if legacy_experts and not (2 <= legacy_experts < experts and top_k > 2):
            raise ValueError('legacy expert range requires an expanded residual route')
        self.shared = base
        self.shared.freeze()
        hidden = base.gate_proj.weight.shape[1] * 32 // base.gate_proj.bits
        self.router = nn.Linear(hidden, experts, bias=False)
        self.router.weight = (0.01 * self.router.weight).astype(mx.float32)
        self.gate_proj = GatherLinear(base.gate_proj, experts, rank)
        self.up_proj = GatherLinear(base.up_proj, experts, rank)
        self.down_proj = GatherLinear(base.down_proj, experts, rank)
        self.top_k, self.experts = top_k, experts
        self.preserve_top2 = preserve_top2 and top_k > 2
        self.legacy_experts = legacy_experts
        if self.preserve_top2:
            # Keep the previous top-two path dominant while giving the
            # additional experts a small trainable contribution immediately.
            self.extra_gate = mx.array(0.1)
        self._enabled = True
        self._route_label = None
        self._aux = mx.array(0.0)
        # Optional inference-only calibration. A zero/absent bias preserves
        # existing checkpoints exactly and is never serialized as a weight.
        self._routing_bias = None
        self._trace_routes = False
        self._route_counts = None
        self._experimental_bias = None

    def __call__(self, x):
        if not self._enabled:
            return self.shared(x)
        scores = self.router(x.astype(mx.float32))
        selection_scores = scores
        if self._routing_bias is not None:
            selection_scores = selection_scores + self._routing_bias
        if self._experimental_bias is not None:
            selection_scores = selection_scores + mx.array(self._experimental_bias, dtype=mx.float32)
        if self.preserve_top2 and self.legacy_experts:
            primary = mx.argsort(selection_scores[..., :self.legacy_experts], axis=-1)[..., -2:]
            primary_mask = mx.any(primary[..., :, None] == mx.arange(self.experts), axis=-2)
            secondary_scores = mx.where(primary_mask, -1e9, selection_scores)
            secondary = mx.argsort(secondary_scores, axis=-1)[..., -(self.top_k-2):]
            indices = mx.stop_gradient(mx.concatenate([secondary, primary], axis=-1))
        else:
            indices = mx.stop_gradient(mx.argsort(selection_scores, axis=-1)[..., -self.top_k:])
        weight_scores = scores if self._experimental_bias is not None else selection_scores
        selected = mx.take_along_axis(weight_scores, indices, axis=-1)
        weights = mx.softmax(selected, axis=-1)
        expanded = mx.expand_dims(x, (-2, -3))
        gate0 = self.gate_proj.base(expanded, indices)
        up0 = self.up_proj.base(expanded, indices)
        gate = gate0 + self.gate_proj.delta(expanded, indices)
        up = up0 + self.up_proj.delta(expanded, indices)
        routed = self.down_proj(nn.gelu_approx(gate) * up, indices).squeeze(-2)
        reference = self.down_proj.base(nn.gelu_approx(gate0) * up0, indices).squeeze(-2)
        delta = routed-reference
        if self.preserve_top2:
            primary = mx.sum(delta[..., -2:, :] *
                             mx.softmax(selected[..., -2:], axis=-1)[..., None].astype(x.dtype), axis=-2)
            additional = mx.sum(delta[..., :-2, :] *
                                mx.softmax(selected[..., :-2], axis=-1)[..., None].astype(x.dtype), axis=-2)
            out = primary + mx.tanh(self.extra_gate).astype(x.dtype) * additional
        else:
            out = mx.sum(delta * weights[..., None].astype(x.dtype), axis=-2)
        probs = mx.softmax(scores, axis=-1)
        hits = (indices[..., None] == mx.arange(self.experts)).astype(mx.float32)
        if self._trace_routes:
            counts = mx.sum(hits, axis=tuple(range(hits.ndim - 1)))
            self._route_counts = counts if self._route_counts is None else self._route_counts + counts
        frequency = mx.mean(mx.mean(hits, axis=-2), axis=(0, 1))
        balance = self.experts * mx.sum(mx.mean(probs, axis=(0, 1)) * mx.stop_gradient(frequency))
        z = mx.mean(mx.logsumexp(scores, axis=-1) ** 2)
        supervised = mx.array(0.0)
        if self._route_label is not None:
            labels = mx.full(scores.shape[:-1], self._route_label, dtype=mx.int32)
            supervised = nn.losses.cross_entropy(scores, labels, reduction="mean")
        # Experimental route bias affects selection only. The selected expert
        # weights still come from unmodified model scores, as in auxiliary-loss-
        # free load balancing. Keep legacy behavior for all existing checkpoints.
        self._aux = (0.0001 * z if self._experimental_bias is not None
                     else 0.005 * balance + 0.0001 * z + 0.01 * supervised)
        return self.shared(x) + 0.5 * out


def attach(model, layers=(39, 40, 41), experts=8, rank=16, ceiling=10_000_000_000,
           top_k=2, preserve_top2=False, legacy_experts=0):
    if not 1 <= top_k <= experts:
        raise ValueError('active expert count out of range')
    model.freeze()
    extra = 0
    modules = []
    for index in layers:
        base = model.layers[index].mlp
        matrices = [base.gate_proj, base.up_proj, base.down_proj]
        extra += experts * sum(m.weight.shape[0] * m.weight.shape[1] * 32 // m.bits for m in matrices)
    # Conservative full upstream 8B count includes embeddings and modality encoders.
    controls = len(layers) * experts * model.layers[layers[0]].mlp.gate_proj.weight.shape[1] * 8
    adapter_count = len(layers) * experts * rank * sum(
        m.weight.shape[0] + m.weight.shape[1] * 32 // m.bits
        for m in [model.layers[layers[0]].mlp.gate_proj, model.layers[layers[0]].mlp.up_proj,
                  model.layers[layers[0]].mlp.down_proj])
    gate_count = len(layers) if preserve_top2 and top_k > 2 else 0
    bound = 8_000_000_000 + extra + adapter_count + controls + gate_count
    if bound > ceiling:
        raise ValueError(f"total logical parameter ceiling exceeded: {bound}")
    for index in layers:
        module = SparseUpcycledFFN(model.layers[index].mlp, experts, rank, top_k,
                                   preserve_top2=preserve_top2,legacy_experts=legacy_experts)
        model.layers[index].mlp = module
        modules.append(module)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if any(not (name.endswith(("lora_a", "lora_b", "router.weight", "extra_gate"))) for name in trainable):
        raise RuntimeError("training scope violation")
    return modules, {"full_upstream_parameter_bound": 8_000_000_000,
                     "new_full_ffn_expert_parameters": extra,
                     "trainable_parameters": sum(v.size for v in trainable.values()),
                     "total_logical_parameter_bound": bound, "ceiling": ceiling,
                     "layers": list(layers), "routed_experts_per_layer": experts,
                     "selected_per_token": top_k, "shared_per_layer": 1, "lora_rank": rank,
                     "active_new_expert_parameters_per_token": extra * top_k // experts,
                     "preserve_top2_residual": preserve_top2 and top_k > 2,
                     "legacy_experts_for_primary_route": legacy_experts,
                     "sparse_dispatch": "mlx.gather_qmm/gather_mm; selected matrices only",
                     "formulation": "shared FFN plus weighted full-FFN expert deltas centered on frozen expert outputs",
                     "reference_compute": "one extra gathered down-projection to preserve exact BF16 initialization parity"}
