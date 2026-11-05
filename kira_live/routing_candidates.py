"""Isolated routing experiments; never selected by the Superapp manifest."""
from pathlib import Path
import json
import hashlib
import mlx.core as mx
import mlx.nn as nn
from .mlx_thinker import KiraMLXThinker


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_parameters(model, full=False):
    """Canonical full tree omits aliases of the backbone already under model."""
    from mlx.utils import tree_flatten
    return {k: v for k, v in tree_flatten(model.parameters())
            if (not k.startswith(('language_model.', 'decoder.')) if full
                else k.startswith(('routing_branches.', 'shared_conversation.')))}


def exact_top_two(logits):
    """Exactly two distinct selections even when all logits tie."""
    order = mx.argsort(logits, axis=-1)[..., -2:]
    mask = mx.sum(order[..., None] == mx.arange(logits.shape[-1]), axis=-2)
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    selected = probabilities * mask
    return selected / mx.maximum(selected.sum(axis=-1, keepdims=True), 1e-9), mask


class TokenResidualBranch(nn.Module):
    def __init__(self, width, rank, source, variant):
        super().__init__()
        self.variant = variant
        indices = (0, 1, 2, 3, 6, 7) if variant == 'token_routed' else (2, 3, 5, 6, 7, 9)
        self.router = nn.Linear(width, 6, bias=False)
        self.router.weight = mx.stack([source.router.weight[i] for i in indices])
        self.gate = nn.Linear(width, 1)
        self.gate.weight = mx.zeros_like(self.gate.weight)
        self.gate.bias = mx.array([-2.0])
        self.down = [nn.Linear(width, rank, bias=False) for _ in indices]
        self.up = [nn.Linear(rank, width, bias=False) for _ in indices]
        for local, old in enumerate(indices):
            basis = source.down[old].weight
            repeats = (rank + basis.shape[0] - 1) // basis.shape[0]
            self.down[local].weight = (mx.tile(basis, (repeats, 1))[:rank]
                                       + 0.001 * mx.random.normal((rank, width))).astype(mx.bfloat16)
            self.up[local].weight = mx.zeros((width, rank), dtype=mx.bfloat16)
        self._trace = None
        self._gate_threshold = 0.5

    def __call__(self, hidden):
        logits = self.router(hidden)
        selected, mask = exact_top_two(logits)
        gate_logits = self.gate(hidden)[..., 0]
        gate = mx.ones_like(gate_logits) if self.variant == 'token_routed' else mx.sigmoid(gate_logits)
        # Inference bypass is genuinely zero, not a permanently-on family.
        if not self.training and self.variant == 'shared_agentic':
            gate = mx.where(gate >= self._gate_threshold, gate, mx.zeros_like(gate))
        delta = mx.zeros_like(hidden)
        for index in range(6):
            value = self.up[index](nn.silu(self.down[index](hidden)))
            delta = delta + selected[..., index:index + 1].astype(value.dtype) * value
        self._trace = (logits, gate_logits, mask, gate)
        return hidden + gate[..., None].astype(hidden.dtype) * delta


class SharedConversationAdapter(nn.Module):
    def __init__(self, width, rank=64):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)

    def __call__(self, hidden):
        return hidden + self.up(nn.silu(self.down(hidden)))


class RoutingCandidateThinker(KiraMLXThinker):
    """Same base/audio/PLE/old experts; new branches in the final four layers."""
    def configure(self, variant, rank=4608):
        if variant not in ('token_routed', 'shared_agentic', 'hybrid_specialist', 'hybrid_specialist_gated') or rank <= 0:
            raise ValueError('Invalid experimental routing configuration')
        self.variant, self.branch_rank = variant, rank
        branch_variant = 'token_routed' if variant.startswith('hybrid_specialist') else variant
        self.routing_branches = [TokenResidualBranch(self.hidden_size, rank, layer, branch_variant)
                                 for layer in self.expert_layers[-4:]]
        self.shared_conversation = ([SharedConversationAdapter(self.hidden_size) for _ in range(4)]
                                    if variant in ('shared_agentic', 'hybrid_specialist', 'hybrid_specialist_gated') else [])
        self._routing_audit = None
        self._agentic_mode = False
        return self

    def set_execution_mode(self, agentic):
        """Explicit harness metadata; never infer tool permissions from this flag."""
        if type(agentic) is not bool:
            raise ValueError('Execution mode must be a boolean')
        self._agentic_mode = agentic

    def build_live_prefix(self, *args, **kwargs):
        # Set on EVERY turn so a Tree turn cannot leak its mode into later chat.
        self.set_execution_mode(bool(kwargs.get('tree_planning', False)))
        return super().build_live_prefix(*args, **kwargs)

    def _apply_expert(self, index, hidden, emotion_features):
        hidden, logits = super()._apply_expert(index, hidden, emotion_features)
        if index >= 12:
            local = index - 12
            gated = self.variant == 'hybrid_specialist_gated'
            if self.shared_conversation and (not gated or not self._agentic_mode):
                hidden = self.shared_conversation[local](hidden)
            if not gated or self._agentic_mode:
                hidden = self.routing_branches[local](hidden)
            else:
                # Genuine bypass: the incompatible A matrices are not run.
                self.routing_branches[local]._trace = None
            if self._routing_audit is not None:
                trace = self.routing_branches[local]._trace
                if trace is None:
                    gate = mx.zeros(hidden.shape[:-1])
                    mask = mx.zeros((*hidden.shape[:-1], 6), dtype=mx.int32)
                else:
                    _logits, _gate_logits, mask, gate = trace
                mx.eval(mask, gate)
                enabled = gate > 0
                self._routing_audit['tokens'] += enabled.size
                self._routing_audit['enabled'] += int(mx.sum(enabled).item())
                counts = mx.sum(mask * enabled[..., None], axis=(0, 1)).tolist()
                for i, count in enumerate(counts):
                    self._routing_audit['expert_tokens'][i] += int(count)
                phase = 'decode' if hidden.shape[1] == 1 else 'prefill'
                self._routing_audit[phase]['tokens'] += enabled.size
                self._routing_audit[phase]['enabled'] += int(mx.sum(enabled).item())
        return hidden, logits

    def start_audit(self):
        self._routing_audit = {'tokens': 0, 'enabled': 0, 'expert_tokens': [0] * 6,
                               'decode': {'tokens': 0, 'enabled': 0},
                               'prefill': {'tokens': 0, 'enabled': 0}}

    def finish_audit(self):
        result, self._routing_audit = self._routing_audit, None
        result['enabled_fraction'] = result['enabled'] / max(result['tokens'], 1)
        for phase in ('decode', 'prefill'):
            result[phase]['enabled_fraction'] = result[phase]['enabled'] / max(result[phase]['tokens'], 1)
        result['scope'] = 'All four layers, prompt-prefill and generated tokens, not assistant-only'
        return result


def load_candidate(directory, donor):
    directory = Path(directory)
    config = json.loads((directory / 'candidate_config.json').read_text())
    if config.get('loadable') is False:
        raise ValueError('This archived pilot is retained for reference, not runtime loading')
    if config.get('candidate_sha256') and sha256(directory / 'candidate.safetensors') != config['candidate_sha256']:
        raise ValueError('Candidate adapter checksum mismatch')
    base = Path(config['accepted_addons'])
    if sha256(base) != config['accepted_addons_sha256']:
        raise ValueError('Accepted baseline changed since candidate training')
    model = RoutingCandidateThinker.from_pretrained(donor)
    model.load_weights(str(base), strict=False)
    model.configure(config['variant'], config['rank'])
    threshold = float(config.get('gate_threshold', 0.5))
    if not 0 < threshold < 1:
        raise ValueError('Invalid calibrated action-gate threshold')
    for branch in model.routing_branches:
        branch._gate_threshold = threshold
    values = mx.load(str(directory / 'candidate.safetensors'))
    full = config.get('checkpoint_scope') == 'full_thinker'
    expected = {k: v.shape for k, v in checkpoint_parameters(model, full).items()}
    if {k: v.shape for k, v in values.items()} != expected:
        raise ValueError('Candidate adapter checkpoint is incomplete or incompatible')
    model.load_weights(list(values.items()), strict=False)
    model.eval()
    return model
