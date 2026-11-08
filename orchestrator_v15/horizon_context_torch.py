"""Untrained Horizon 1 v2 reference kernels, not a replacement image backbone.

Full-width shared attention uses modality-routed low-rank residual QKV paths.
Deployment requires a compatible pretrained projection and visual training.
"""
import torch
import math
from torch import nn
from torch.nn import functional as F

from .horizon_context import expand_layout, generalized_attention_mask

EXPERTS = ('understanding','generation','relations')
MODES = ('editing','creativity')


class JointContextResidual(nn.Module):
    def __init__(self, width, heads=4, rank=32, base_qkv=None):
        super().__init__()
        if width < 1 or heads < 1 or rank < 1 or width % heads:
            raise ValueError('Invalid shared context dimensions')
        self.width, self.heads, self.rank = width, heads, rank
        self.prototype_random_base = base_qkv is None
        self.base_qkv = base_qkv if base_qkv is not None else nn.Linear(width,3*width,bias=False)
        if not isinstance(self.base_qkv,nn.Linear) or self.base_qkv.in_features != width or self.base_qkv.out_features != 3*width:
            raise ValueError('A compatible full-width pretrained QKV projection is required')
        self.base_qkv.requires_grad_(False)
        self.qkv_down = nn.ModuleDict({mode:nn.ModuleDict({expert:nn.Linear(width,rank,bias=False)
                                                         for expert in EXPERTS}) for mode in MODES})
        self.qkv_up = nn.ModuleDict({mode:nn.ModuleDict({expert:nn.Linear(rank,3*width,bias=False)
                                                       for expert in EXPERTS}) for mode in MODES})
        self.down = nn.ModuleDict({mode:nn.ModuleDict({expert:nn.Linear(width,rank,bias=False)
                                                     for expert in EXPERTS}) for mode in MODES})
        self.up = nn.ModuleDict({mode:nn.ModuleDict({expert:nn.Linear(rank,width,bias=False)
                                                   for expert in EXPERTS}) for mode in MODES})
        self.gates = nn.ParameterDict({mode+'_'+expert:nn.Parameter(torch.tensor(-4.0))
                                       for mode in MODES for expert in EXPERTS})
        for paths in self.qkv_up.values():
            for projection in paths.values():
                nn.init.zeros_(projection.weight)
        for paths in self.up.values():
            for projection in paths.values():
                nn.init.zeros_(projection.weight)

    def forward(self, hidden, spans, *, mode=None):
        if mode is None:
            return hidden
        if mode not in MODES or hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError('Use a visual mode and batch/token/width context')
        tokens = expand_layout(spans)
        if hidden.shape[1] != len(tokens):
            raise ValueError('Context layout and hidden token count disagree')
        routes = ['understanding' if span.kind in {'text','semantic_vision'} else
                  'relations' if span.kind == 'relation' else 'generation' for span,_ in tokens]
        route_masks = {expert:torch.tensor([route == expert for route in routes],device=hidden.device)[None,:,None]
                       for expert in EXPERTS}
        native = self.base_qkv(hidden)
        residual = torch.zeros_like(native)
        for expert in EXPERTS:
            residual = residual + self.qkv_up[mode][expert](self.qkv_down[mode][expert](hidden)) * route_masks[expert]
        query,key,value = (native + residual).chunk(3,dim=-1)
        batch,length,_ = hidden.shape
        def heads(tensor):
            return tensor.reshape(batch,length,self.heads,self.width//self.heads).transpose(1,2)
        query,key,value = map(heads,(query,key,value))
        # Stabilize Q/K magnitudes without changing token addressability.
        query = F.rms_norm(query,(self.width//self.heads,))
        key = F.rms_norm(key,(self.width//self.heads,))
        mask = torch.tensor(generalized_attention_mask(spans),device=hidden.device,dtype=torch.bool)
        context = F.scaled_dot_product_attention(query,key,value,attn_mask=mask)
        context = context.transpose(1,2).reshape(batch,length,self.width)
        delta = torch.zeros_like(hidden)
        for expert in EXPERTS:
            branch = self.up[mode][expert](F.silu(self.down[mode][expert](context)))
            delta = delta + self.gates[mode+'_'+expert].sigmoid() * branch * route_masks[expert]
        return hidden + delta

    def trainable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class RelationBindingHead(nn.Module):
    """Role-sensitive edge scorer; its features need real visual alignment."""
    def __init__(self, width, relation_types, rank=32):
        super().__init__()
        if min(width,relation_types,rank) < 1:
            raise ValueError('Invalid relationship head dimensions')
        self.width = width
        self.subject = nn.Linear(width,rank,bias=False)
        self.target = nn.Linear(width,rank,bias=False)
        self.anchor = nn.Linear(width,rank,bias=False)
        self.predicate = nn.Embedding(relation_types,rank)
        self.score = nn.Sequential(nn.Linear(4*rank,rank),nn.SiLU(),nn.Linear(rank,1))

    def forward(self, subject, target, predicate, anchor=None):
        if subject.shape != target.shape or subject.shape[-1] != self.width:
            raise ValueError('Subject/target features must share the configured width')
        if predicate.shape != subject.shape[:-1] or predicate.dtype != torch.long:
            raise ValueError('Each edge needs an integer predicate ID')
        anchor = torch.zeros_like(subject) if anchor is None else anchor
        if anchor.shape != subject.shape:
            raise ValueError('Second anchor/owner features must match the edge shape')
        features = torch.cat((self.subject(subject),self.target(target),
                              self.predicate(predicate),self.anchor(anchor)),dim=-1)
        return self.score(features).squeeze(-1)


def relationship_ranking_loss(positive, negative, known, margin=0.2):
    if positive.shape != negative.shape or known.shape != positive.shape or known.dtype != torch.bool:
        raise ValueError('Pair scores and explicit known-label masks must align')
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Relationship margin must be finite and nonnegative')
    if known.any():
        if not torch.isfinite(positive[known]).all() or not torch.isfinite(negative[known]).all():
            raise ValueError('Known relationship scores must be finite')
        return F.softplus(negative[known]-positive[known]+margin).mean()
    return (positive.masked_fill(~known,0).sum()+negative.masked_fill(~known,0).sum()) * 0
