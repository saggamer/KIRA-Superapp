"""One trainable transformer for language tokens and image-flow prediction.

Research core, randomly initialized: does not merge existing Qwen/FLUX weights.
Inputs are token IDs, semantic vision features, and VAE latent patches. Encoders
and pixel decoding are deliberately outside the shared transformer.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math
import copy

import torch
from torch import nn
from torch.nn import functional as F
from .horizon_context import ContextSpan, expand_layout

KINDS = ('text', 'semantic_vision', 'clean_latent', 'noisy_latent', 'relation')


@dataclass(frozen=True)
class UnifiedConfig:
    vocab_size: int
    width: int = 256
    heads: int = 8
    layers: int = 4
    latent_width: int = 128
    vision_width: int = 256
    max_tokens: int = 2048
    experts: int = 1
    attention_experts: int = 0
    qk_norm: int = 0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in ('attention_experts', 'qk_norm'):
                if type(value) is not int or value not in (0, 1):
                    raise ValueError('Architecture switches must be integer 0 or 1')
                continue
            if type(value) is not int or value < 1:
                raise ValueError('All unified dimensions must be positive integers')
        if self.width % self.heads or self.width < 4 or self.max_tokens > 2048:
            raise ValueError('Invalid head width or dense-reference token limit')
        if self.experts not in (1, 3):
            raise ValueError('Use a dense core or three modality-routed experts')
        if self.attention_experts and self.experts != 3:
            raise ValueError('Attention experts require three routed experts')


def unified_attention_mask(spans, device='cpu', max_tokens=2048):
    """Tensor implementation of the reference generalized causal mask."""
    tokens = expand_layout(spans, max_tokens)
    sample = torch.tensor([s.sample for s,_ in tokens], device=device)
    block = torch.tensor([s.block for s,_ in tokens], device=device)
    position = torch.tensor([p for _,p in tokens], device=device)
    kind = torch.tensor([KINDS.index(s.kind) for s,_ in tokens], device=device)
    identities = {s.image_id:i for i,(s,_) in enumerate(tokens) if s.image_id is not None}
    image = torch.tensor([identities.get(s.image_id,-1) for s,_ in tokens], device=device)
    same_block = block[:,None] == block[None,:]
    allowed = (sample[:,None] == sample[None,:]) & (
        (block[None,:] < block[:,None]) | (same_block & (
            (kind[:,None] != KINDS.index('text')) | (position[None,:] <= position[:,None]))))
    noisy = kind == KINDS.index('noisy_latent')
    allowed &= (~noisy[None,:]) | same_block
    clean = (kind == KINDS.index('clean_latent')) | (kind == KINDS.index('semantic_vision'))
    allowed &= ~(noisy[:,None] & clean[None,:] & (image[:,None] == image[None,:]))
    return allowed


@dataclass
class UnifiedSegment:
    span: ContextSpan
    values: torch.Tensor
    xy: torch.Tensor | None = None  # [batch, tokens, 2], normalized patch centers
    timestep: torch.Tensor | None = None  # [batch], noise fraction in [0,1]
    supervise_text: bool = False


class SharedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.width
        self.heads = config.heads
        self.norm_attention = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
        self.norm_ffn = nn.RMSNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.SiLU(), nn.Linear(4*d, d))
        self.visual_ffn = nn.ModuleDict({name: copy.deepcopy(self.ffn) for name in ('editing','creativity')}
                                       if config.experts == 3 else {})
        self.q_norm = nn.RMSNorm(d//config.heads) if config.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(d//config.heads) if config.qk_norm else nn.Identity()
        self.visual_attention = nn.ModuleDict({name: nn.ModuleDict({
            'norm_attention':copy.deepcopy(self.norm_attention), 'qkv':copy.deepcopy(self.qkv),
            'output':copy.deepcopy(self.output), 'norm_ffn':copy.deepcopy(self.norm_ffn),
            'q_norm':copy.deepcopy(self.q_norm), 'k_norm':copy.deepcopy(self.k_norm)})
            for name in ('editing','creativity')} if config.attention_experts else {})

    def forward(self, hidden, mask, routes=None):
        b, n, d = hidden.shape
        if self.visual_attention:
            if routes is None:
                raise ValueError('Full transformer experts require explicit token routes')
            joint = hidden.new_zeros((b,n,3*d))
            for name,indices in routes.items():
                if not indices:continue
                unit = self.visual_attention[name] if name != 'understanding' else None
                norm = unit['norm_attention'] if unit is not None else self.norm_attention
                projection = unit['qkv'] if unit is not None else self.qkv
                q_norm = unit['q_norm'] if unit is not None else self.q_norm
                k_norm = unit['k_norm'] if unit is not None else self.k_norm
                q,k,v = projection(norm(hidden[:,indices])).chunk(3,dim=-1)
                shape=(b,len(indices),self.heads,d//self.heads)
                q=q_norm(q.reshape(shape)).reshape(b,len(indices),d)
                k=k_norm(k.reshape(shape)).reshape(b,len(indices),d)
                joint[:,indices]=torch.cat((q,k,v),dim=-1).to(joint.dtype)
            q,k,v=joint.chunk(3,dim=-1)
        else:
            q, k, v = self.qkv(self.norm_attention(hidden)).chunk(3, dim=-1)
            shape=(b,n,self.heads,d//self.heads)
            q=self.q_norm(q.reshape(shape)).reshape(b,n,d)
            k=self.k_norm(k.reshape(shape)).reshape(b,n,d)
        def heads(x):
            return x.reshape(b, n, self.heads, d//self.heads).transpose(1, 2)
        context = F.scaled_dot_product_attention(*map(heads, (q, k, v)), attn_mask=mask)
        context = context.transpose(1, 2).reshape(b, n, d)
        if self.visual_attention:
            delta=torch.zeros_like(hidden)
            for name,indices in routes.items():
                if not indices:continue
                projection=self.output if name == 'understanding' else self.visual_attention[name]['output']
                delta[:,indices]=projection(context[:,indices]).to(delta.dtype)
            hidden=hidden+delta
        else:
            hidden = hidden + self.output(context)
        if not self.visual_ffn:
            return hidden + self.ffn(self.norm_ffn(hidden))
        delta = torch.zeros_like(hidden)
        for name, indices in routes.items():
            if indices:
                expert = self.ffn if name == 'understanding' else self.visual_ffn[name]
                norm = (self.visual_attention[name]['norm_ffn']
                        if self.visual_attention and name != 'understanding' else self.norm_ffn)
                delta[:,indices] = expert(norm(hidden[:,indices])).to(delta.dtype)
        return hidden + delta


class HorizonUnifiedTransformer(nn.Module):
    """One attention operation over expert-projected tokens, with two output heads."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.width
        self.text = nn.Embedding(config.vocab_size, d)
        self.latent_in = nn.Linear(config.latent_width, d)
        self.vision_in = nn.Linear(config.vision_width, d)
        self.relation_in = nn.Linear(d, d)
        self.modality = nn.Embedding(len(KINDS), d)
        self.position = nn.Embedding(config.max_tokens, d)
        self.xy = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(SharedBlock(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(d)
        self.language_head = nn.Linear(d, config.vocab_size, bias=False)
        self.language_head.weight = self.text.weight
        self.flow_head = nn.Linear(d, config.latent_width)
        self._mask_cache = {}

    def _embed(self, segments):
        spans = [s.span for s in segments]
        tokens = expand_layout(spans, self.config.max_tokens)
        # A direct attention mask cannot prevent multi-layer leakage via an
        # intervening text/source token. Reject same-target clean features entirely.
        noisy = {(s.sample, s.image_id) for s in spans if s.kind == 'noisy_latent'}
        if any((s.sample, s.image_id) in noisy for s in spans
               if s.kind in {'clean_latent', 'semantic_vision'}):
            raise ValueError('No clean/semantic target features may accompany its noisy target')
        device = self.text.weight.device
        dtype = self.text.weight.dtype
        batch = segments[0].values.shape[0]
        embeddings = []
        for segment in segments:
            span, x = segment.span, segment.values
            if x.shape[:2] != (batch, span.length) or x.device != device:
                raise ValueError('Segment batch, length and device must match')
            if span.kind == 'text':
                if x.ndim != 2 or x.dtype != torch.long or (x < 0).any() or (x >= self.config.vocab_size).any():
                    raise ValueError('Text must contain vocabulary token IDs')
                h = self.text(x)
            else:
                width = (self.config.latent_width if span.kind in {'clean_latent','noisy_latent'}
                         else self.config.vision_width if span.kind == 'semantic_vision' else self.config.width)
                if x.ndim != 3 or x.shape[-1] != width or not x.is_floating_point() or not torch.isfinite(x).all():
                    raise ValueError('Invalid continuous modality features')
                layer = self.latent_in if span.kind in {'clean_latent','noisy_latent'} else self.vision_in if span.kind == 'semantic_vision' else self.relation_in
                h = layer(x.to(dtype))
            if span.kind in {'semantic_vision','clean_latent','noisy_latent'}:
                xy = segment.xy
                if xy is None or xy.shape != (batch, span.length, 2) or xy.device != device or not torch.isfinite(xy).all() or ((xy < 0)|(xy > 1)).any():
                    raise ValueError('Visual tokens require normalized 2D patch coordinates')
                h = h + self.xy(xy.to(dtype))
            elif segment.xy is not None:
                raise ValueError('Only visual tokens accept patch coordinates')
            if span.kind == 'noisy_latent':
                t = segment.timestep
                if t is None or t.shape != (batch,) or t.device != device or not torch.isfinite(t).all() or ((t < 0)|(t > 1)).any():
                    raise ValueError('Noisy latents require a bounded timestep per batch item')
                frequencies = torch.exp(-math.log(10000)*torch.arange(self.config.width//2, device=device).float()/max(1,self.config.width//2-1))
                phase = t.float()[:,None]*1000*frequencies[None]
                encoding = torch.cat((phase.sin(), phase.cos()), dim=-1)
                if encoding.shape[-1] < self.config.width:
                    encoding = F.pad(encoding,(0,1))
                h = h + self.time(encoding.to(dtype))[:,None,:]
            elif segment.timestep is not None:
                raise ValueError('Only noisy targets accept timesteps')
            if segment.supervise_text and span.kind != 'text':
                raise ValueError('Text supervision is only valid on text')
            embeddings.append(h + self.modality.weight[KINDS.index(span.kind)])
        hidden = torch.cat(embeddings, dim=1)
        positions = torch.arange(len(tokens), device=device)
        return hidden + self.position(positions)[None], spans

    def forward(self, segments, *, visual_mode=None):
        hidden, spans = self._embed(segments)
        if self.config.experts == 3 and any(s.kind == 'noisy_latent' for s in spans) and visual_mode not in ('editing','creativity'):
            raise ValueError('Noisy tokens need an explicit visual expert')
        if visual_mode not in (None,'editing','creativity'):
            raise ValueError('Unknown visual expert')
        key = (tuple(spans), str(hidden.device))
        if key not in self._mask_cache:
            if len(self._mask_cache) >= 8: self._mask_cache.clear()
            self._mask_cache[key] = unified_attention_mask(spans, hidden.device, self.config.max_tokens)
        mask = self._mask_cache[key]
        routes = {'understanding':[], 'editing':[], 'creativity':[]}
        for index,(span,_) in enumerate(expand_layout(spans,self.config.max_tokens)):
            routes[visual_mode if span.kind == 'noisy_latent' and visual_mode else 'understanding'].append(index)
        for block in self.blocks:
            hidden = block(hidden, mask, routes)
        hidden = self.norm(hidden)
        offset, text_positions, image_positions = 0, [], []
        for span in spans:
            if span.kind == 'text':text_positions.extend(range(offset,offset+span.length))
            if span.kind == 'noisy_latent':image_positions.extend(range(offset,offset+span.length))
            offset += span.length
        return {'hidden':hidden, 'text_positions':text_positions, 'flow_positions':image_positions,
                'text_logits':self.language_head(hidden[:,text_positions]),
                'flow':self.flow_head(hidden[:,image_positions])}

    def instruction_loss(self, correct_segments, wrong_segments, flow_targets,
                         verified_negative, *, visual_mode, margin=0.1, ranking_weight=1.0):
        """Rank explicitly conflicting instructions with exactly matched visual noise.

        Only the text conditioning may differ. Random unverified other captions
        and equivalent paraphrases must not be labeled verified negatives.
        """
        from .horizon_instruction_torch import instruction_flow_loss
        if visual_mode not in ('editing','creativity') or len(correct_segments)!=len(wrong_segments):
            raise ValueError('Provide paired visual instruction contexts')
        batch=correct_segments[0].values.shape[0] if correct_segments else 0
        device=correct_segments[0].values.device if correct_segments else None
        if verified_negative.shape!=(batch,) or verified_negative.dtype!=torch.bool or verified_negative.device!=device:
            raise ValueError('Each paired example needs a verified-negative flag on its device')
        text_changed=torch.zeros(batch,dtype=torch.bool,device=device)
        for correct,wrong in zip(correct_segments,wrong_segments):
            a,b=correct.span,wrong.span
            if (a.kind,a.block,a.sample,a.image_id)!=(b.kind,b.block,b.sample,b.image_id):
                raise ValueError('Counterfactual instructions must share the visual layout')
            if correct.supervise_text or wrong.supervise_text:
                raise ValueError('Instruction prefixes cannot be language targets in this objective')
            if a.kind=='text':
                if correct.values.shape!=wrong.values.shape:
                    text_changed[:]=True
                else:
                    text_changed |= (correct.values!=wrong.values).reshape(batch,-1).any(dim=1)
            elif a.length!=b.length or not torch.equal(correct.values,wrong.values):
                raise ValueError('Source features and noisy targets must match exactly')
            for name in ('xy','timestep'):
                x,y=getattr(correct,name),getattr(wrong,name)
                if (x is None)!=(y is None) or (x is not None and not torch.equal(x,y)):
                    raise ValueError('Patch coordinates and noise times must match exactly')
        if not text_changed.any():raise ValueError('The counterfactual instruction must differ')
        if (verified_negative & ~text_changed).any():
            raise ValueError('An identical instruction cannot be a verified negative')
        expected={i for i,s in enumerate(correct_segments) if s.span.kind=='noisy_latent'}
        if not expected or set(flow_targets)!=expected:
            raise ValueError('Provide one velocity target per noisy segment')
        positive=self(correct_segments,visual_mode=visual_mode)['flow']
        negative=self(wrong_segments,visual_mode=visual_mode)['flow']
        target=torch.cat([flow_targets[i] for i in sorted(expected)],dim=1)
        return instruction_flow_loss(positive,negative,target,verified_negative,
                                     margin=margin,ranking_weight=ranking_weight)

    def joint_loss(self, segments, flow_targets, *, visual_mode=None):
        """Next-token CE on supervised text + velocity MSE on explicit noisy targets.

        flow_targets maps segment index to noise-minus-clean latent velocities.
        Clean targets belong only in this loss argument, never model inputs.
        """
        output = self(segments, visual_mode=visual_mode)
        text_losses, flow_losses = [], []
        position, offset = 0, 0
        expected = {i for i,s in enumerate(segments) if s.span.kind == 'noisy_latent'}
        if set(flow_targets) != expected:
            raise ValueError('Provide exactly one velocity target for each noisy segment')
        for index, segment in enumerate(segments):
            length = segment.span.length
            if segment.supervise_text:
                if length < 2:raise ValueError('Supervised text needs a next token')
                logits = self.language_head(output['hidden'][:,position:position+length-1])
                text_losses.append(F.cross_entropy(logits.flatten(0,1), segment.values[:,1:].flatten()))
            if segment.span.kind == 'noisy_latent':
                target = flow_targets[index]
                prediction = output['flow'][:,offset:offset+length]
                if target.requires_grad or target.shape != prediction.shape or target.device != prediction.device or not torch.isfinite(target).all():
                    raise ValueError('Velocity targets must be finite detached aligned tensors')
                flow_losses.append(F.mse_loss(prediction.float(), target.float()))
                offset += length
            position += length
        zero = output['hidden'].sum()*0
        text_loss = torch.stack(text_losses).mean() if text_losses else zero
        flow_loss = torch.stack(flow_losses).mean() if flow_losses else zero
        if not text_losses and not flow_losses:raise ValueError('No supervised targets')
        return {'loss':text_loss+flow_loss,'text_loss':text_loss,'flow_loss':flow_loss}

    def save_checkpoint(self, directory):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        # Clone tied tensors so serialization does not duplicate shared storage.
        save_file({k:v.detach().cpu().clone().contiguous() for k,v in self.state_dict().items()},str(directory/'model.safetensors'))
        (directory/'config.json').write_text(json.dumps({'architecture':'HorizonUnifiedTransformer',
            'status':'research core; no pretrained donor conversion or real-image capability established',
            'config':asdict(self.config)},indent=2))

    @classmethod
    def upcycle(cls, model):
        """Copy a compatible Horizon core into full experts without changing its function.

        Does not convert Qwen/FLUX/BAGEL tensors or invent pretrained knowledge.
        Existing q/k normalization settings and FFN specialists are retained.
        """
        if model.config.attention_experts:
            raise ValueError('Model already contains full transformer experts')
        config=replace(model.config,experts=3,attention_experts=1)
        restored=cls(config).to(device=model.text.weight.device,dtype=model.text.weight.dtype)
        original=model.state_dict();mapped={}
        for name in restored.state_dict():
            source=name
            if '.visual_attention.' in name:
                prefix,tail=name.split('.visual_attention.',1)
                source=prefix+'.'+tail.split('.',1)[1]
            elif '.visual_ffn.' in name and name not in original:
                prefix,tail=name.split('.visual_ffn.',1)
                source=prefix+'.ffn.'+tail.split('.',1)[1]
            if source not in original:
                raise ValueError('Unsupported donor tensor: '+source)
            mapped[name]=original[source]
        restored.load_state_dict(mapped,strict=True)
        restored.train(model.training)
        return restored

    @classmethod
    def load_checkpoint(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        info=json.loads((directory/'config.json').read_text())
        if info['architecture'] != cls.__name__:raise ValueError('Wrong checkpoint architecture')
        model=cls(UnifiedConfig(**info['config']))
        state=load_file(str(directory/'model.safetensors'))
        if not torch.equal(state['text.weight'],state['language_head.weight']):
            raise ValueError('Tied language weights disagree')
        model.load_state_dict(state,strict=True)
        return model
