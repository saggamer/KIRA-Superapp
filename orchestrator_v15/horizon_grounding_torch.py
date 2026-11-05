"""Source-only region prediction and spatial hints inside the native transformer.

Grounding input is clean source latents plus the complete instruction. No target,
annotated mask, or noisy target is supplied to that forward pass.
"""
from functools import wraps
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F

class GroundingHead(nn.Module):
    def __init__(self,width,rank=128):
        super().__init__();self.width=width;self.rank=rank
        self.norm=nn.LayerNorm(width);self.hidden=nn.Linear(width,rank)
        self.output=nn.Linear(rank,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.region_embedding=nn.Parameter(torch.zeros(width))
    def forward(self,features):
        return self.output(F.gelu(self.hidden(self.norm(features.float()))))

def install_grounding(transformer,rank=128):
    if hasattr(transformer,'horizon_grounding'):raise ValueError('Grounding already installed')
    width=transformer.proj_out.in_features
    transformer.horizon_grounding=GroundingHead(width,rank).to(next(transformer.parameters()).device)
    transformer.horizon_region_hint=None
    original=transformer.forward
    @wraps(original)
    def forward(*args,**kwargs):
        ground_only=kwargs.pop('horizon_ground_only',False)
        hint=kwargs.pop('horizon_edit_region',None)
        if hint is None:hint=transformer.horizon_region_hint
        if ground_only and hint is not None:
            raise ValueError('Source-only grounding must not receive an edit mask or region hint')
        handles=[];captured=[]
        if ground_only:
            def capture(module,inputs,output):captured.append(transformer.horizon_grounding(inputs[0]))
            handles.append(transformer.proj_out.register_forward_hook(capture))
        elif hint is not None:
            def add_region(module,inputs,output):
                if hint.ndim!=3 or hint.shape[0]!=output.shape[0] or hint.shape[-1]!=1 or output.shape[1]!=2*hint.shape[1]:
                    raise ValueError('Spatial hint requires aligned target/source token grids')
                if not torch.isfinite(hint).all() or (hint<0).any() or (hint>1).any():raise ValueError('Invalid spatial hint')
                paired=torch.cat([hint,hint],dim=1).to(output.device,output.dtype)
                return output+paired*transformer.horizon_grounding.region_embedding.to(output.dtype)
            handles.append(transformer.x_embedder.register_forward_hook(add_region))
        try:
            result=original(*args,**kwargs)
            if ground_only:
                if len(captured)!=1:raise ValueError('Native grounding feature capture changed')
                return captured[0]
            return result
        finally:
            for handle in handles:handle.remove()
    transformer.forward=forward
    return transformer.horizon_grounding

def source_grounding(transformer,source_latents,prompt_embeds,text_ids,source_ids,pack):
    if source_latents.ndim!=4:raise ValueError('Grounding needs BCHW source latents')
    return transformer(hidden_states=pack(source_latents),timestep=torch.zeros(source_latents.shape[0],device=source_latents.device,dtype=source_latents.dtype),
                       guidance=None,encoder_hidden_states=prompt_embeds,txt_ids=text_ids,img_ids=source_ids,
                       horizon_ground_only=True,return_dict=False)

def region_loss(logits,labels):
    if logits.shape!=labels.shape or logits.ndim!=3 or logits.shape[-1]!=1:raise ValueError('Aligned B,N,1 mask supervision required')
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all() or (labels<0).any() or (labels>1).any():raise ValueError('Invalid mask supervision')
    labels=labels.detach().float();scores=logits.float()
    positive=labels.sum((1,2));negative=(1-labels).sum((1,2))
    weight=(negative/positive.clamp_min(1)).clamp(1,20)[:,None,None]
    bce=F.binary_cross_entropy_with_logits(scores,labels,reduction='none',pos_weight=weight).mean((1,2))
    probability=scores.sigmoid();dice=1-(2*(probability*labels).sum((1,2))+1)/(probability.sum((1,2))+positive+1)
    return (bce+dice).mean()

def annotated_flow_loss(prediction,velocity,weighting,region):
    if prediction.shape!=velocity.shape or region.shape!=(prediction.shape[0],1,*prediction.shape[-2:]):raise ValueError('Aligned native flow/mask grids required')
    if (region<0).any() or (region>1).any() or not torch.isfinite(region).all():raise ValueError('Invalid training mask')
    error=(weighting.float()*(prediction.float()-velocity.detach().float()).square()).mean(1,keepdim=True)
    if not torch.isfinite(error).all():raise ValueError('Nonfinite native flow loss')
    region=region.detach().float();edited=region.sum((1,2,3));preserved=(1-region).sum((1,2,3))
    edit_error=(error*region).sum((1,2,3))/edited.clamp_min(1)
    keep_error=(error*(1-region)).sum((1,2,3))/preserved.clamp_min(1)
    return torch.where((edited>0)&(preserved>0),.4*edit_error+.6*keep_error,error.mean((1,2,3))).mean()

def save_grounding(transformer,directory):
    from safetensors.torch import save_file
    path=Path(directory);path.mkdir(parents=True,exist_ok=True);head=transformer.horizon_grounding
    save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(path/'horizon_grounding.safetensors'))
    (path/'horizon_grounding.json').write_text(json.dumps({'width':head.width,'rank':head.rank,'input':'clean source latents and original instruction only','time':0.,'mask_supervision':'human inverse-alpha edit masks and explicit procedural/identity labels','generation_hint':'learned residual on corresponding target and source token locations'},indent=2))

def load_grounding(transformer,directory):
    from safetensors.torch import load_file
    path=Path(directory);config=json.loads((path/'horizon_grounding.json').read_text())
    head=install_grounding(transformer,config['rank'])
    if config['width']!=head.width:raise ValueError('Grounding/native width mismatch')
    head.load_state_dict(load_file(str(path/'horizon_grounding.safetensors')),strict=True)
    return head
