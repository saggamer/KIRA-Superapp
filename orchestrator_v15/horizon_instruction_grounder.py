"""Explicit image/text fusion and verified instruction-contrastive mask learning."""
import torch
from torch import nn
from torch.nn import functional as F

class InstructionGrounder(nn.Module):
 def __init__(self,image_width=3072,text_width=7680,width=128):
  super().__init__();self.image_width=image_width;self.text_width=text_width;self.width=width
  self.image_norm=nn.LayerNorm(image_width);self.image_projection=nn.Linear(image_width+3,width)
  self.text_norm=nn.LayerNorm(text_width);self.text_projection=nn.Linear(text_width,width)
  self.position=nn.Linear(6,width);self.film=nn.Linear(width,2*width)
  self.cross=nn.MultiheadAttention(width,4,dropout=0,batch_first=True)
  self.spatial=nn.TransformerEncoderLayer(width,4,dim_feedforward=2*width,dropout=0,batch_first=True,norm_first=True)
  self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,64),nn.GELU(),nn.Linear(64,1))
 def forward(self,image_features,rgb,text_features,text_mask,height=32,width=32):
  batch,tokens,channels=image_features.shape
  if channels!=self.image_width or tokens!=height*width or rgb.shape!=(batch,tokens,3) or text_features.shape[0]!=batch or text_features.shape[-1]!=self.text_width or text_mask.shape!=text_features.shape[:2]:raise ValueError('Aligned source RGB/image features and native instruction tokens required')
  if not text_mask.bool().any(1).all():raise ValueError('Instruction cannot be all padding')
  image=self.image_projection(torch.cat([self.image_norm(image_features.float()),rgb.float()],-1))
  text=self.text_projection(self.text_norm(text_features.float()))
  mask=text_mask.bool();pool=(text*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
  y,x=torch.meshgrid(torch.linspace(-1,1,height,device=image.device),torch.linspace(-1,1,width,device=image.device),indexing='ij')
  coordinates=torch.stack([x,y,torch.sin(torch.pi*x),torch.cos(torch.pi*x),torch.sin(torch.pi*y),torch.cos(torch.pi*y)],-1).reshape(1,tokens,6)
  image=image+self.position(coordinates)
  gamma,beta=self.film(pool).chunk(2,-1);image=image*(1+gamma[:,None].tanh())+beta[:,None]
  context=self.cross(image,text,text,key_padding_mask=~mask,need_weights=False)[0]
  return self.output(self.spatial(image+context))

def per_example_region_loss(logits,labels):
 if logits.shape!=labels.shape or logits.ndim!=3 or logits.shape[-1]!=1:raise ValueError('Aligned mask logits and labels required')
 labels=labels.detach().float();logits=logits.float()
 if not torch.isfinite(logits).all() or not torch.isfinite(labels).all() or (labels<0).any() or (labels>1).any():raise ValueError('Invalid region loss')
 positives=labels.sum((1,2));weight=((1-labels).sum((1,2))/positives.clamp_min(1)).clamp(1,20)[:,None,None]
 bce=F.binary_cross_entropy_with_logits(logits,labels,reduction='none',pos_weight=weight).mean((1,2))
 probability=logits.sigmoid();dice=1-(2*(probability*labels).sum((1,2))+1)/(probability.sum((1,2))+positives+1)
 return bce+dice

def paired_instruction_loss(logits,labels,swaps=None,verified=None,margin=.25):
 positive=per_example_region_loss(logits,labels);ranking=positive.sum()*0
 if swaps is not None:
  if verified is None or verified.dtype!=torch.bool or verified.shape!=(logits.shape[0],):raise ValueError('Explicit verified counterfactual pairs required')
  if swaps.shape!=(logits.shape[0],) or swaps.dtype!=torch.long or swaps.min()<0 or swaps.max()>=len(logits):raise ValueError('Invalid pair mapping')
  if ((swaps==torch.arange(len(logits),device=swaps.device))&verified).any():raise ValueError('An instruction cannot conflict with itself')
  negative=per_example_region_loss(logits[swaps],labels)
  if verified.any():ranking=F.relu(margin+positive[verified]-negative[verified]).mean()
 return positive.mean()+.5*ranking,{'region_loss':float(positive.detach().mean()),'ranking_loss':float(ranking.detach())}
