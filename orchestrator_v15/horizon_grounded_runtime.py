"""Native-pipeline grounding helpers; no target/reference is an inference input."""
import numpy as np
import torch
from PIL import Image
from .horizon_grounding_torch import source_grounding
from .horizon_relationships import validate_conditioning_budget

@torch.no_grad()
def source_features(pipe,source):
 source=source.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
 pixels=pipe.image_processor.preprocess(source,height=512,width=512).to(device='cuda',dtype=pipe.vae.dtype)
 latents=pipe._encode_vae_image(pixels,generator=None)
 ids=pipe._prepare_latent_ids(latents).to(latents.device)
 return latents,ids

@torch.no_grad()
def predict_region(pipe,source_latents,source_ids,prompt):
 if pipe.transformer.horizon_region_hint is not None:raise ValueError('Clear spatial hints before source-only prediction')
 validate_conditioning_budget(pipe.tokenizer,prompt,256)
 embeds,ids=pipe.encode_prompt(prompt=prompt,device=source_latents.device,max_sequence_length=256)
 logits=source_grounding(pipe.transformer,source_latents,embeds,ids,source_ids,pipe._pack_latents)
 probability=logits.float().sigmoid().transpose(1,2).reshape(1,1,*source_latents.shape[-2:])
 return probability

def region_tokens(probability,threshold):
 return (probability>threshold).float().flatten(2).transpose(1,2)

def region_image(probability,size=(512,512),threshold=None):
 value=probability.detach().float().squeeze().cpu().numpy()
 if threshold is not None:value=(value>threshold).astype('float32')
 return Image.fromarray(np.uint8(np.clip(value*255,0,255))).resize(size,Image.Resampling.NEAREST)
