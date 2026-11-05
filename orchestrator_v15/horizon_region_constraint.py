"""Explicit flow-latent region constraint; uses a predicted region, never a target."""
import torch

def constrain_flow_latents(latents,clean_source,initial_noise,region,next_sigma):
 if latents.shape!=clean_source.shape or latents.shape!=initial_noise.shape or region.shape!=(latents.shape[0],latents.shape[1],1):raise ValueError('Aligned packed source/noise/predicted-region tensors required')
 if not torch.isfinite(region).all() or (region<0).any() or (region>1).any():raise ValueError('Invalid predicted region')
 sigma=torch.as_tensor(next_sigma,device=latents.device,dtype=latents.dtype)
 source_trajectory=(1-sigma)*clean_source.to(latents.dtype)+sigma*initial_noise.to(latents.dtype)
 return region.to(latents.dtype)*latents+(1-region.to(latents.dtype))*source_trajectory

def flow_region_callback(clean_source,initial_noise,region):
 def callback(pipe,step,timestep,kwargs):
  kwargs['latents']=constrain_flow_latents(kwargs['latents'],clean_source,initial_noise,region,pipe.scheduler.sigmas[step+1])
  return kwargs
 return callback
