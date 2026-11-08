"""BAGEL-style rectified-flow targets with explicit reproducible noise times."""
import math
import torch


def sample_flow_timesteps(batch_size, *, device='cpu', generator=None, shift=1.0):
    if type(batch_size) is not int or batch_size < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError('Use a positive batch size and finite positive time shift')
    t = torch.randn((batch_size,), device=device, generator=generator, dtype=torch.float32).sigmoid()
    return (shift*t/(1+(shift-1)*t)).clamp(1e-5,1-1e-5)


def flow_training_pair(clean, noise, timestep):
    if clean.shape != noise.shape or clean.device != noise.device or clean.ndim < 2:
        raise ValueError('Clean and noise tensors must share batch/feature shape and device')
    if not clean.is_floating_point() or not noise.is_floating_point():
        raise ValueError('Flow features must be floating point')
    if timestep.shape != (clean.shape[0],) or timestep.device != clean.device:
        raise ValueError('Each batch item needs a timestep on the same device')
    if not torch.isfinite(clean).all() or not torch.isfinite(noise).all() or not torch.isfinite(timestep).all() or ((timestep<0)|(timestep>1)).any():
        raise ValueError('Flow inputs must be finite with timesteps in [0,1]')
    t = timestep.reshape((-1,)+(1,)*(clean.ndim-1))
    return (1-t)*clean+t*noise, (noise-clean).detach()
