"""Training-only latent-change proxy, never a semantic evaluation mask."""
import torch

def balanced_edit_flow_loss(prediction, velocity, source_latents, target_latents,
                            weighting, preservation_share=0.6, threshold=0.1):
    if prediction.shape != velocity.shape or source_latents.shape != target_latents.shape or prediction.shape != target_latents.shape:
        raise ValueError('Aligned BCHW prediction, velocity, source and target required')
    if prediction.ndim != 4 or not 0 < preservation_share < 1 or threshold < 0:
        raise ValueError('Invalid latent edit objective')
    for tensor in (prediction, velocity, source_latents, target_latents, weighting):
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite latent edit objective')
    # This proxy is derived only from TRAINING references. VAE differences are
    # not segmentation labels and can spread beyond the intended edit region.
    changed = ((source_latents.detach().float() - target_latents.detach().float()).square().mean(1, keepdim=True).sqrt() > threshold).float()
    error = (weighting.detach().float() * (prediction.float() - velocity.detach().float()).square()).mean(1, keepdim=True)
    changed_count = changed.sum((1,2,3))
    preserved_count = (1-changed).sum((1,2,3))
    edit = (error * changed).sum((1,2,3)) / changed_count.clamp_min(1)
    preserve = (error * (1-changed)).sum((1,2,3)) / preserved_count.clamp_min(1)
    both = (changed_count > 0) & (preserved_count > 0)
    balanced = (1-preservation_share) * edit + preservation_share * preserve
    loss = torch.where(both, balanced, error.mean((1,2,3))).mean()
    return loss
