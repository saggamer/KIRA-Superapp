"""Condition-sensitive flow ranking for explicitly verified wrong instructions."""
import math
import torch
from torch.nn import functional as F


def instruction_flow_loss(correct,wrong,target,verified_negative,*,margin=0.1,ranking_weight=1.0):
    if correct.shape!=wrong.shape or correct.shape!=target.shape or correct.ndim<2:
        raise ValueError('Both instruction predictions must match the target shape')
    if correct.device!=wrong.device or correct.device!=target.device or target.requires_grad:
        raise ValueError('Use predictions on one device and a detached target')
    if not correct.is_floating_point() or not wrong.is_floating_point() or not target.is_floating_point():
        raise ValueError('Flow predictions and targets must be floating point')
    if verified_negative.shape!=(correct.shape[0],) or verified_negative.dtype!=torch.bool or verified_negative.device!=correct.device:
        raise ValueError('Each example needs an explicit verified-negative flag')
    if not math.isfinite(margin) or margin<=0 or not math.isfinite(ranking_weight) or ranking_weight<0:
        raise ValueError('Use a finite positive margin and nonnegative ranking weight')
    if not torch.isfinite(correct).all() or not torch.isfinite(target).all() or not torch.isfinite(wrong[verified_negative]).all():
        raise ValueError('Known predictions and targets must be finite')
    dims=tuple(range(1,correct.ndim))
    positive=(correct.float()-target.float()).square().mean(dims)
    # Unknown/paraphrase negatives contribute no ranking signal, even if missing.
    mask=verified_negative.reshape((-1,)+(1,)*(correct.ndim-1))
    safe_wrong=torch.where(mask,wrong,correct.detach())
    negative=(safe_wrong.float()-target.float()).square().mean(dims)
    gaps=negative-positive
    rank=F.relu(margin-gaps).masked_fill(~verified_negative,0).sum()/verified_negative.sum().clamp_min(1)
    return {'loss':positive.mean()+ranking_weight*rank,'positive_loss':positive.mean(),
            'ranking_loss':rank,'conditioning_gap':gaps.masked_fill(~verified_negative,0),
            'verified_negatives':int(verified_negative.sum())}
