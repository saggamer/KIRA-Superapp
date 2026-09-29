"""Function-preserving expert widening; never promotes a checkpoint."""
from pathlib import Path
import re
import numpy as np
from safetensors import safe_open


def expert_checkpoint_rank(path: str | Path) -> int:
    ranks = set()
    with safe_open(str(path), framework="np") as tensors:
        keys = set(tensors.keys())
        for layer in range(16):
            for expert in range(10):
                prefix = f"expert_layers.{layer}"
                down = f"{prefix}.down.{expert}.weight"
                up = f"{prefix}.up.{expert}.weight"
                if down not in keys or up not in keys:
                    raise ValueError(f"Incomplete expert checkpoint: {prefix}, expert {expert}")
                d = tensors.get_slice(down).get_shape()
                u = tensors.get_slice(up).get_shape()
                if len(d) != 2 or len(u) != 2 or u != [d[1], d[0]] or d[0] <= 0:
                    raise ValueError(f"Inconsistent expert shapes: {down}")
                ranks.add(d[0])
    if len(ranks) != 1:
        raise ValueError("Expert checkpoint mixes ranks.")
    return ranks.pop()


def expert_checkpoint_config(path: str | Path) -> dict:
    """Read every matrix pair; never infer a mixed layout from a single tensor."""
    from .expert_profiles import expert_layout
    shapes = []
    with safe_open(str(path), framework="np") as tensors:
        for layer in range(16):
            ranks = []
            for expert in range(10):
                d = tensors.get_slice(f"expert_layers.{layer}.down.{expert}.weight").get_shape()
                u = tensors.get_slice(f"expert_layers.{layer}.up.{expert}.weight").get_shape()
                if len(d) != 2 or u != [d[1], d[0]] or d[0] <= 0:
                    raise ValueError("Invalid expert pair.")
                ranks.append(d[0])
            shapes.append(tuple(ranks))
    if len(set(shapes)) != 1:
        raise ValueError("Expert ranks differ between layers.")
    ranks = shapes[0]
    if ranks == expert_layout(384, "split_50_50")[0]:
        return {"expert_rank": 384, "expert_profile": "split_50_50"}
    if len(set(ranks)) == 1:
        return {"expert_rank": ranks[0], "expert_profile": "legacy"}
    raise ValueError("Unsupported expert capacity layout.")


def split_expert_weights(weights, *, seed=20260929):
    """Widen from accepted channels and compensate changed family gains exactly."""
    from .expert_profiles import expert_layout, FAMILY_INDICES, LEGACY_BUDGETS
    ranks, budgets = expert_layout(384, "split_50_50")
    rng = np.random.default_rng(seed)
    result = dict(weights)
    expected = {f"expert_layers.{l}.{part}.{e}.weight" for l in range(16)
                for e in range(10) for part in ("down", "up")}
    if not expected.issubset(weights):
        raise ValueError("Missing source expert matrices.")
    gains = {e: old / new for family, old, new in zip(FAMILY_INDICES, LEGACY_BUDGETS, budgets) for e in family}
    for layer in range(16):
        for expert, rank in enumerate(ranks):
            down_key = f"expert_layers.{layer}.down.{expert}.weight"
            up_key = f"expert_layers.{layer}.up.{expert}.weight"
            down, up = weights[down_key], weights[up_key]
            old_rank, width = down.shape
            if up.shape != (width, old_rank) or old_rank >= rank:
                raise ValueError("Split surgery must widen a consistent legacy source.")
            tail = rng.normal(0, 0.01, (rank-old_rank, width)).astype(down.dtype)
            result[down_key] = np.concatenate((down, tail), axis=0)
            result[up_key] = np.concatenate(((up * gains[expert]).astype(up.dtype),
                                             np.zeros((width, rank-old_rank), dtype=up.dtype)), axis=1)
    return result


def widen_expert_weights(weights, rank: int, *, seed=20260928):
    """Copy old channels exactly; random down/zero up adds a trainable null path."""
    rng = np.random.default_rng(seed)
    result = dict(weights)
    widened = 0
    for key, down in weights.items():
        if not re.fullmatch(r"expert_layers\.\d+\.down\.\d+\.weight", key):
            continue
        up_key = key.replace(".down.", ".up.")
        up = weights.get(up_key)
        if down.ndim != 2 or up is None or up.shape != down.shape[::-1]:
            raise ValueError(f"Invalid expert matrix pair: {key}")
        old_rank, width = down.shape
        if rank <= old_rank:
            raise ValueError("Target rank must exceed every source rank.")
        tail = rng.normal(0, 0.01, (rank - old_rank, width)).astype(down.dtype)
        result[key] = np.concatenate((down, tail), axis=0)
        result[up_key] = np.concatenate((up, np.zeros((width, rank - old_rank), dtype=up.dtype)), axis=1)
        widened += 2
    if widened != 320:
        raise ValueError(f"Expected 320 expert matrices, found {widened}.")
    return result
