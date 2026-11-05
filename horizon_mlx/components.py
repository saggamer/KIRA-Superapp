"""MLX implementations of the trained Horizon auxiliary networks.

Safetensors retain PyTorch's [out, in] linear layout, also used by MLX.
No donor backbone weights are included in this module.
"""
from pathlib import Path
import math
import mlx.core as mx


def linear(x, weights, name):
    y = x @ weights[name + ".weight"].T
    return y + weights[name + ".bias"] if name + ".bias" in weights else y


def norm(x, weights, name):
    return mx.fast.layer_norm(x, weights[name + ".weight"], weights[name + ".bias"], 1e-5)


def attention(q, k, v, mask=None):
    scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = mx.where(mask, scores, -float("inf"))
    return mx.softmax(scores, axis=-1) @ v


class SemanticBridge:
    def __init__(self, path, mode='editing'):
        if mode not in {'editing','creativity'}:raise ValueError('Unknown Horizon bridge mode')
        self.mode=mode
        self.weights = {name:value.astype(mx.float32) for name,value in mx.load(str(path)).items()}
        if f'output.{mode}.weight' not in self.weights:raise ValueError('Requested bridge mode is missing')

    def __call__(self, native, source_mask=None):
        w = self.weights
        x = native.astype(mx.float32)
        q, k, v = [linear(x, w, name)[:, None] for name in ("query", "key", "value")]
        mask = None if source_mask is None else source_mask[:, None, None, :]
        mixed = attention(q, k, v, mask)[:, 0]
        delta = linear(mixed, w, f"output.{self.mode}") * mx.sigmoid(w[f"gates.{self.mode}"])
        return native + delta.astype(native.dtype)


class InstructionGrounder:
    def __init__(self, path):
        self.weights = {name:value.astype(mx.float32) for name,value in mx.load(str(path)).items()}

    def _mha(self, query, context, name, mask=None):
        w = self.weights
        packed_w = mx.split(w[name + ".in_proj_weight"], 3, axis=0)
        packed_b = mx.split(w[name + ".in_proj_bias"], 3, axis=0)
        q, k, v = [x @ weight.T + bias for x, weight, bias in
                   zip((query, context, context), packed_w, packed_b)]
        batch, _, width = q.shape
        q, k, v = [x.reshape(batch, x.shape[1], 4, width // 4).transpose(0, 2, 1, 3)
                   for x in (q, k, v)]
        padding = None if mask is None else mask[:, None, None, :]
        out = attention(q, k, v, padding).transpose(0, 2, 1, 3).reshape(batch, -1, width)
        return linear(out, w, name + ".out_proj")

    def __call__(self, image_features, rgb, text_features, text_mask, height=32, width=32):
        b, n, c = image_features.shape
        if c != 3072 or n != height * width or rgb.shape != (b, n, 3):
            raise ValueError("Aligned native source features and RGB patches are required")
        if text_features.shape[0] != b or text_features.shape[-1] != 7680 or text_mask.shape != text_features.shape[:2]:
            raise ValueError("Aligned complete native instruction tokens are required")
        if not bool(mx.all(mx.any(text_mask, axis=1))):
            raise ValueError("An instruction cannot contain only padding")
        w = self.weights
        image = linear(mx.concatenate([norm(image_features.astype(mx.float32), w, "image_norm"),
                                        rgb.astype(mx.float32)], axis=-1), w, "image_projection")
        text = linear(norm(text_features.astype(mx.float32), w, "text_norm"), w, "text_projection")
        mask = text_mask.astype(mx.bool_)
        pool = mx.sum(text * mask[..., None], axis=1) / mx.sum(mask, axis=1, keepdims=True)
        y, x = mx.meshgrid(mx.linspace(-1, 1, height), mx.linspace(-1, 1, width), indexing="ij")
        coordinates = mx.stack([x, y, mx.sin(mx.pi*x), mx.cos(mx.pi*x),
                                mx.sin(mx.pi*y), mx.cos(mx.pi*y)], axis=-1).reshape(1, n, 6)
        image = image + linear(coordinates, w, "position")
        gamma, beta = mx.split(linear(pool, w, "film"), 2, axis=-1)
        image = image * (1 + mx.tanh(gamma[:, None])) + beta[:, None]
        image = image + self._mha(image, text, "cross", mask)
        normalized = norm(image, w, "spatial.norm1")
        image = image + self._mha(normalized, normalized, "spatial.self_attn")
        # PyTorch TransformerEncoderLayer defaults to ReLU, independently of the output GELU.
        image = image + linear(mx.maximum(linear(norm(image, w, "spatial.norm2"), w,
                                                  "spatial.linear1"), 0), w, "spatial.linear2")
        out = linear(norm(image, w, "output.0"), w, "output.1")
        out = 0.5 * out * (1 + mx.erf(out / math.sqrt(2)))
        return linear(out, w, "output.3")


def constrain_region(latents, source, initial_noise, region, next_sigma):
    """Preserve the source/noise trajectory outside the model-predicted region."""
    outside = (1 - next_sigma) * source + next_sigma * initial_noise
    return (region * latents + (1 - region) * outside).astype(latents.dtype)
