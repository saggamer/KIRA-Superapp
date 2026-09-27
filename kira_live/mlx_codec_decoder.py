from __future__ import annotations

"""Native MLX decoder for Qwen3-TTS 12 Hz codec groups to 24 kHz audio."""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .mlx_talker import TalkerMLP


class CausalConv1d(nn.Module):
    def __init__(self, inputs, outputs, kernel, *, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(inputs, outputs, kernel, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel = (kernel - 1) * dilation + 1
        self.left_padding = self.kernel - stride

    def __call__(self, hidden: mx.array) -> mx.array:
        length = hidden.shape[1]
        frames = (length - self.kernel + self.left_padding) / self.stride + 1
        ideal = (math.ceil(frames) - 1) * self.stride + self.kernel - self.left_padding
        extra = ideal - length
        hidden = mx.pad(hidden, ((0, 0), (self.left_padding, extra), (0, 0)))
        return self.conv(hidden)


class TransposedConv1d(nn.Module):
    """Exact stride transposed-convolution using zero insertion + Conv1d."""

    def __init__(self, inputs: int, outputs: int, kernel: int, stride: int):
        super().__init__()
        self.weight = mx.zeros((outputs, kernel, inputs))
        self.bias = mx.zeros((outputs,))
        self.stride = int(stride)
        self.kernel = int(kernel)

    def __call__(self, hidden: mx.array) -> mx.array:
        batch, length, channels = hidden.shape
        zeros = mx.zeros((batch, length, self.stride - 1, channels), dtype=hidden.dtype)
        expanded = mx.concatenate((hidden[:, :, None, :], zeros), axis=2).reshape(
            batch, length * self.stride, channels
        )
        expanded = expanded[:, : (length - 1) * self.stride + 1]
        weight = mx.flip(self.weight, axis=1)
        output = mx.conv1d(expanded, weight, padding=self.kernel - 1) + self.bias
        return output


class CausalTransConv1d(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride):
        super().__init__()
        self.conv = TransposedConv1d(inputs, outputs, kernel, stride)
        self.right_padding = kernel - stride

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = self.conv(hidden)
        return hidden[:, : hidden.shape[1] - self.right_padding] if self.right_padding else hidden


class SnakeBeta(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.alpha = mx.zeros((width,))
        self.beta = mx.zeros((width,))

    def __call__(self, hidden: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)[None, None, :]
        beta = mx.exp(self.beta)[None, None, :]
        return hidden + mx.square(mx.sin(hidden * alpha)) / (beta + 1e-9)


class ConvNeXtBlock(nn.Module):
    def __init__(self, width: int = 1024):
        super().__init__()
        self.dwconv = CausalConv1d(width, width, 7, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, 4 * width, bias=True)
        self.pwconv2 = nn.Linear(4 * width, width, bias=True)
        self.gamma = mx.zeros((width,))

    def __call__(self, hidden: mx.array) -> mx.array:
        update = self.dwconv(hidden)
        update = self.pwconv2(nn.gelu(self.pwconv1(self.norm(update))))
        return hidden + self.gamma * update


class LayerScale(nn.Module):
    def __init__(self, width: int = 512):
        super().__init__()
        self.scale = mx.ones((width,)) * 0.01

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.scale * hidden


class CodecAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 1024, bias=False)
        self.k_proj = nn.Linear(512, 1024, bias=False)
        self.v_proj = nn.Linear(512, 1024, bias=False)
        self.o_proj = nn.Linear(1024, 512, bias=False)
        self.rope = nn.RoPE(64, traditional=False, base=10_000.0)

    def __call__(self, hidden: mx.array, mask: mx.array) -> mx.array:
        batch, length, _ = hidden.shape
        q = self.q_proj(hidden).reshape(batch, length, 16, 64).transpose(0, 2, 1, 3)
        k = self.k_proj(hidden).reshape(batch, length, 16, 64).transpose(0, 2, 1, 3)
        v = self.v_proj(hidden).reshape(batch, length, 16, 64).transpose(0, 2, 1, 3)
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q), self.rope(k), v, scale=64**-0.5, mask=mask
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(batch, length, 1024))


class CodecTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = CodecAttention()
        self.mlp = TalkerMLP(hidden_size=512, intermediate_size=1024)
        self.input_layernorm = nn.RMSNorm(512, eps=1e-5)
        self.post_attention_layernorm = nn.RMSNorm(512, eps=1e-5)
        self.self_attn_layer_scale = LayerScale()
        self.mlp_layer_scale = LayerScale()

    def __call__(self, hidden: mx.array, mask: mx.array) -> mx.array:
        hidden = hidden + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(hidden), mask))
        return hidden + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(hidden)))


class CodecTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [CodecTransformerLayer() for _ in range(8)]
        self.norm = nn.RMSNorm(512, eps=1e-5)
        self.input_proj = nn.Linear(1024, 512, bias=True)
        self.output_proj = nn.Linear(512, 1024, bias=True)

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = self.input_proj(hidden)
        length = hidden.shape[1]
        positions = mx.arange(length)
        allowed = (positions[None, :] <= positions[:, None]) & (
            positions[None, :] > positions[:, None] - 72
        )
        mask = mx.where(allowed, mx.array(0, hidden.dtype), mx.array(-1e9, hidden.dtype))
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return self.output_proj(self.norm(hidden))


class Codebook(nn.Module):
    def __init__(self):
        super().__init__()
        self.cluster_usage = mx.ones((2048,))
        self.embedding_sum = mx.zeros((2048, 256))

    def decode(self, codes: mx.array) -> mx.array:
        embeddings = self.embedding_sum / mx.maximum(self.cluster_usage[:, None], 1e-5)
        return mx.take(embeddings, codes, axis=0)


class VQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebook = Codebook()

    def decode(self, codes: mx.array) -> mx.array:
        return self.codebook.decode(codes)


class ResidualVQ(nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.layers = [VQ() for _ in range(count)]

    def decode(self, codes: mx.array) -> mx.array:
        value = mx.zeros(codes.shape[:1] + codes.shape[2:] + (256,))
        for index, layer in enumerate(self.layers):
            value = value + layer.decode(codes[:, index])
        return value


class ResidualQuantizer(nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.input_proj = nn.Conv1d(512, 256, 1, bias=False)
        self.output_proj = nn.Conv1d(256, 512, 1, bias=False)
        self.vq = ResidualVQ(count)

    def decode(self, codes: mx.array) -> mx.array:
        return self.output_proj(self.vq.decode(codes))


class SplitQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rvq_first = ResidualQuantizer(1)
        self.rvq_rest = ResidualQuantizer(15)

    def decode(self, codes: mx.array) -> mx.array:
        return self.rvq_first.decode(codes[:, :1]) + self.rvq_rest.decode(codes[:, 1:])


class ResidualUnit(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.act1 = SnakeBeta(width)
        self.conv1 = CausalConv1d(width, width, 7, dilation=dilation)
        self.act2 = SnakeBeta(width)
        self.conv2 = CausalConv1d(width, width, 1)

    def __call__(self, hidden: mx.array) -> mx.array:
        return hidden + self.conv2(self.act2(self.conv1(self.act1(hidden))))


class DecoderBlock(nn.Module):
    def __init__(self, index: int, rate: int):
        super().__init__()
        inputs = 1536 // (2**index)
        outputs = 1536 // (2 ** (index + 1))
        self.block = [
            SnakeBeta(inputs),
            CausalTransConv1d(inputs, outputs, 2 * rate, rate),
            ResidualUnit(outputs, 1),
            ResidualUnit(outputs, 3),
            ResidualUnit(outputs, 9),
        ]

    def __call__(self, hidden: mx.array) -> mx.array:
        for block in self.block:
            hidden = block(hidden)
        return hidden


class KiraMLXCodecDecoder(nn.Module):
    sample_rate = 24_000
    samples_per_frame = 1_920

    def __init__(self):
        super().__init__()
        self.pre_transformer = CodecTransformer()
        self.quantizer = SplitQuantizer()
        self.pre_conv = CausalConv1d(512, 1024, 3)
        self.upsample = [
            [CausalTransConv1d(1024, 1024, 2, 2), ConvNeXtBlock()],
            [CausalTransConv1d(1024, 1024, 2, 2), ConvNeXtBlock()],
        ]
        self.decoder = [
            CausalConv1d(1024, 1536, 7),
            DecoderBlock(0, 8),
            DecoderBlock(1, 5),
            DecoderBlock(2, 4),
            DecoderBlock(3, 3),
            SnakeBeta(96),
            CausalConv1d(96, 1, 7),
        ]

    def __call__(self, codes: mx.array) -> mx.array:
        if codes.ndim != 3 or codes.shape[-1] != 16:
            raise ValueError("Codec IDs must be [batch, frames, 16].")
        hidden = self.quantizer.decode(codes.transpose(0, 2, 1))
        hidden = self.pre_transformer(self.pre_conv(hidden))
        for pair in self.upsample:
            for block in pair:
                hidden = block(hidden)
        for block in self.decoder:
            hidden = block(hidden)
        return mx.clip(hidden[..., 0], -1, 1)

    @classmethod
    def from_safetensors(cls, checkpoint: Path) -> "KiraMLXCodecDecoder":
        model = cls()
        raw = mx.load(str(checkpoint))
        weights = []
        for name, value in raw.items():
            if not name.startswith("decoder."):
                continue
            key = name.removeprefix("decoder.").replace("._codebook.", ".codebook.")
            if key.endswith(".conv.weight"):
                # PyTorch Conv1d is [out,in/groups,k]; ConvTranspose1d is
                # [in,out,k]. Both MLX modules store [out,k,in/groups].
                parts = key.split(".")
                transposed = (
                    (parts[0] == "upsample" and parts[2] == "0")
                    or (parts[0] == "decoder" and parts[1] in {"1", "2", "3", "4"} and parts[3] == "1")
                )
                value = value.transpose(1, 2, 0) if transposed else value.transpose(0, 2, 1)
            elif key.endswith("input_proj.weight") or key.endswith("output_proj.weight"):
                if value.ndim == 3:
                    value = value.transpose(0, 2, 1)
            weights.append((key, value))
        if len(weights) != 271:
            raise ValueError(f"Incomplete codec decoder weights: expected 271, got {len(weights)}")
        model.load_weights(weights, strict=True)
        return model
