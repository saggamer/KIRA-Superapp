from __future__ import annotations

"""Native MLX execution for the Qwen3-TTS Talker weight island.

This is the hidden-state entry point KIRA needs.  It intentionally starts at
Talker's 1,024-wide decoder stream instead of converting hidden states back to
text and calling the public TTS wrapper.
"""

from pathlib import Path
import threading
from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn


class TalkerMLP(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 3072):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TalkerAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        heads: int = 16,
        kv_heads: int = 8,
        head_dim: int = 128,
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.heads = int(heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.q_proj = nn.Linear(hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_theta)

    def __call__(
        self,
        hidden: mx.array,
        mask: mx.array | None = None,
        cache: dict[str, mx.array | None] | None = None,
    ) -> mx.array:
        batch, length, _ = hidden.shape
        offset = 0 if cache is None or cache["keys"] is None else cache["keys"].shape[-2]
        queries = self.q_proj(hidden).reshape(batch, length, self.heads, self.head_dim)
        keys = self.k_proj(hidden).reshape(batch, length, self.kv_heads, self.head_dim)
        values = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            if cache["keys"] is not None:
                keys = mx.concatenate((cache["keys"], keys), axis=-2)
                values = mx.concatenate((cache["values"], values), axis=-2)
            cache["keys"] = keys
            cache["values"] = values
        if self.kv_heads != self.heads:
            repeats = self.heads // self.kv_heads
            keys = mx.repeat(keys, repeats, axis=1)
            values = mx.repeat(values, repeats, axis=1)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.head_dim**-0.5, mask=mask
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch, length, self.heads * self.head_dim
        )
        return self.o_proj(attended)


class TalkerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TalkerAttention()
        self.mlp = TalkerMLP()
        self.input_layernorm = nn.RMSNorm(1024, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(1024, eps=1e-6)

    def __call__(
        self,
        hidden: mx.array,
        mask: mx.array | None = None,
        cache: dict[str, mx.array | None] | None = None,
    ) -> mx.array:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), mask, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TalkerDecoder(nn.Module):
    def __init__(self, layer_count: int = 28):
        super().__init__()
        self.codec_embedding = nn.Embedding(3072, 1024)
        self.layers = [TalkerDecoderLayer() for _ in range(layer_count)]
        self.norm = nn.RMSNorm(1024, eps=1e-6)

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: list[dict[str, mx.array | None]] | None = None,
    ) -> mx.array:
        length = inputs_embeds.shape[-2]
        if cache is None:
            layer_cache = [None] * len(self.layers)
            mask = nn.MultiHeadAttention.create_additive_causal_mask(length).astype(
                inputs_embeds.dtype
            )
        else:
            layer_cache = cache
            offset = 0 if cache[0]["keys"] is None else cache[0]["keys"].shape[-2]
            if length == 1:
                mask = None
            else:
                future = mx.arange(offset + length)[None, :] > (
                    offset + mx.arange(length)[:, None]
                )
                mask = mx.where(future, -mx.inf, 0.0).astype(inputs_embeds.dtype)
        hidden = inputs_embeds
        for layer, current_cache in zip(self.layers, layer_cache):
            hidden = layer(hidden, mask, current_cache)
        return self.norm(hidden)

    def make_cache(self) -> list[dict[str, mx.array | None]]:
        return [{"keys": None, "values": None} for _ in self.layers]


class TalkerCodePredictorDecoder(nn.Module):
    """Five-layer intra-frame decoder for codec groups 1 through 15."""

    def __init__(self):
        super().__init__()
        self.codec_embedding = [nn.Embedding(2048, 1024) for _ in range(15)]
        self.layers = [TalkerDecoderLayer() for _ in range(5)]
        self.norm = nn.RMSNorm(1024, eps=1e-6)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        length = inputs_embeds.shape[-2]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length).astype(
            inputs_embeds.dtype
        )
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return self.norm(hidden)


class TalkerCodePredictor(nn.Module):
    """Qwen3-TTS sub-talker with its real 15 embedding/head pairs."""

    def __init__(self):
        super().__init__()
        self.model = TalkerCodePredictorDecoder()
        self.lm_head = [nn.Linear(1024, 2048, bias=False) for _ in range(15)]

    def generate(
        self,
        talker_hidden: mx.array,
        first_code_embedding: mx.array,
    ) -> mx.array:
        """Greedily generate groups 1..15 for independent 12 Hz frames.

        Both inputs are ``[frames, 1024]``. The official sub-talker operates
        across code groups within a frame, so frames can be flattened across
        batch and time without allowing information to cross frame boundaries.
        """
        if talker_hidden.shape != first_code_embedding.shape:
            raise ValueError("Talker hidden state and group-0 embedding must match.")
        sequence = mx.stack((talker_hidden, first_code_embedding), axis=1)
        generated = []
        for group in range(15):
            hidden = self.model(sequence)
            code = mx.argmax(self.lm_head[group](hidden[:, -1]), axis=-1).astype(mx.int32)
            generated.append(code)
            if group < 14:
                sequence = mx.concatenate(
                    (sequence, self.model.codec_embedding[group](code)[:, None, :]),
                    axis=1,
                )
        return mx.stack(generated, axis=-1)


class TalkerTextProjection(nn.Module):
    """Official Talker text embedding/projection used only for protocol states."""

    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Embedding(151_936, 2048)
        self.linear_fc1 = nn.Linear(2048, 2048, bias=True)
        self.linear_fc2 = nn.Linear(2048, 1024, bias=True)

    def __call__(self, token_ids: mx.array) -> mx.array:
        hidden = self.text_embedding(token_ids)
        return self.linear_fc2(nn.silu(self.linear_fc1(hidden)))


class KiraMLXTalker(nn.Module):
    """Qwen Talker decoder with a direct KIRA hidden-state input."""

    def __init__(self):
        super().__init__()
        self.model = TalkerDecoder()
        self.codec_head = nn.Linear(1024, 3072, bias=False)
        self.code_predictor = TalkerCodePredictor()
        self.text_projection = TalkerTextProjection()

    def __call__(
        self,
        semantic_conditioning: mx.array,
        codec_conditioning: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if semantic_conditioning.ndim != 3 or semantic_conditioning.shape[-1] != 1024:
            raise ValueError("Talker conditioning must be [batch, time, 1024].")
        hidden = semantic_conditioning
        if codec_conditioning is not None:
            if codec_conditioning.shape[0] != hidden.shape[0] or codec_conditioning.shape[-1] != 1024:
                raise ValueError("Codec conditioning must match Talker batch and width.")
            if codec_conditioning.shape[-2] != hidden.shape[-2]:
                indices = (
                    mx.arange(hidden.shape[-2]) * codec_conditioning.shape[-2] // hidden.shape[-2]
                ).astype(mx.int32)
                codec_conditioning = mx.take(codec_conditioning, indices, axis=-2)
            hidden = hidden + codec_conditioning
        hidden = self.model(hidden)
        return hidden, self.codec_head(hidden)

    def generate_codec_groups(
        self,
        talker_hidden: mx.array,
        first_code_ids: mx.array | None = None,
    ) -> mx.array:
        """Return all 16 codec groups as ``[batch, time, 16]`` IDs."""
        if talker_hidden.ndim != 3 or talker_hidden.shape[-1] != 1024:
            raise ValueError("Talker hidden state must be [batch, time, 1024].")
        batch, time, width = talker_hidden.shape
        if first_code_ids is None:
            first_code_ids = mx.argmax(self.codec_head(talker_hidden), axis=-1).astype(mx.int32)
        if first_code_ids.shape != (batch, time):
            raise ValueError("First codec group must be [batch, time].")
        flat_hidden = talker_hidden.reshape(batch * time, width)
        flat_first = first_code_ids.reshape(batch * time)
        first_embedding = self.model.codec_embedding(flat_first)
        remaining = self.code_predictor.generate(flat_hidden, first_embedding)
        all_groups = mx.concatenate((flat_first[:, None], remaining), axis=-1)
        return all_groups.reshape(batch, time, 16)

    def generate_streaming_codec_groups(
        self,
        semantic_states: mx.array,
        *,
        speaker_embedding: mx.array | None = None,
        cancelled: threading.Event | None = None,
        max_frames: int = 720,
    ) -> mx.array:
        """Run the official Talker recurrence from KIRA semantic states.

        The semantic stream replaces Qwen Talker's trailing text projection;
        no text is decoded or re-prompted between the Thinker and Talker.
        """
        frames = list(
            self.iter_streaming_codec_groups(
                semantic_states,
                speaker_embedding=speaker_embedding,
                cancelled=cancelled,
                max_frames=max_frames,
            )
        )
        if not frames:
            return mx.zeros((1, 0, 16), dtype=mx.int32)
        return mx.concatenate(frames, axis=1)

    def iter_streaming_codec_groups(
        self,
        semantic_states: mx.array,
        *,
        speaker_embedding: mx.array | None = None,
        cancelled: threading.Event | None = None,
        max_frames: int = 720,
    ) -> Iterator[mx.array]:
        """Yield one complete 16-code frame as soon as it is generated."""
        if semantic_states.ndim != 3 or semantic_states.shape[0] != 1:
            raise ValueError("Talker semantic states must be [1, time, 1024].")
        if semantic_states.shape[1] == 0:
            return
        role_ids = mx.array([[151644, 77091, 198]], dtype=mx.int32)
        tts_ids = mx.array([[151672, 151673, 151671]], dtype=mx.int32)
        role = self.text_projection(role_ids)
        tts_special = self.text_projection(tts_ids)
        tts_bos = tts_special[:, 0:1]
        tts_eos = tts_special[:, 1:2]
        tts_pad = tts_special[:, 2:3]
        # English Base-model protocol: think, think-bos, language, think-eos,
        # optional speaker x-vector, codec-pad, codec-bos.
        codec_prefill_ids = mx.array(
            [[2154, 2156, 2050, 2157, 2148, 2149]], dtype=mx.int32
        )
        codec_prefill = self.model.codec_embedding(codec_prefill_ids)
        fixed_parts = [role, mx.repeat(tts_pad, 4, axis=1) + codec_prefill[:, :4]]
        if speaker_embedding is not None:
            if speaker_embedding.ndim == 1:
                speaker_embedding = speaker_embedding[None, None, :]
            elif speaker_embedding.ndim == 2:
                speaker_embedding = speaker_embedding[:, None, :]
            if speaker_embedding.shape != (1, 1, 1024):
                raise ValueError("Talker speaker embedding must be [1024], [1,1024], or [1,1,1024].")
            fixed_parts.append(tts_pad + speaker_embedding.astype(tts_pad.dtype))
        fixed_parts.extend(
            (
                tts_bos + codec_prefill[:, 4:5],
                semantic_states[:, :1] + codec_prefill[:, 5:6],
            )
        )
        fixed = mx.concatenate(fixed_parts, axis=1)
        trailing = mx.concatenate((semantic_states[:, 1:], tts_eos), axis=1)
        cache = self.model.make_cache()
        hidden = self.model(fixed, cache=cache)
        for frame_index in range(max_frames):
            if cancelled is not None and cancelled.is_set():
                break
            last_hidden = hidden[:, -1:]
            first_logits = self.codec_head(last_hidden)
            # Qwen3-TTS suppresses the protocol/control range for generated
            # group-0 audio codes, while keeping the single EOS code legal.
            # Selecting from this compact view is cheaper than materializing
            # a 3,072-wide -inf mask on every 12 Hz frame.
            valid_logits = mx.concatenate(
                (first_logits[..., :2048], first_logits[..., 2150:2151]), axis=-1
            )
            compact_first = mx.argmax(valid_logits, axis=-1).astype(mx.int32)
            first = mx.where(compact_first == 2048, 2150, compact_first)
            mx.eval(first)
            if frame_index >= 2 and int(first.item()) == 2150:
                break
            groups = self.generate_codec_groups(last_hidden, first)
            mx.eval(groups)
            yield groups
            combined = self.model.codec_embedding(groups[..., 0])
            for group in range(1, 16):
                combined = combined + self.code_predictor.model.codec_embedding[group - 1](
                    groups[..., group]
                )
            text_state = trailing[:, frame_index:frame_index + 1]
            if text_state.shape[1] == 0:
                text_state = tts_pad
            hidden = self.model(combined + text_state, cache=cache)

    @classmethod
    def from_safetensors(cls, checkpoint: Path) -> "KiraMLXTalker":
        model = cls()
        raw = mx.load(str(checkpoint))
        weights = []
        for name, value in raw.items():
            if (
                name.startswith("talker.model.layers.")
                or name.startswith("talker.model.codec_embedding.")
                or name.startswith("talker.model.norm.")
                or name.startswith("talker.codec_head.")
                or name.startswith("talker.code_predictor.")
                or name.startswith("talker.model.text_embedding.")
                or name.startswith("talker.text_projection.")
            ):
                key = name.removeprefix("talker.")
                if key.startswith("model.text_embedding."):
                    key = "text_projection.text_embedding." + key.removeprefix(
                        "model.text_embedding."
                    )
                elif key.startswith("text_projection."):
                    key = "text_projection." + key.removeprefix("text_projection.")
                weights.append((key, value))
        expected = (28 * 11 + 3) + (5 * 11 + 1 + 15 + 15) + 5
        if len(weights) != expected:
            raise ValueError(f"Incomplete Talker weights: expected {expected}, got {len(weights)}")
        model.load_weights(weights, strict=True)
        return model
