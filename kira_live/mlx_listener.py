from __future__ import annotations

"""Native MLX Qwen3-ASR acoustic encoder for KIRA's Listener island."""

import math
from pathlib import Path
from dataclasses import dataclass
import threading
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ListenerSemanticOutput:
    """Internal Listener result; transcript text is diagnostic, not a handoff."""

    token_ids: tuple[int, ...]
    text: str
    hidden_states: mx.array
    acoustic_hidden: mx.array
    unit_ids: mx.array
    cancelled: bool


class ListenerAttention(nn.Module):
    def __init__(self, width: int = 896, heads: int = 14):
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.k_proj = nn.Linear(width, width, bias=True)
        self.v_proj = nn.Linear(width, width, bias=True)
        self.q_proj = nn.Linear(width, width, bias=True)
        self.out_proj = nn.Linear(width, width, bias=True)

    def __call__(self, hidden: mx.array) -> mx.array:
        length = hidden.shape[0]
        q = self.q_proj(hidden).reshape(length, self.heads, self.head_dim).transpose(1, 0, 2)[None]
        k = self.k_proj(hidden).reshape(length, self.heads, self.head_dim).transpose(1, 0, 2)[None]
        v = self.v_proj(hidden).reshape(length, self.heads, self.head_dim).transpose(1, 0, 2)[None]
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5
        )
        return self.out_proj(attended[0].transpose(1, 0, 2).reshape(length, self.width))


class ListenerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ListenerAttention()
        self.self_attn_layer_norm = nn.LayerNorm(896)
        self.fc1 = nn.Linear(896, 3584, bias=True)
        self.fc2 = nn.Linear(3584, 896, bias=True)
        self.final_layer_norm = nn.LayerNorm(896)

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = hidden + self.self_attn(self.self_attn_layer_norm(hidden))
        return hidden + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(hidden))))


class ListenerAudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, 480, 3, stride=2, padding=1, bias=True)
        self.conv2d2 = nn.Conv2d(480, 480, 3, stride=2, padding=1, bias=True)
        self.conv2d3 = nn.Conv2d(480, 480, 3, stride=2, padding=1, bias=True)
        self.conv_out = nn.Linear(7680, 896, bias=False)
        self.layers = [ListenerEncoderLayer() for _ in range(18)]
        self.ln_post = nn.LayerNorm(896)

    @staticmethod
    def _positions(length: int, width: int = 896) -> mx.array:
        increment = math.log(10_000.0) / (width // 2 - 1)
        inverse = mx.exp(-increment * mx.arange(width // 2, dtype=mx.float32))
        scaled = mx.arange(length, dtype=mx.float32)[:, None] * inverse[None, :]
        return mx.concatenate([mx.sin(scaled), mx.cos(scaled)], axis=-1)

    @staticmethod
    def _post_cnn_length(length: int) -> int:
        for _ in range(3):
            length = (length - 1) // 2 + 1 if length > 0 else 0
        return length

    def __call__(self, input_features: mx.array, feature_length: int) -> mx.array:
        if input_features.ndim != 3 or input_features.shape[0] != 1 or input_features.shape[1] != 128:
            raise ValueError("Listener features must be [1, 128, frames].")
        padded = input_features.shape[-1]
        if padded % 100:
            raise ValueError("Listener frame count must be padded to a multiple of 100.")
        chunks = padded // 100
        chunked = input_features.reshape(1, 128, chunks, 100)
        chunked = chunked.transpose(0, 2, 1, 3).reshape(chunks, 128, 100, 1)
        convolved = nn.gelu(self.conv2d1(chunked))
        convolved = nn.gelu(self.conv2d2(convolved))
        convolved = nn.gelu(self.conv2d3(convolved))
        _, freq, time, channels = convolved.shape
        convolved = convolved.transpose(0, 2, 3, 1).reshape(chunks, time, channels * freq)
        convolved = self.conv_out(convolved)
        convolved = convolved + self._positions(time).astype(convolved.dtype)[None]

        valid = []
        remaining = max(0, min(int(feature_length), padded))
        for chunk in range(chunks):
            raw_length = min(100, remaining)
            remaining -= raw_length
            post_length = self._post_cnn_length(raw_length)
            if post_length:
                valid.append(convolved[chunk, :post_length])
        if not valid:
            raise ValueError("Listener received no voiced feature frames.")
        hidden = mx.concatenate(valid, axis=0)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.ln_post(hidden)


class ListenerProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(896, 896, bias=True)
        self.linear_2 = nn.Linear(896, 1024, bias=True)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.linear_2(nn.gelu(self.linear_1(hidden)))


class KiraMLXListener(nn.Module):
    def __init__(self, language_model: Any | None = None, tokenizer: Any | None = None):
        super().__init__()
        self.audio_tower = ListenerAudioTower()
        self.multi_modal_projector = ListenerProjector()
        self.language_model = language_model
        self.tokenizer = tokenizer

    def __call__(self, input_features: mx.array, feature_length: int) -> tuple[mx.array, mx.array]:
        acoustic = self.audio_tower(input_features, feature_length)
        projected = self.multi_modal_projector(acoustic)[None]
        # Stable content-derived acoustic IDs for the shared n-gram path. These
        # are not transcripts and never leave the hidden-state graph.
        bits = projected[0, :, :16] > 0
        powers = mx.array([1 << index for index in range(16)], dtype=mx.int32)[None, :]
        unit_ids = mx.sum(bits.astype(mx.int32) * powers, axis=-1)[None]
        return projected, unit_ids

    @staticmethod
    def _language_args(config: dict):
        from mlx_lm.models.qwen3 import ModelArgs

        text = config["text_config"]
        rope = text.get("rope_parameters") or {}
        return ModelArgs(
            model_type="qwen3",
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            intermediate_size=int(text["intermediate_size"]),
            num_attention_heads=int(text["num_attention_heads"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            vocab_size=int(text["vocab_size"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            rope_theta=float(rope.get("rope_theta", 1_000_000.0)),
            head_dim=int(text["head_dim"]),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", True)),
            rope_scaling=None,
        )

    def _prompt_embeddings(self, acoustic_hidden: mx.array) -> mx.array:
        if self.language_model is None or self.tokenizer is None:
            raise RuntimeError("Listener semantic decoder is not loaded.")
        # This is Qwen3-ASR's own multimodal protocol.  The acoustic states
        # replace its audio-pad slots directly; no transcript is fed to KIRA's
        # Thinker and no model-to-model text prompting occurs.
        prefix = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
        suffix = [151670, 151645, 198, 151644, 77091, 198]
        embed = self.language_model.model.embed_tokens
        prefix_hidden = embed(mx.array([prefix], dtype=mx.int32))
        suffix_hidden = embed(mx.array([suffix], dtype=mx.int32))
        return mx.concatenate(
            (prefix_hidden, acoustic_hidden.astype(prefix_hidden.dtype), suffix_hidden),
            axis=1,
        )

    def semantic_decode(
        self,
        input_features: mx.array,
        feature_length: int,
        *,
        cancelled: threading.Event | None = None,
        max_tokens: int = 256,
    ) -> ListenerSemanticOutput:
        """Decode with the checkpoint's full 28-layer Listener language island."""
        if self.language_model is None or self.tokenizer is None:
            raise RuntimeError("Use from_composite_checkpoint() for semantic decoding.")
        from mlx_lm.models.cache import make_prompt_cache

        acoustic_hidden, unit_ids = self(input_features, feature_length)
        prompt_hidden = self._prompt_embeddings(acoustic_hidden)
        cache = make_prompt_cache(self.language_model)
        empty = mx.zeros((1, prompt_hidden.shape[1]), dtype=mx.int32)
        logits = self.language_model(empty, cache=cache, input_embeddings=prompt_hidden)
        generated: list[int] = []
        was_cancelled = False
        stop_ids = {151643, 151645}
        for _ in range(max_tokens):
            if cancelled is not None and cancelled.is_set():
                was_cancelled = True
                break
            token = mx.argmax(logits[:, -1], axis=-1).astype(mx.int32)
            mx.eval(token)
            token_id = int(token.item())
            if token_id in stop_ids:
                break
            generated.append(token_id)
            logits = self.language_model(token[:, None], cache=cache)

        if generated:
            token_array = mx.array([generated], dtype=mx.int32)
            response_embeddings = self.language_model.model.embed_tokens(token_array)
            full_embeddings = mx.concatenate((prompt_hidden, response_embeddings), axis=1)
            all_hidden = self.language_model.model(
                mx.zeros((1, full_embeddings.shape[1]), dtype=mx.int32),
                input_embeddings=full_embeddings,
            )
            semantic_hidden = all_hidden[:, -len(generated):]
            # Qwen3-ASR emits ``language X<asr_text>`` before the actual
            # transcript. Keep that protocol wholly inside the Listener.
            if 151704 in generated:
                content_start = generated.index(151704) + 1
                content_ids = generated[content_start:]
                semantic_hidden = semantic_hidden[:, content_start:]
            else:
                content_ids = generated
            text = self.tokenizer.decode(content_ids, skip_special_tokens=True)
        else:
            content_ids = []
            semantic_hidden = mx.zeros((1, 0, 1024), dtype=acoustic_hidden.dtype)
            text = ""
        return ListenerSemanticOutput(
            tuple(content_ids), text, semantic_hidden, acoustic_hidden, unit_ids, was_cancelled
        )

    def teacher_semantic_states(
        self, acoustic_hidden: mx.array, transcript_token_ids: mx.array
    ) -> mx.array:
        """Return Listener decoder states for known transcript tokens in one pass.

        This is used only to prepare bridge supervision. Runtime speech still
        uses autoregressive ``semantic_decode`` and never receives ground truth.
        """
        if transcript_token_ids.ndim != 2 or transcript_token_ids.shape[0] != 1:
            raise ValueError("Transcript IDs must be [1, time].")
        prompt_hidden = self._prompt_embeddings(acoustic_hidden)
        protocol_ids = mx.array([[11528, 6364, 151704]], dtype=mx.int32)
        response_ids = mx.concatenate((protocol_ids, transcript_token_ids), axis=1)
        response_hidden = self.language_model.model.embed_tokens(response_ids)
        full = mx.concatenate((prompt_hidden, response_hidden), axis=1)
        hidden = self.language_model.model(
            mx.zeros((1, full.shape[1]), dtype=mx.int32), input_embeddings=full
        )
        return hidden[:, -transcript_token_ids.shape[1]:]

    @classmethod
    def from_safetensors(cls, checkpoint: Path) -> "KiraMLXListener":
        model = cls()
        raw = mx.load(str(checkpoint))
        weights = []
        for name, value in raw.items():
            if name.startswith("model.audio_tower."):
                key = "audio_tower." + name.removeprefix("model.audio_tower.")
            elif name.startswith("model.multi_modal_projector."):
                key = "multi_modal_projector." + name.removeprefix("model.multi_modal_projector.")
            else:
                continue
            if ".conv2d" in name and name.endswith(".weight"):
                value = value.transpose(0, 2, 3, 1)
            weights.append((key, value))
        if len(weights) != 301:
            raise ValueError(f"Incomplete Listener weights: expected 301, got {len(weights)}")
        model.load_weights(weights, strict=True)
        return model

    @classmethod
    def from_composite_checkpoint(
        cls, checkpoint: Path, config: dict, tokenizer: Any
    ) -> "KiraMLXListener":
        """Load all 611 Listener tensors: acoustic tower plus Qwen3 decoder."""
        from mlx_lm.models.qwen3 import Model

        language_model = Model(cls._language_args(config))
        model = cls(language_model=language_model, tokenizer=tokenizer)
        raw = mx.load(str(checkpoint))
        weights = []
        for name, value in raw.items():
            if name.startswith("model.audio_tower."):
                key = "audio_tower." + name.removeprefix("model.audio_tower.")
                if ".conv2d" in name and name.endswith(".weight"):
                    value = value.transpose(0, 2, 3, 1)
            elif name.startswith("model.multi_modal_projector."):
                key = "multi_modal_projector." + name.removeprefix(
                    "model.multi_modal_projector."
                )
            elif name.startswith("model.language_model."):
                key = "language_model.model." + name.removeprefix(
                    "model.language_model."
                )
            else:
                continue
            weights.append((key, value))
        if len(weights) != 611:
            raise ValueError(
                f"Incomplete composite Listener weights: expected 611, got {len(weights)}"
            )
        model.load_weights(weights, strict=True)
        return model
