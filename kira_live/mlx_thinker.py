from __future__ import annotations

"""Native Qwen3.5 Thinker entry point for KIRA's hidden-state bridge."""

from pathlib import Path
from dataclasses import dataclass
import threading
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .mlx_layers import HashedNgramPLE, ThinkerOnlyExpertMoE
from .policy import live_text_system_instruction, live_conversation_system_instruction, tree_execution_instruction
from .generation_stream import bind_generation_stream


@dataclass(frozen=True)
class ThinkerGeneration:
    token_ids: tuple[int, ...]
    text: str
    hidden_states: mx.array
    cancelled: bool


@dataclass(frozen=True)
class ThinkerLivePrefix:
    embeddings: mx.array
    ngram_ids: mx.array


class KiraMLXThinker(nn.Module):
    """Wrap MLX-LM's real Qwen3.5 weights without constructing a text prompt."""

    def __init__(self, model: Any, tokenizer: Any, *, expert_rank: int = 128, expert_profile: str = "legacy"):
        super().__init__()
        if expert_rank <= 0:
            raise ValueError("expert_rank must be positive.")
        self.expert_rank = int(expert_rank)
        self.expert_profile = expert_profile
        self.model = model
        self.tokenizer = tokenizer
        self.language_model = getattr(model, "language_model", model)
        self.decoder = getattr(self.language_model, "model", None)
        if self.decoder is None or not hasattr(self.decoder, "embed_tokens"):
            raise TypeError("Loaded checkpoint does not expose the Qwen3.5 hidden-state decoder.")
        self.hidden_size = int(self.decoder.embed_tokens.weight.shape[-1])
        if len(self.decoder.layers) != 24:
            raise ValueError("KIRA Thinker expects the 24-layer Qwen3.5 layout.")
        self.expert_layers = [
            ThinkerOnlyExpertMoE(hidden_size=self.hidden_size, rank=self.expert_rank, profile=expert_profile)
            for _ in range(16)
        ]
        self.ngram_layer_indices = (2, 6, 10, 14, 18, 22)
        self.ngram_layers = [
            HashedNgramPLE(
                bucket_count=262_144,
                embedding_dim=64,
                hidden_size=self.hidden_size,
                ngram_sizes=(2, 3, 4),
                dilation=2,
                modality_salt=43_117 + index * 1_009,
            )
            for index in self.ngram_layer_indices
        ]
        # Leading underscore keeps this dynamic activation out of MLX's
        # parameter tree. Sequence lengths vary from example to example.
        self._last_router_logits = None
        self._generation_emotion = None
        self._generation_ngram_ids = None
        self._generation_ngram_offset = 0
        self._generation_ngram_history = None

    @classmethod
    def from_pretrained(cls, checkpoint: Path, *, expert_rank: int = 128, expert_profile: str = "legacy") -> "KiraMLXThinker":
        from mlx_lm import load

        try:
            model, tokenizer = load(str(checkpoint), lazy=True)
        except TypeError:
            model, tokenizer = load(str(checkpoint))
        return cls(model, tokenizer, expert_rank=expert_rank, expert_profile=expert_profile)

    @property
    def active_moe_parameters(self) -> int:
        """Token-selected expert matrices plus the always-used router/context."""
        from .expert_profiles import selected_parameter_budget
        return selected_parameter_budget(self.hidden_size, len(self.expert_layers), self.expert_rank, self.expert_profile)["total"]

    def embed(self, token_ids: mx.array) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError("Thinker token IDs must be [batch, time].")
        return self.decoder.embed_tokens(token_ids)

    def build_live_prefix(
        self,
        listener_semantic_states: mx.array,
        *,
        memory_context: str = "",
        listener_unit_ids: mx.array | None = None,
        tree_planning: bool = False,
        conversation_turns: list[dict[str, str]] | None = None,
        artifact_content: bool = False,
    ) -> ThinkerLivePrefix:
        """Place direct Listener states into KIRA's private chat protocol.

        Only KIRA-owned fixed policy/memory text is tokenized here. The user's
        speech remains a hidden-state span and is never decoded then re-prompted
        into the Thinker.
        """
        if (listener_semantic_states.ndim != 3 or listener_semantic_states.shape[0] != 1
                or listener_semantic_states.shape[-1] != self.hidden_size):
            raise ValueError(f"Listener semantic states must be [1, time, {self.hidden_size}].")
        # Keep the same identity and safety contract while focusing the small
        # live decoder on conversation instead of unrelated coding instructions.
        policy = live_conversation_system_instruction()
        if not tree_planning:
            policy += "\nSpeak naturally without JSON or routing tags. Use relevant chat memory as evidence; the user's possessions are not your own."
        if tree_planning:
            policy = live_text_system_instruction() + tree_execution_instruction()
        if memory_context.strip():
            policy += "\n\n" + memory_context.strip()
        if artifact_content:
            policy += '\nWrite ONLY the requested document body, not tool commands or promises. For a story, write a complete original story in four paragraphs, about 180 words, with an ending. KIRA OS will create the Word file from your content. Do not discuss your ability to create files.'
        before_text = f"<|im_start|>system\n{policy}<|im_end|>\n"
        for turn in conversation_turns or ():
            role = turn.get('role')
            if role not in {'user', 'assistant'}:
                continue
            content = str(turn.get('content') or '')
            # A stored transcript must not manufacture protocol/system turns.
            for marker in ('<|im_start|>', '<|im_end|>', '<|im_sep|>', '<|endoftext|>'):
                content = content.replace(marker, '')
            before_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        before_text += "<|im_start|>user\n"
        # Match the public-answer prefill used by KIRA's text worker. Qwen3.5
        # otherwise starts a fresh thinking segment for voice turns, which can
        # leak internal tokens or wander before the short live token cap.
        after_text = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        before_ids = mx.array(
            [self.tokenizer.encode(before_text, add_special_tokens=False)], dtype=mx.int32
        )
        after_ids = mx.array(
            [self.tokenizer.encode(after_text, add_special_tokens=False)], dtype=mx.int32
        )
        before_embeddings = self.embed(before_ids)
        after_embeddings = self.embed(after_ids)
        embeddings = mx.concatenate(
            (
                before_embeddings,
                listener_semantic_states.astype(before_embeddings.dtype),
                after_embeddings,
            ),
            axis=1,
        )
        if listener_unit_ids is None:
            audio_ids = mx.zeros(listener_semantic_states.shape[:2], dtype=mx.int32)
        else:
            if listener_unit_ids.ndim != 2 or listener_unit_ids.shape[0] != 1:
                raise ValueError("Listener unit IDs must be [1, time].")
            indices = (
                mx.arange(listener_semantic_states.shape[1])
                * listener_unit_ids.shape[1]
                // listener_semantic_states.shape[1]
            ).astype(mx.int32)
            audio_ids = mx.take(listener_unit_ids, indices, axis=1)
        ngram_ids = mx.concatenate((before_ids, audio_ids, after_ids), axis=1)
        return ThinkerLivePrefix(embeddings, ngram_ids)

    def forward_hidden(
        self,
        input_embeddings: mx.array,
        emotion_features: mx.array | None = None,
        ngram_ids: mx.array | None = None,
        logits_from: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """Run all real Thinker layers from fused KIRA embeddings."""
        if input_embeddings.ndim != 3 or input_embeddings.shape[-1] != self.hidden_size:
            raise ValueError(f"Thinker input embeddings must be [batch, time, {self.hidden_size}].")
        hidden, logits = self._forward(
            input_embeddings,
            [None] * len(self.decoder.layers),
            emotion_features,
            ngram_ids,
            logits_from=logits_from,
        )
        return hidden, logits

    def _forward(
        self,
        input_embeddings: mx.array,
        cache: list[Any],
        emotion_features: mx.array | None,
        ngram_ids: mx.array | None = None,
        ngram_history: mx.array | None = None,
        logits_from: int = 0,
    ) -> tuple[mx.array, mx.array]:
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        hidden = input_embeddings
        ple_ids = (
            mx.concatenate((ngram_history, ngram_ids), axis=1)
            if ngram_history is not None and ngram_ids is not None
            else ngram_ids
        )
        fa_mask = create_attention_mask(hidden, cache[self.decoder.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[self.decoder.ssm_idx])
        router_logits = []
        for index, (layer, layer_cache) in enumerate(zip(self.decoder.layers, cache)):
            if index in self.ngram_layer_indices and ngram_ids is not None:
                if ngram_ids.shape != hidden.shape[:-1]:
                    raise ValueError("Thinker n-gram IDs must match hidden batch/time.")
                ple_index = self.ngram_layer_indices.index(index)
                ple = self.ngram_layers[ple_index](ple_ids)
                hidden = hidden + ple[:, -hidden.shape[1]:]
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
            if index >= 8:
                hidden, routed = self._apply_expert(index - 8, hidden, emotion_features)
                router_logits.append(routed)
        hidden = self.decoder.norm(hidden)
        self._last_router_logits = mx.stack(router_logits, axis=0)
        prediction_hidden = hidden[:, logits_from:]
        args = getattr(self.language_model, "args", None)
        if args is not None and bool(getattr(args, "tie_word_embeddings", False)):
            logits = self.decoder.embed_tokens.as_linear(prediction_hidden)
        else:
            logits = self.language_model.lm_head(prediction_hidden)
        return hidden, logits

    def _apply_expert(self, index, hidden, emotion_features):
        """Extension hook; the accepted daily-driver path is unchanged."""
        return self.expert_layers[index](hidden, emotion_features)

    @property
    def last_router_logits(self):
        return self._last_router_logits

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        """MLX-LM generation interface with KIRA experts inside every step."""
        if input_embeddings is None:
            input_embeddings = self.embed(inputs)
            ngram_ids = inputs
        elif self._generation_ngram_ids is not None:
            length = input_embeddings.shape[1]
            start = self._generation_ngram_offset
            stop = start + length
            if stop <= self._generation_ngram_ids.shape[1]:
                ngram_ids = self._generation_ngram_ids[:, start:stop]
            elif inputs.ndim == 2 and inputs.shape[1] == length:
                # After the hidden Listener prefix has been consumed,
                # MLX-LM supplies each newly generated token through `inputs`.
                # Use those IDs for causal PLE instead of slicing beyond the
                # fixed prefix and producing a zero-length n-gram tensor.
                ngram_ids = inputs
            else:
                ngram_ids = None
            self._generation_ngram_offset = stop
        else:
            ngram_ids = inputs if inputs.shape[1] == input_embeddings.shape[1] else None
        if cache is None:
            cache = self.make_cache()
        _, logits = self._forward(
            input_embeddings,
            cache,
            self._generation_emotion,
            ngram_ids,
            self._generation_ngram_history,
            logits_from=-1,
        )
        if ngram_ids is not None:
            prior = self._generation_ngram_history
            combined = mx.concatenate((prior, ngram_ids), axis=1) if prior is not None else ngram_ids
            self._generation_ngram_history = combined[:, -7:]
        return logits

    @property
    def layers(self):
        return self.decoder.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def generate_from_embeddings(
        self,
        input_embeddings: mx.array,
        *,
        emotion_features: mx.array | None = None,
        ngram_ids: mx.array | None = None,
        cancelled: threading.Event | None = None,
        max_tokens: int = 160,
        compute_response_hidden: bool = True,
        previous_replies: list[tuple[int, ...]] | None = None,
    ) -> ThinkerGeneration:
        """Cancellably decode public response tokens from a hidden audio prefix.

        The Listener prefix stays as embeddings throughout; it is never decoded
        to a transcript and then re-prompted into the Thinker.
        """
        if input_embeddings.ndim != 3 or input_embeddings.shape[0] != 1:
            raise ValueError("Generation embeddings must be [1, time, width].")
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors
        from .response_repetition import make_reply_copy_penalty
        # MLX-LM's module-level stream may have been imported by the UI thread.
        # Bind only our invocation to this inference thread's active GPU stream.
        generate_step = bind_generation_stream(generate_step, mx.default_stream(mx.gpu))

        self._generation_emotion = emotion_features
        if ngram_ids is not None and ngram_ids.shape != input_embeddings.shape[:-1]:
            raise ValueError("Generation n-gram IDs must match embedding batch/time.")
        self._generation_ngram_ids = ngram_ids
        self._generation_ngram_offset = 0
        self._generation_ngram_history = None
        stop_ids = set()
        for value in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        ):
            if isinstance(value, int) and value >= 0:
                stop_ids.add(value)
        configured = getattr(self.tokenizer, "eos_token_ids", ()) or ()
        if isinstance(configured, int):
            configured = (configured,)
        stop_ids.update(int(value) for value in configured)
        generated: list[int] = []
        was_cancelled = False
        processors = make_logits_processors(repetition_penalty=1.08, repetition_context_size=64)
        if previous_replies:
            processors.append(make_reply_copy_penalty(previous_replies))
        try:
            iterator = generate_step(
                mx.array([], dtype=mx.int32),
                self,
                input_embeddings=input_embeddings[0],
                max_tokens=max_tokens,
                logits_processors=processors,
            )
            for token, _logprobs in iterator:
                if cancelled is not None and cancelled.is_set():
                    was_cancelled = True
                    break
                token = int(token)
                if token in stop_ids:
                    break
                generated.append(token)
        finally:
            self._generation_emotion = None
            self._generation_ngram_ids = None
            self._generation_ngram_offset = 0
            self._generation_ngram_history = None
        if generated and compute_response_hidden:
            token_array = mx.array([generated], dtype=mx.int32)
            response_embeddings = self.embed(token_array).astype(input_embeddings.dtype)
            full_embeddings = mx.concatenate((input_embeddings, response_embeddings), axis=1)
            if ngram_ids is not None:
                full_ngram_ids = mx.concatenate((ngram_ids, token_array), axis=1)
            else:
                full_ngram_ids = None
            full_hidden, _ = self.forward_hidden(
                full_embeddings,
                emotion_features,
                full_ngram_ids,
            )
            response_hidden = full_hidden[:, -len(generated):]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
        else:
            response_hidden = mx.zeros((1, 0, self.hidden_size), dtype=input_embeddings.dtype)
            text = self.tokenizer.decode(generated, skip_special_tokens=True) if generated else ""
        return ThinkerGeneration(tuple(generated), text, response_hidden, was_cancelled)
