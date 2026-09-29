from __future__ import annotations

"""KIRA's integrated Qwen speech island.

The Thinker does not chat with a second language model.  Its public token
stream and the Listener's acoustic emotion state are transported directly to
the Qwen3-TTS Talker/codec stack.  Text exists at this boundary only as the
deterministic decoding of those public tokens required by Qwen3-TTS's text
embedding input.
"""

from dataclasses import dataclass
import os
import threading
from typing import Iterator, Sequence

import numpy as np

from .emotion import EmotionState
from .privacy import sanitize_public_output
from .response_prosody import response_style


TALKER_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
TALKER_SPEAKER = "Aiden"


def optimize_talker(model, bits: int) -> dict:
    """Quantize only Talker linear layers in memory; never change the codec."""
    if bits not in (0, 8):
        raise ValueError('KIRA_LIVE_TALKER_BITS must be 0 (original BF16) or 8.')
    count = 0
    if bits:
        import mlx.core as mx
        import mlx.nn as nn
        def eligible(_name, module):
            nonlocal count
            accepted = isinstance(module, nn.Linear) and module.weight.shape[-1] % 64 == 0
            count += int(accepted)
            return accepted
        nn.quantize(model.talker, group_size=64, bits=bits, class_predicate=eligible)
        mx.eval(model.talker.parameters())
    return {'linear_bits': bits, 'group_size': 64 if bits else None,
            'quantized_linear_layers': count, 'waveform_decoder_unchanged': True,
            'original_weight_files_unchanged': True}


@dataclass(frozen=True)
class IntegratedSpeechPlan:
    token_ids: tuple[int, ...]
    public_text: str
    speaker: str
    language: str
    style_instruction: str


def emotion_style_instruction(state: EmotionState) -> str:
    """Map measured prosody to restrained speech style, without an LLM prompt."""
    if state.confidence < 0.35:
        return "Speak clearly, naturally, and steadily. Do not exaggerate emotion."

    styles = {
        "neutral": "natural, composed, and attentive",
        "joy": "warm, bright, and genuinely pleased",
        "sadness": "gentle, supportive, and softly reassuring",
        "anger": "calm, controlled, respectful, and firm",
        "fear": "steady, reassuring, patient, and safe",
        "surprise": "engaged and lightly animated while remaining clear",
    }
    pace = (
        "Use a slightly quicker conversational pace."
        if state.arousal >= 0.72
        else "Use an unhurried pace."
        if state.arousal <= 0.28
        else "Use a natural conversational pace."
    )
    return (
        f"Speak in a {styles.get(state.label, styles['neutral'])} manner. "
        f"{pace} Keep the voice coherent and avoid theatrical exaggeration."
    )


class KiraIntegratedQwenVoice:
    """MLX Qwen3-TTS Talker driven by KIRA tokens and direct prosody state."""

    def __init__(
        self,
        *,
        model_id: str = TALKER_MODEL_ID,
        revision: str | None = None,
        speaker: str = TALKER_SPEAKER,
    ):
        # Keep this import lazy: controller/unit tests must not require Metal.
        from mlx_audio.tts.utils import load_model

        load_kwargs = {"lazy": True}
        if revision:
            load_kwargs["revision"] = revision
        try:
            self.model = load_model(model_id, local_files_only=True, **load_kwargs)
        except TypeError:
            self.model = load_model(model_id, **load_kwargs)
        self.model.eval()
        speakers = {str(item).lower() for item in self.model.get_supported_speakers()}
        if speaker.lower() not in speakers:
            raise ValueError(f"KIRA speaker {speaker!r} is unavailable: {sorted(speakers)}")
        if getattr(self.model.config, "tts_model_size", "") != "1b7":
            raise ValueError("KIRA emotional voice requires the 1.7B instruction-capable Talker.")
        self.model_id = model_id
        self.revision = revision
        self.speaker = speaker
        self.optimization = optimize_talker(self.model, int(os.environ.get('KIRA_LIVE_TALKER_BITS', '8')))
        self.playback_prebuffer = 1 if self.optimization['linear_bits'] == 8 else 2

    @staticmethod
    def build_plan(
        token_ids: Sequence[int],
        public_text: str,
        thinker_tokenizer,
        emotion: EmotionState,
    ) -> IntegratedSpeechPlan:
        ids = tuple(int(item) for item in token_ids)
        decoded = sanitize_public_output(
            thinker_tokenizer.decode(list(ids), skip_special_tokens=True), live=True
        )
        safe_text = sanitize_public_output(public_text, live=True)
        # The Thinker token stream is authoritative. A harmless tokenizer
        # spacing difference may use the already-sanitized generated string.
        if decoded and " ".join(decoded.split()) != " ".join(safe_text.split()):
            safe_text = decoded
        if not safe_text:
            raise ValueError("The Thinker produced no public speech tokens.")
        return IntegratedSpeechPlan(
            token_ids=ids,
            public_text=safe_text,
            speaker=TALKER_SPEAKER,
            language="English",
            style_instruction=response_style(safe_text, emotion_style_instruction(emotion)),
        )

    def iter_audio(
        self,
        plan: IntegratedSpeechPlan,
        cancelled: threading.Event,
    ) -> Iterator[tuple[np.ndarray, int]]:
        generator = self.model.generate_custom_voice(
            text=plan.public_text,
            speaker=plan.speaker,
            language=plan.language,
            instruct=plan.style_instruction,
            temperature=0.78,
            top_k=30,
            top_p=0.92,
            repetition_penalty=1.08,
            max_tokens=2048,
            # Text remains a complete conversational turn. A bounded audio
            # producer/consumer pipeline overlaps synthesis and playback.
            stream=True,
            streaming_interval=0.48,
            verbose=False,
        )
        try:
            for result in generator:
                if cancelled.is_set():
                    break
                samples = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if samples.size:
                    yield samples, int(result.sample_rate)
        finally:
            close = getattr(generator, "close", None)
            if close is not None:
                close()
            decoder = getattr(
                getattr(self.model, "speech_tokenizer", None), "decoder", None
            )
            if decoder is not None and hasattr(decoder, "reset_streaming_state"):
                decoder.reset_streaming_state()
