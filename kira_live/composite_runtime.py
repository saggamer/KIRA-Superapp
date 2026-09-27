from __future__ import annotations

"""Strict runtime seam for one KIRA Live graph with three weight islands.

The public contracts deliberately contain no prompt or message-handoff method.
The auxiliary transcript can leave the Listener for UI/debugging, but it can
never enter the Thinker or Talker computation.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


Tensor = Any


@dataclass(frozen=True)
class ListenerState:
    hidden_states: Tensor
    unit_ids: Tensor
    auxiliary_transcript: str | None = None


@dataclass(frozen=True)
class ThinkerSeed:
    hidden_states: Tensor
    token_ids: Tensor
    memory_slots: Tensor


@dataclass(frozen=True)
class CompositeBridgeState:
    thinker_hidden: Tensor
    talker_conditioning: Tensor
    codec_conditioning: Tensor
    router_logits: Tensor


@dataclass(frozen=True)
class ThinkerState:
    hidden_states: Tensor
    token_ids: Tensor


@dataclass(frozen=True)
class LiveModelOutput:
    audio: Tensor
    sample_rate: int
    auxiliary_transcript: str | None
    router_logits: Tensor


@runtime_checkable
class ListenerIsland(Protocol):
    def encode(self, audio_samples: Tensor, sample_rate: int) -> ListenerState: ...


@runtime_checkable
class ThinkerIsland(Protocol):
    def seed(self) -> ThinkerSeed: ...

    def decode_hidden(self, fused_hidden: Tensor, token_ids: Tensor) -> ThinkerState: ...


@runtime_checkable
class CompositeBridge(Protocol):
    def fuse(
        self,
        *,
        listener_hidden: Tensor,
        listener_unit_ids: Tensor,
        thinker: ThinkerSeed,
        emotion_features: Tensor,
        talker_code_ids: Tensor,
    ) -> CompositeBridgeState: ...


@runtime_checkable
class TalkerIsland(Protocol):
    def synthesize_hidden(
        self,
        *,
        thinker: ThinkerState,
        semantic_conditioning: Tensor,
        codec_conditioning: Tensor,
    ) -> tuple[Tensor, int]: ...


class KiraLiveCompositeRuntime:
    """Execute one hidden-state transaction across all three weight islands."""

    def __init__(
        self,
        listener: ListenerIsland,
        thinker: ThinkerIsland,
        bridge: CompositeBridge,
        talker: TalkerIsland,
    ) -> None:
        self.listener = listener
        self.thinker = thinker
        self.bridge = bridge
        self.talker = talker

    def respond(
        self,
        audio_samples: Tensor,
        *,
        sample_rate: int,
        emotion_features: Tensor,
        talker_code_ids: Tensor,
    ) -> LiveModelOutput:
        listener_state = self.listener.encode(audio_samples, sample_rate)
        thinker_seed = self.thinker.seed()
        bridge_state = self.bridge.fuse(
            listener_hidden=listener_state.hidden_states,
            listener_unit_ids=listener_state.unit_ids,
            thinker=thinker_seed,
            emotion_features=emotion_features,
            talker_code_ids=talker_code_ids,
        )
        thinker_state = self.thinker.decode_hidden(
            bridge_state.thinker_hidden, thinker_seed.token_ids
        )
        audio, output_rate = self.talker.synthesize_hidden(
            thinker=thinker_state,
            semantic_conditioning=bridge_state.talker_conditioning,
            codec_conditioning=bridge_state.codec_conditioning,
        )
        return LiveModelOutput(
            audio=audio,
            sample_rate=output_rate,
            auxiliary_transcript=listener_state.auxiliary_transcript,
            router_logits=bridge_state.router_logits,
        )
