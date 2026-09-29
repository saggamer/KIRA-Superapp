from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any

from .config import KiraLiveConfig
from .emotion import EmotionState, EmotionTracker
from .router import ExpertRoute, LiveExpertRouter
from .workflow_memory import PersistentWorkflowMemory


class LiveState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class LiveEvent:
    kind: str
    state: LiveState
    created_at: float
    payload: dict[str, Any] = field(default_factory=dict)


class KiraLiveSession:
    """Dependency-free control plane for the realtime audio/model pipeline."""

    def __init__(
        self,
        config: KiraLiveConfig | None = None,
        *,
        memory: PersistentWorkflowMemory | None = None,
        chat_id: str | None = None,
        workflow_id: str | None = None,
    ):
        self.config = config or KiraLiveConfig()
        self.config.validate()
        self.state = LiveState.IDLE
        self.emotions = EmotionTracker()
        self.router = LiveExpertRouter(self.config.router)
        self.events: list[LiveEvent] = []
        self.last_route: ExpertRoute | None = None
        self.memory = memory
        self.chat_id = str(chat_id or "")
        self.workflow_id = workflow_id

    def _emit(self, kind: str, **payload: Any) -> LiveEvent:
        event = LiveEvent(kind, self.state, time.time(), payload)
        self.events.append(event)
        self.events = self.events[-200:]
        return event

    def start(self) -> LiveEvent:
        self.state = LiveState.LISTENING
        return self._emit(
            "listening_started",
            sample_rate=self.config.audio.sample_rate,
            frame_ms=self.config.audio.frame_ms,
            raw_audio_persisted=self.config.store_raw_audio,
        )

    def voice_started(self, probability: float) -> LiveEvent:
        if self.state == LiveState.SPEAKING:
            self.state = LiveState.INTERRUPTED
            self._emit("barge_in", probability=float(probability))
        self.state = LiveState.LISTENING
        return self._emit("utterance_started", probability=float(probability))

    def update_emotion(self, state: EmotionState) -> LiveEvent:
        return self._emit(
            "emotion_updated",
            label=state.label,
            confidence=state.confidence,
            valence=state.valence,
            arousal=state.arousal,
        )

    def voice_ended(self) -> LiveEvent:
        self.state = LiveState.TRANSCRIBING
        return self._emit("utterance_ended")

    def transcript_ready(self, text: str) -> LiveEvent:
        cleaned = str(text or "").strip()
        if not cleaned:
            self.state = LiveState.LISTENING
            return self._emit("empty_transcript")
        self.last_route = self.router.route(cleaned, self.emotions.state())
        memory_context = ""
        if self.memory is not None and self.chat_id:
            self.memory.append(
                self.chat_id,
                "user",
                cleaned,
                workflow_id=self.workflow_id,
            )
            memory_context = self.memory.build_context(
                self.chat_id,
                cleaned,
                workflow_id=self.workflow_id,
                recent=self.config.memory.recent_exact_turns,
                retrieved=self.config.memory.retrieval_events,
                max_chars=self.config.memory.prompt_budget_chars,
            )
        self.state = LiveState.THINKING
        return self._emit(
            "transcript_ready",
            text=cleaned,
            emotion_context=self.emotions.state().prompt_context(),
            memory_context=memory_context,
            experts=self.last_route.weights,
        )

    def response_started(self) -> LiveEvent:
        self.state = LiveState.SPEAKING
        return self._emit("response_started")

    def response_finished(self, text: str = "") -> LiveEvent:
        if self.memory is not None and self.chat_id and str(text).strip():
            self.memory.append(
                self.chat_id,
                "assistant",
                str(text).strip(),
                workflow_id=self.workflow_id,
            )
        self.state = LiveState.LISTENING
        return self._emit("response_finished")

    def response_interrupted(self, text: str = "") -> LiveEvent:
        if self.memory is not None and self.chat_id and str(text).strip():
            self.memory.append(
                self.chat_id, "assistant", str(text).strip(),
                kind="interrupted_response", workflow_id=self.workflow_id,
            )
        # Do not overwrite LISTENING/TRANSCRIBING: barge-in already moved the
        # controller to the user's new utterance.
        return self._emit("response_interrupted")

    def stop(self) -> LiveEvent:
        self.state = LiveState.IDLE
        return self._emit("session_stopped")
