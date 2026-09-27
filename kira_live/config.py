from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelRef:
    model_id: str
    runtime: str
    role: str
    optional: bool = False


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    frame_ms: int = 20
    pre_roll_ms: int = 320
    start_speech_ms: int = 80
    end_silence_ms: int = 520
    max_utterance_seconds: float = 45.0
    vad_start_threshold: float = 0.62
    vad_end_threshold: float = 0.38
    emotion_window_seconds: float = 3.0

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate * self.frame_ms // 1_000


@dataclass(frozen=True)
class RouterConfig:
    experts: tuple[str, ...] = (
        "conversation",
        "emotion_prosody",
        "agent_planning",
        "tool_use",
        "episodic_memory",
        "workflow_state",
        "safety_permissions",
        "failure_recovery",
        "speech_style",
        "verification_completion",
    )
    top_k: int = 2
    temperature: float = 0.8
    emotion_floor: float = 0.08
    safety_floor: float = 0.06


@dataclass(frozen=True)
class MemoryConfig:
    exact_event_log: bool = True
    recent_exact_turns: int = 48
    retrieval_events: int = 48
    # Roughly 12k text tokens: substantially more continuity than the old
    # budget while retaining enough headroom for low-latency live decoding.
    prompt_budget_chars: int = 48_000
    recurrent_memory_slots: int = 256
    workflow_ledger_enabled: bool = True
    persist_raw_audio: bool = False


@dataclass(frozen=True)
class KiraLiveConfig:
    """Stable contract shared by the prototype, training, and native runtimes."""

    version: str = "kira-live-1"
    thinker: ModelRef = field(
        default_factory=lambda: ModelRef(
            "Qwen/Qwen3.5-0.8B", "mlx-vlm", "text_and_agentic_thinker"
        )
    )
    listener: ModelRef = field(
        default_factory=lambda: ModelRef(
            "Qwen/Qwen3-ASR-0.6B-hf", "coreml_or_mlx", "streaming_asr"
        )
    )
    talker: ModelRef = field(
        default_factory=lambda: ModelRef(
            "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            "mlx",
            "integrated_emotional_streaming_talker",
        )
    )
    vad: ModelRef = field(
        default_factory=lambda: ModelRef(
            "TEN-framework/ten-vad", "coreml", "always_on_voice_activity"
        )
    )
    emotion_encoder: ModelRef = field(
        default_factory=lambda: ModelRef(
            "kira-live/acoustic-emotion-small", "coreml", "tone_and_emotion"
        )
    )
    audio: AudioConfig = field(default_factory=AudioConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mtp_enabled: bool = True
    store_raw_audio: bool = False
    local_only: bool = True

    def validate(self) -> None:
        if self.audio.samples_per_frame <= 0:
            raise ValueError("Audio frame size must be positive.")
        if not 0 < self.router.top_k <= len(self.router.experts):
            raise ValueError("top_k must select at least one configured expert.")
        if not 0 <= self.audio.vad_end_threshold < self.audio.vad_start_threshold <= 1:
            raise ValueError("VAD thresholds must provide positive hysteresis.")
        if self.memory.recurrent_memory_slots <= 0:
            raise ValueError("At least one recurrent memory slot is required.")
