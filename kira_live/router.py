from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping

from .config import RouterConfig
from .emotion import EmotionState


@dataclass(frozen=True)
class ExpertRoute:
    weights: dict[str, float]
    reasons: tuple[str, ...]

    @property
    def active_experts(self) -> tuple[str, ...]:
        return tuple(self.weights)


class LiveExpertRouter:
    """Reference router for token-level residual adapters.

    Training replaces these lexical priors with a learned gate. Keeping the
    deterministic policy in the runtime makes routing testable and provides a
    safe fallback when a gate checkpoint is unavailable.
    """

    AGENTIC = re.compile(
        r"\b(?:build|fix|edit|run|test|search|open|create|download|schedule|file|code)\b",
        re.IGNORECASE,
    )
    RISK = re.compile(
        r"\b(?:delete|remove|password|credential|payment|publish|send|install|permission)\b",
        re.IGNORECASE,
    )

    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()

    def route(
        self,
        text: str,
        emotion: EmotionState,
        learned_logits: Mapping[str, float] | None = None,
    ) -> ExpertRoute:
        scores = {name: 0.0 for name in self.config.experts}
        scores["conversation"] = 1.0
        reasons = []

        if self.AGENTIC.search(text or ""):
            scores["agent_planning"] += 1.0
            scores["tool_use"] += 1.65
            scores["workflow_state"] += 0.8
            reasons.append("action_intent")
        if self.RISK.search(text or ""):
            scores["safety_permissions"] += 2.1
            reasons.append("sensitive_or_mutating_intent")

        emotional_mass = 1.0 - emotion.probabilities.get("neutral", 0.0)
        if emotion.confidence >= 0.30 and emotional_mass >= 0.20:
            scores["emotion_prosody"] += 0.7 + emotional_mass + 0.5 * emotion.arousal
            reasons.append("reliable_acoustic_cue")
        scores["emotion_prosody"] += self.config.emotion_floor
        scores["safety_permissions"] += self.config.safety_floor

        if learned_logits:
            for name in scores:
                scores[name] += float(learned_logits.get(name, 0.0))
            reasons.append("learned_token_gate")

        temperature = max(0.05, self.config.temperature)
        maximum = max(scores.values())
        probabilities = {
            name: math.exp((score - maximum) / temperature)
            for name, score in scores.items()
        }
        selected = sorted(probabilities, key=probabilities.get, reverse=True)[: self.config.top_k]
        total = sum(probabilities[name] for name in selected)
        weights = {name: probabilities[name] / total for name in selected}
        return ExpertRoute(weights=weights, reasons=tuple(reasons or ["default_conversation"]))
