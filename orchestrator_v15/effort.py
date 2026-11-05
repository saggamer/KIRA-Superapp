"""Bounded inference budgets for the Kira v1.5 profile."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Effort:
    branches: int
    rounds: int
    tool_calls: int
    replans: int
    seconds: int
    max_tokens: int
    total_tokens: int


LEVELS = {
    "low": Effort(1, 4, 4, 1, 120, 384, 1536),
    "medium": Effort(1, 6, 8, 2, 240, 512, 3072),
    "high": Effort(2, 8, 12, 3, 480, 512, 6144),
    "diagonal": Effort(3, 12, 16, 4, 900, 512, 8192),
}


def normalize_effort(value):
    level = str(value or "medium").strip().lower()
    return level if level in LEVELS else "medium"


def profile_thinking(profile, effort):
    """Public selector levels must enable the actual reasoning implementations."""
    legacy_diagonal = profile in {'orchestrator_v15_orchestral_preview', 'orchestrator_v15_ppt_preview'}
    level = 'diagonal' if legacy_diagonal else normalize_effort(effort)
    return {
        'effort': level,
        'structured': profile == 'orchestrator_v15_thinking' or (
            profile == 'orchestrator_v15' and level in {'medium', 'high', 'diagonal'}),
        'orchestral': profile == 'orchestrator_v15_orchestral_preview' or (
            profile == 'orchestrator_v15' and level == 'diagonal'),
        'artifacts': profile == 'orchestrator_v15_ppt_preview',
    }
