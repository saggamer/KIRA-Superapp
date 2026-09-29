"""Explicit, opt-in expert capacity layouts; legacy daily-driver defaults stay intact."""

AGENTIC_INDICES = (2, 3, 5, 6, 7, 9)
CONVERSATION_INDICES = (0, 1, 4, 8)
FAMILY_INDICES = ((1, 8), AGENTIC_INDICES, (0, 4))
LEGACY_BUDGETS = (0.25, 0.40, 0.35)
SPLIT_BUDGETS = (0.25, 0.50, 0.25)


def expert_layout(rank=128, profile="legacy"):
    if rank <= 0:
        raise ValueError("Expert rank must be positive.")
    if profile == "legacy":
        return (rank,) * 10, LEGACY_BUDGETS
    if profile == "split_50_50":
        if rank != 384:
            raise ValueError("split_50_50 uses conversation rank 384 and agentic rank 768.")
        return tuple(768 if i in AGENTIC_INDICES else 384 for i in range(10)), SPLIT_BUDGETS
    raise ValueError(f"Unknown expert profile: {profile}")


def selected_parameter_budget(width=1024, layers=16, rank=384, profile="split_50_50"):
    ranks, _ = expert_layout(rank, profile)
    family_counts = [layers * 2 * width * sum(sorted((ranks[i] for i in family), reverse=True)[:2])
                     for family in FAMILY_INDICES]
    return {"agentic_matrices": family_counts[1],
            "conversation_emotion_matrices": family_counts[0] + family_counts[2],
            "shared_router_context": layers * 18 * width,
            "total": sum(family_counts) + layers * 18 * width}
