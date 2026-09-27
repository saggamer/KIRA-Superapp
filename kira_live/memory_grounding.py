from __future__ import annotations

"""Conservative extractive recall for simple, explicit personal facts."""

import re


_QUESTION = re.compile(
    r"^\s*(?:what\s+(?:is|was)|what's|do\s+you\s+remember)\s+my\s+"
    r"(?P<attribute>[a-z][a-z0-9' -]{1,80}?)\s*\??\s*$",
    re.IGNORECASE,
)


def grounded_recall(query: str, memory_context: str) -> str | None:
    """Answer only when a user event explicitly states the requested relation.

    Missing or ambiguous evidence is left to the Thinker. This is a grounding
    failsafe, not a second language model or an instruction from the memory.
    """
    match = _QUESTION.match(str(query or ""))
    if not match:
        return None
    attribute = match.group("attribute").strip()
    pattern = re.compile(
        rf"\bmy\s+{re.escape(attribute)}\s+(?:is|was)\s+([^.!?\n]{{1,100}})",
        re.IGNORECASE,
    )
    facts: list[str] = []
    for line in str(memory_context or "").splitlines():
        if not re.match(r"^\s*-\s*#\d+\s+user:\s*", line, re.IGNORECASE):
            continue
        statement = line.split("user:", 1)[-1]
        found = pattern.search(statement)
        if found:
            value = found.group(1).strip(" \t,;:-")
            if value:
                facts.append(value)
    if not facts:
        return None
    return f"You told me your {attribute} is {facts[-1]}."
