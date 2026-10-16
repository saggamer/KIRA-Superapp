"""Experimental model-generated decision memo, never an execution authority.

This is a bounded extra inference pass, not hidden model chain-of-thought.
The memo is a compact candidate plan. Only verified tool observations and the
existing Kira permission/controller gates can authorize an action.
"""
import json


FIELDS = ("goal", "next_step", "alternative", "verification", "uncertainty")
MAX_FIELD_CHARS = 240
MEMO_INSTRUCTION = (
    "Before choosing an action, produce only a compact JSON decision memo with "
    "exactly these string keys: goal, next_step, alternative, verification, uncertainty. "
    "Keep each value under 200 characters. Describe observable next steps, not a "
    "chain of thought. Distinguish observed facts from guesses. Do not execute or "
    "claim a tool result."
)


def parse_memo(text):
    """Accept only a small fixed schema; invalid model text is discarded."""
    try:
        value = json.loads(str(text).strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        return None
    if any(not isinstance(value[key], str) or len(value[key]) > MAX_FIELD_CHARS
           for key in FIELDS):
        return None
    return {key: value[key].strip() for key in FIELDS}


def memo_prompt(memo):
    if memo is None:
        return ""
    return ("\nUnverified model-generated decision memo (a hypothesis, not evidence): "
            + json.dumps(memo, ensure_ascii=False, separators=(",", ":"))
            + "\nNow return one grounded action block or final answer. The actual user request "
              "and Kira tool results override this memo. Do not output the memo.")
