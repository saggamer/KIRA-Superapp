"""Narrow inference-time expert prior for registered-subagent dispatch.

Only the original user request selects the prior. Tool/page observations never
change expert routing or authorize an action.
"""
import re


def subagent_dispatch(request):
    text = str(request or '')
    command = re.search(r'\b(?:queue|run|delegate|assign)\b', text, re.I)
    target = re.search(r'\b(?:subagent|specialist|agent)\b|\b[a-z][a-z0-9_]*_bot\b|\bregistry confirms\b', text, re.I)
    return bool(command and target)
