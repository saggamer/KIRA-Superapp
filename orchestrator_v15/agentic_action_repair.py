"""Narrow Kira action-shape repairs grounded in an explicit user request."""
import re

from .contracts import block, parse_block


def _registry_request(request):
    return bool(re.search(r'\b(?:registered|registry)\b.{0,70}\b(?:agents?|subagents?|specialists?)\b|'
                   r'\b(?:agents?|subagents?|specialists?)\b.{0,70}\b(?:registered|registry)\b',
                   request, re.I))


def _app_inventory_request(request):
    return bool(re.search(r'\b(?:list|show|which|what|find)\b.{0,70}'
                   r'\b(?:available|installed)\s+apps?\b|'
                   r'\b(?:list|show|which|what|find)\b.{0,70}'
                   r'\bapps?\s+(?:are\s+)?(?:available|installed)\b', request, re.I))


def _app_context_name(request, fields):
    if not re.search(r'\b(?:inspect|read|show)\b.{0,90}\bintegration context\b', request, re.I):
        return None
    match = (re.search(r'\b([A-Z][A-Za-z0-9.+_-]{1,30})\s+(?:app\s+)?integration context\b', request)
             or re.search(r'\bintegration context\s+for\s+([A-Z][A-Za-z0-9.+_-]{1,30})\b', request))
    return match.group(1) if match else fields.get('APP') or fields.get('TARGET')


def repair_action(answer, request):
    """Return a repaired block only when the user intent makes its scope exact."""
    parsed = parse_block(answer)
    tag, fields = parsed if parsed else (None, {})
    if _registry_request(request) and (tag in {'APP_LIST', 'MCP_DISCOVER', 'BROWSER_TOOL'} or
                                       (tag is None and '[' in answer)):
        return block('SUBAGENT_REGISTRY')
    if _app_inventory_request(request) and tag in {'BROWSER_TOOL', 'MCP_DISCOVER'}:
        return block('APP_LIST')
    app_name = _app_context_name(request, fields)
    if app_name and tag in {'BROWSER_TOOL', 'APP_INTEGRATION'}:
        return block('APP_INTEGRATION', app=app_name, action='context')
    if tag == 'SUBAGENT_TASK' and fields.get('AGENT') and fields.get('TASK'):
        queued = bool(re.search(r'\bqueue\b|\bwithout running\b|\bdo not run\b', request, re.I))
        run_now = bool(re.search(r'\brun\b.{0,45}\bnow\b|\brun\s+[A-Za-z0-9_-]+\s+now\b', request, re.I))
        if queued or run_now:
            expected = 'false' if queued else 'true'
            if fields.get('RUN', '').lower() != expected:
                return block('SUBAGENT_TASK', **{**fields, 'RUN': expected})
    return answer


def direct_result(tag, request, result):
    """End simple Kira reads/delegations from observed tool evidence."""
    result = str(result)
    if tag == 'APP_INTEGRATION' and _app_context_name(request, {}):
        return result
    if tag == 'SUBAGENT_REGISTRY' and _registry_request(request):
        return result
    if tag == 'SUBAGENT_TASK' and result.startswith(('SUBAGENT_TASK:', 'SUBAGENT_TASK ERROR:')):
        return result
    return None
