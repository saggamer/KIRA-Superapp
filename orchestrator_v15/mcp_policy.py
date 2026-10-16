"""Small, auditable scope checks for Kira's MCP configuration workflow."""
import re


def apply_requested(request):
    """A draft or inspection request alone does not authorize config application."""
    request = str(request or '')
    if re.search(r'\b(?:do not|don\'t|never|without|not yet|ask (?:me )?before)\b'
                 r'.{0,100}\b(?:apply|enable|activate|connect|install)\b', request, re.I):
        return False
    return bool(re.search(
        r'(?:^\s*|[.!?]\s*|\b(?:please|then|now|go ahead and|'
        r'i want (?:you )?to|can you|could you)\s+)'
        r'(?:apply|enable|activate|connect|install)\b', request, re.I))


def live_call_requested(request):
    """Recognize requests that need an MCP server tool, not config management."""
    request = str(request or '')
    return bool(re.search(r'\b(?:tools/call|mcp_call|mcp_tool)\b', request, re.I) or
                re.search(r'\b(?:use|call|invoke|run)\b.{0,90}\bMCP\b.{0,90}\btool\b', request, re.I) or
                re.search(r'\b(?:use|search|read|query)\b.{0,70}\b(?:connected app|mcp app|mcp connector)\b',
                          request, re.I))


def normalize_live_call(proposal, server, advertised_tools):
    """Repair one observed wrong wrapper without inventing tools or arguments."""
    match = re.fullmatch(r'\s*\[BROWSER_TOOL\]\s*\nMCP_CALL\s*\n'
                         r'TARGET:\s*([^\n]+)\nTOOL:\s*([^\n]+)\n'
                         r'ARGS_JSON:\s*([^\n]+)\n\[/BROWSER_TOOL\]\s*',
                         str(proposal or ''))
    if not match or match.group(1).strip() != server or match.group(2).strip() not in advertised_tools:
        return proposal
    import json
    try:
        arguments = json.loads(match.group(3))
    except ValueError:
        return proposal
    if not isinstance(arguments, dict):
        return proposal
    return ('[MCP_CALL]\nSERVER: ' + server + '\nTOOL: ' + match.group(2).strip() +
            '\nARGS_JSON: ' + json.dumps(arguments, separators=(',', ':')) + '\n[/MCP_CALL]')


def broaden_empty_search(tool_name, arguments, result):
    """One lexical retry for a read-only search that returned no content."""
    if not str(tool_name).lower().startswith('search_') or not isinstance(arguments, dict):
        return None
    content = result.get('content') if isinstance(result, dict) else None
    if not isinstance(content, list) or any(str(item).strip() for item in content):
        return None
    query = arguments.get('query')
    if not isinstance(query, str):
        return None
    words = re.findall(r'[\w-]+', query)
    generic = {'note', 'notes', 'reminder', 'reminders', 'message', 'messages',
               'about', 'for', 'from', 'with', 'the', 'my', 'a', 'an'}
    reduced = [word for word in words if word.casefold() not in generic]
    if not reduced or len(reduced) == len(words):
        return None
    replacement = dict(arguments)
    replacement['query'] = ' '.join(reduced)
    return replacement
