"""Strict Spark native tool-call adapter for Kira's bracketed browser bridge."""
import re


BROWSER_TOOL_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'BROWSER_TOOL',
        'description': 'Operate the active Diagonal Browser tab using Kira browser actions.',
        'parameters': {
            'type': 'object',
            'properties': {
                'action': {'type': 'string', 'enum': [
                    'navigate', 'observe', 'act', 'list_tabs', 'select_tab',
                    'back', 'inspect_page', 'search', 'scroll',
                    'search_history', 'read_history', 'revisit_history']},
                'url': {'type': 'string'},
                'tabid': {'type': 'string'},
                'query': {'type': 'string'},
                'history_id': {'type': 'string'},
                'kind': {'type': 'string', 'enum': ['all', 'search', 'page']},
                'source': {'type': 'string', 'enum': ['all', 'llm', 'desktop']},
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50},
                'offset': {'type': 'integer', 'minimum': 0},
                'text_offset': {'type': 'integer', 'minimum': 0},
                'result_offset': {'type': 'integer', 'minimum': 0},
                'result_index': {'type': 'integer', 'minimum': 0},
                'operation': {'type': 'string', 'enum': ['click', 'fill', 'select_option', 'scroll']},
                'pixels': {'type': 'integer'},
                'target_role': {'type': 'string'},
                'target_name': {'type': 'string'},
                'index': {'type': 'integer'},
                'observation_id': {'type': 'string'},
                'value': {'type': 'string'},
                'screenshot': {'type': 'boolean'},
            },
            'required': ['action'],
            'additionalProperties': False,
        },
    },
}

PAIR = re.compile(r'<arg_key>([a-z_]+)</arg_key><arg_value>(.*?)</arg_value>', re.S)
CALL = re.compile(r'\s*<tool_call>([A-Z_]+)(.*?)</tool_call>\s*', re.S)
RAW_CALL = re.compile(r'\s*([A-Z_]+)((?:<arg_key>.*?</arg_key><arg_value>.*?</arg_value>)+)\s*', re.S)


def native_browser_call_to_block(text):
    """Return a validated Kira block, or None for prose/malformed calls."""
    match = CALL.fullmatch(text) or RAW_CALL.fullmatch(text)
    if not match or match[1] != 'BROWSER_TOOL':
        return None
    body = match[2]
    found = list(PAIR.finditer(body))
    if ''.join(item.group(0) for item in found) != body:
        return None
    allowed = BROWSER_TOOL_SCHEMA['function']['parameters']['properties']
    args = {}
    for item in found:
        key, value = item.groups()
        if key not in allowed or key in args or '\n' in value or '\r' in value:
            return None
        args[key] = value
    action = args.get('action', '').lower()
    if action not in allowed['action']['enum']:
        return None
    if action == 'navigate' and not args.get('url'):
        return None
    if action == 'search' and not args.get('query'):
        return None
    if action == 'select_tab' and not args.get('tabid'):
        return None
    if action in {'read_history', 'revisit_history'} and not args.get('history_id'):
        return None
    for key in ('offset', 'text_offset', 'result_offset', 'result_index'):
        if key in args:
            try:
                if int(args[key]) < 0:
                    return None
            except ValueError:
                return None
    if action == 'scroll' or args.get('operation', '').lower() == 'scroll':
        try:
            pixels = int(args.get('pixels', '600'))
        except ValueError:
            return None
        if pixels == 0 or not -1600 <= pixels <= 1600:
            return None
    if action == 'act':
        operation = args.get('operation', '').lower()
        if operation not in allowed['operation']['enum']:
            return None
        if 'index' in args and not args.get('observation_id'):
            return None
        if operation != 'scroll' and not ('index' in args or args.get('target_role') and args.get('target_name')):
            return None
        if operation in {'fill', 'select_option'} and not args.get('value'):
            return None
    return '[BROWSER_TOOL]\n' + '\n'.join(
        f'{key.upper()}: {value}' for key, value in args.items()) + '\n[/BROWSER_TOOL]'
