"""Public-output validation for model proposals; never repairs or executes tools."""
import re


TOOL_TAGS = frozenset('''BROWSER_TOOL SUBAGENT_TASK SUBAGENT_REGISTRY SUBAGENT_CREATE
APP_LIST APP_INTEGRATION WINDOW_REPORT SCREENSHOT_CONTEXT SYSTEM_REPORT MEMORY_REPORT
NETWORK_REPORT OS_MAP OPEN CLOSE READ_FILE LIST_DIR SEARCH_FILES FIND_IN_COMPUTER
MCP_DISCOVER MCP_DRAFT MCP_BOOTSTRAP MCP_APPLY MCP_TOOLS MCP_CALL WEB_SEARCH WEB_RESEARCH
WEB_OPEN WEB_FETCH WEB_BROWSE WEB_IMAGE_SEARCH WEB_IMAGE_DOWNLOAD IMAGE_GENERATE IMAGE_EDIT
IMAGE_ANALYZE NATIVE_PPTX NATIVE_PDF NATIVE_DOCX PDF_TO_PPTX SHELL COMMAND TERMINAL
PYTHON APPLESCRIPT WRITE_FILE APPEND_FILE DELETE_PATH MOVE_PATH PROJECT_COMMAND
IDE_OPEN IDE_CONTEXT IDE_PASTE_RUN PROJECT_INDEX SCHEDULE_TASK SPECIALIST_TASK
SPECIALIST_REGISTRY MODEL_SCAN BRANCH_REGISTRY TREE_STATUS TREE_MEMORY SANDBOX_PROFILE
HYBRID_ACTIONS CLIPBOARD_READ CLIPBOARD_WRITE HASH_FILE FILE_COMPARE ARCHIVE_PREVIEW
ZIP_CREATE NOTE_DRAFT CALENDAR_DRAFT REMINDER_DRAFT BLENDER_CONTEXT MCP_SETUP'''.split())
MARKER = re.compile(r'\[/?(' + '|'.join(sorted(TOOL_TAGS, key=len, reverse=True)) + r')\]')
COMPLETE = re.compile(r'\s*\[([A-Z_]+)\][ \t]*\n(?:(.*?)\n)?\[/\1\]\s*', re.S)
REPEATED_LABEL = re.compile(r'\b([A-Z][A-Z0-9_]{2,40}:)(?:\s*\1){5,}')


def repeated_protocol(text):
    """Catch the observed label loop, not ordinary repeated words or code."""
    return bool(REPEATED_LABEL.search(str(text or '')))


def output_issue(text, *, allow_tools=True):
    """Return a short reason for invalid output, excluding its raw contents."""
    value = str(text or '').strip()
    if not value:
        return 'empty'
    if repeated_protocol(value):
        return 'repeated_protocol'
    markers = list(MARKER.finditer(value))
    leading = re.match(r'^\[/?([A-Z][A-Z_]{1,63})\b', value)
    protocol_prefix = leading and (leading[1] in TOOL_TAGS or '_' in leading[1])
    if markers or protocol_prefix:
        matched = COMPLETE.fullmatch(value)
        if not matched:
            return 'malformed_action'
        if matched[1] not in TOOL_TAGS:
            return 'unknown_action'
        # One proposal cannot contain a second action embedded in its body.
        if len(markers) != 2:
            return 'mixed_actions'
        if not allow_tools:
            return 'unexpected_action'
    return None


def browser_progress_summary(controller, reason):
    """Retain real source progress if recovery cannot produce public prose."""
    pages = {}
    for observation in controller.observations:
        url = str(observation.get('url') or '')
        if url.startswith(('http://', 'https://')):
            title = re.sub(r'[\[\]\r\n]', '', str(observation.get('title') or url))[:160]
            pages[url] = title
    lines = [reason, 'I could not verify completion of the full request.']
    if pages:
        lines.append('\nPages reached:')
        lines.extend(f'- {title}: {url}' for url, title in list(pages.items())[-3:])
    return '\n'.join(lines)
