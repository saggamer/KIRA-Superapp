"""Bounded, read-only MCP calls for a Kira-owned, approved app target."""

import asyncio
import json
import os
from pathlib import Path
import re
import shutil


READ_ONLY_NAME = re.compile(r'^(?:get|list|search|read|find|lookup|inspect|query|fetch|retrieve|view|describe)(?:_|$)', re.I)
ENV_REFERENCE = re.compile(r'^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$')


def _server_config(config_path, server_name):
    path = Path(config_path)
    if not path.is_file():
        raise ValueError('No approved Kira MCP app is configured.')
    data = json.loads(path.read_text(encoding='utf-8'))
    servers = data.get('mcpServers') if isinstance(data, dict) else None
    if not isinstance(servers, dict) or server_name not in servers:
        raise ValueError('The requested MCP server is not configured in Kira.')
    config = servers[server_name]
    if not isinstance(config, dict):
        raise ValueError('Invalid MCP server configuration.')
    command, args, variables = config.get('command'), config.get('args', []), config.get('env', {})
    if (not isinstance(command, str) or not command or not shutil.which(command) or
            not isinstance(args, list) or not all(isinstance(item, str) for item in args) or
            not isinstance(variables, dict)):
        raise ValueError('The configured MCP server command is unavailable or malformed.')
    env = {'PATH': os.environ.get('PATH', '')}
    for key, reference in variables.items():
        match = ENV_REFERENCE.fullmatch(reference) if isinstance(reference, str) else None
        if not isinstance(key, str) or not match:
            raise ValueError('MCP environment entries must refer to local variables, not contain secrets.')
        variable = match.group(1) or match.group(2)
        if variable not in os.environ:
            raise ValueError(f'MCP environment variable {variable} is unavailable.')
        env[key] = os.environ[variable]
    return command, args, env


async def _operation(command, args, env, tool_name=None, arguments=None):
    from mcp import Client, StdioServerParameters
    from jsonschema import validate

    params = StdioServerParameters(command=command, args=args, env=env)
    async with Client(params, read_timeout_seconds=15) as client:
        listed = await client.list_tools()
        tools = list(listed.tools)[:32]
        if tool_name is None:
            return {'tools': [{'name': tool.name,
                               'description': str(tool.description or '')[:240],
                               'input_schema': tool.input_schema}
                              for tool in tools]}
        tool = next((item for item in tools if item.name == tool_name), None)
        if tool is None:
            return {'ok': False, 'error': 'The requested tool was not advertised by this MCP server.'}
        annotations = getattr(tool, 'annotations', None)
        read_only_hint = getattr(annotations, 'read_only_hint', None) if annotations else None
        if not (read_only_hint is True or READ_ONLY_NAME.search(tool.name)):
            return {'ok': False, 'error': 'This MCP tool may change external state; Kira has not authorized a live write call.'}
        try:
            validate(arguments, tool.input_schema)
        except Exception as exc:
            return {'ok': False, 'error': 'MCP tool arguments do not match the advertised schema: ' + str(exc)[:200]}
        result = await client.call_tool(tool_name, arguments, read_timeout_seconds=15)
        content = [str(getattr(item, 'text', ''))[:3000] for item in result.content
                   if getattr(item, 'type', '') == 'text']
        return {'tool': tool_name, 'is_error': bool(result.is_error),
                'content': content[:6], 'structured_content': result.structured_content}


def run(config_path, server_name, *, tool_name=None, arguments=None):
    """Start one configured server per call; never keep a background process."""
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', str(server_name)):
        return {'ok': False, 'error': 'Invalid MCP server name.'}
    if tool_name is not None and (not re.fullmatch(r'[A-Za-z0-9._-]{1,100}', str(tool_name)) or
                                 not isinstance(arguments, dict)):
        return {'ok': False, 'error': 'Invalid MCP tool name or arguments.'}
    try:
        command, args, env = _server_config(config_path, server_name)
        value = asyncio.run(asyncio.wait_for(_operation(command, args, env, tool_name, arguments), timeout=25))
        return {'ok': True, 'server': server_name, **value}
    except Exception as exc:
        return {'ok': False, 'server': server_name, 'error': str(exc)[:500]}
