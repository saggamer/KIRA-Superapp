"""Isolated Notes-like app used only for MCP transport tests."""
from mcp.server import MCPServer

server = MCPServer('Kira test notes')


@server.tool()
def search_notes(query: str) -> str:
    """Search the user's test notes by text."""
    notes = {'trip': 'Pack a charger and passport.',
             'shopping': 'Buy coffee and oat milk.'}
    return '\n'.join(f'{title}: {text}' for title, text in notes.items()
                     if query.casefold() in (title + ' ' + text).casefold())


@server.tool()
def delete_note(title: str) -> str:
    """Delete a note; the Kira read-only bridge must block this tool."""
    return f'Deleted {title}'


if __name__ == '__main__':
    server.run('stdio')
