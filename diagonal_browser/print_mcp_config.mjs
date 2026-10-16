import path from 'node:path';
import { fileURLToPath } from 'node:url';

const server = path.join(path.dirname(fileURLToPath(import.meta.url)), 'server.mjs');
process.stdout.write(JSON.stringify({ mcpServers: { 'diagonal-browser': { command: 'node', args: [server] } } }, null, 2) + '\n');
