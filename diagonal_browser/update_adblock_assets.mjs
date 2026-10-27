import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { bundledSnapshot, downloadAdblockSnapshot } from './adblock.mjs';

const bytes = await downloadAdblockSnapshot();
await mkdir(path.dirname(bundledSnapshot), { recursive: true });
await writeFile(bundledSnapshot, bytes);
const { networkRules, cosmeticRules, sources, updatedAt } = JSON.parse(gunzipSync(bytes));
console.log(JSON.stringify({ bytes: bytes.length, networkRules, cosmeticRules, sources: sources.map(item => item.url), updatedAt }));
