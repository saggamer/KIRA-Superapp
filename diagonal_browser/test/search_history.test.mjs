import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import os from 'node:os';
import path from 'node:path';
import { SearchHistory, historyUrl } from '../search_history.mjs';

test('history persists, filters cached content, paginates, and supports independent writers', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-history-'));
  try {
    const history = new SearchHistory(dir);
    const otherWriter = new SearchHistory(dir);
    const entries = await Promise.all(Array.from({ length: 24 }, (_, index) => (index % 2 ? history : otherWriter).save({
      kind: 'search', source: index % 2 ? 'llm' : 'desktop', query: `Research ${index}`,
      results: [{ title: 'Neutron star', url: 'https://example.com/article', snippet: 'Magnetic astronomy evidence' }],
    })));
    const restarted = new SearchHistory(dir);
    const first = await restarted.search({ query: 'magnetic astronomy', source: 'llm', limit: 5 });
    assert.equal(first.total, 12); assert.equal(first.entries.length, 5); assert.equal(first.nextOffset, 5);
    const second = await restarted.search({ query: 'magnetic astronomy', source: 'llm', offset: first.nextOffset, limit: 5 });
    assert.equal(second.entries.some(item => first.entries.some(previous => previous.historyId === item.historyId)), false);
    assert.equal((await restarted.search({ query: 'unseen query' })).total, 0);
    assert.equal((await stat(path.join(dir, 'search_history', `${entries[0].id}.json`))).mode & 0o777, 0o600);
    const page = await restarted.save({ kind: 'page', source: 'llm', url: 'https://example.com/page', title: 'Long source', text: 'a'.repeat(7000) + 'unique lower evidence' + 'b'.repeat(20000) });
    assert.equal((await restarted.search({ query: 'unique lower evidence', kind: 'page' })).total, 1);
    const chunk = await restarted.read(page.id);
    assert.equal(chunk.cached, true); assert.equal(chunk.textLength, 25000); assert.equal(chunk.textTruncated, true);
    assert.equal(chunk.text.length, 5000); assert.equal(chunk.nextTextOffset, 5000);
    assert.match((await restarted.read(page.id, { textOffset: 5000 })).text, /unique lower evidence/);
    await assert.rejects(restarted.get('../../settings.json'), /Invalid history ID/);
    await assert.rejects(restarted.read(page.id, { textOffset: -1 }), /nonnegative/);
    await writeFile(path.join(dir, 'search_history', '00000000-0000-0000-0000-000000000000.json'), '{partial');
    assert.equal((await restarted.search()).total, 25);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('history sanitizes credential URLs and keeps stable result indices across chunks', async () => {
  assert.equal(historyUrl('javascript:alert(1)'), '');
  assert.equal(historyUrl('https://user:password@example.com/path?q=stars&code=secret&access_token=private#id_token=secret'), 'https://example.com/path?q=stars');
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-history-'));
  try {
    const history = new SearchHistory(dir);
    const entry = await history.save({ kind: 'search', query: 'images', mode: 'images', results: Array.from({ length: 24 }, (_, index) => ({ title: `Image ${index}`, url: `https://example.com/${index}`, thumbnail: 'https://example.com/thumb', image: 'https://example.com/image' })) });
    const first = await history.read(entry.id);
    assert.equal(first.results.length, 10); assert.equal(first.nextResultOffset, 10);
    const second = await history.read(entry.id, { resultOffset: 10 });
    assert.equal(second.results[0].resultIndex, 10); assert.equal(second.results[0].title, 'Image 10');
    assert.equal(second.results[0].image, 'https://example.com/image');
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('Python web research records are readable by the same Node history tools', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-history-python-'));
  try {
    await promisify(execFile)('python3', ['-c', 'from diagonal_browser.search_history import save_history; save_history(kind="search", query="shared Python research", results=[{"title":"A source", "url":"https://example.com/?token=private", "snippet":"cached Python evidence"}])'], { cwd: path.resolve(import.meta.dirname, '../..'), env: { ...process.env, DIAGONAL_PROFILE_DIR: dir } });
    const history = new SearchHistory(dir);
    const result = await history.search({ query: 'cached Python evidence', source: 'llm' });
    assert.equal(result.total, 1);
    const entry = await history.read(result.entries[0].historyId);
    assert.equal(entry.query, 'shared Python research'); assert.equal(entry.results[0].url, 'https://example.com/');
  } finally { await rm(dir, { recursive: true, force: true }); }
});
