import test from 'node:test';
import assert from 'node:assert/strict';
import { AiSession, parseAiAction } from '../ai_session.mjs';

function fixture(actions) {
  const calls = [];
  const ai = new AiSession({ model: async () => JSON.stringify(actions.shift() || { action: 'answer', text: 'Finished' }), observe: async () => ({ url: 'https://example.com' }), act: async action => { calls.push(action); return { summary: 'Used current page' }; }, research: async action => { calls.push(action); return { text: 'New source' }; } });
  ai.open(); return { ai, calls };
}
test('current-page actions work and search waits for single-use permission', async () => {
  const { ai, calls } = fixture([{ action: 'click', element: 1 }, { action: 'search', query: 'more details', reason: 'Compare sources' }]);
  await ai.send('Investigate this page');
  assert.equal(calls.length, 1); assert.equal(ai.state.busy, false);
  const id = ai.state.pending.id;
  await assert.rejects(ai.decide('forged', true));
  assert.equal(calls.length, 1);
  await ai.decide(id, true);
  assert.equal(calls[1].action, 'search'); assert.equal(ai.state.pending, null);
  await assert.rejects(ai.decide(id, true));
});
test('declining research never invokes browser research', async () => {
  const { ai, calls } = fixture([{ action: 'open', url: 'https://example.org' }]);
  await ai.send('Compare'); await ai.decide(ai.state.pending.id, false);
  assert.deepEqual(calls, []); assert.equal(ai.state.messages.at(-1).text, 'Finished');
});
test('closing AI mode revokes pending permissions and ignores late model actions', async () => {
  let finish; let acted = false;
  const ai = new AiSession({ model: () => new Promise(resolve => { finish = resolve; }), observe: async () => ({}), act: async () => { acted = true; }, research: async () => {} });
  ai.open(); const running = ai.send('Click');
  await new Promise(resolve => setImmediate(resolve));
  ai.close(); finish('{"action":"click","element":1}'); await running;
  assert.equal(acted, false); assert.equal(ai.state.open, false);
  ai.open(); ai.request({ action: 'search', query: 'x' }); const id = ai.state.pending.id; ai.close();
  await assert.rejects(ai.decide(id, true));
});
test('model cannot execute arbitrary code or open privileged protocols', () => {
  for (const action of [{ action: 'evaluate', code: 'process.exit()' }, { action: 'open', url: 'file:///etc/passwd' }, { action: 'open', url: 'https://user:secret@example.com' }, { action: 'click', element: '1' }]) assert.throws(() => parseAiAction(JSON.stringify(action)));
});
test('a site permission request pauses inference and drops its late response', async () => {
  let finish; let acted = false;
  const ai = new AiSession({ model: () => new Promise(resolve => { finish = resolve; }), observe: async () => ({}), act: async () => { acted = true; }, research: async () => ({}) });
  ai.open(); const running = ai.send('Use this page'); await new Promise(resolve => setImmediate(resolve));
  ai.request({ action: 'open', url: 'https://example.org', reason: 'Page opened a source' });
  assert.equal(ai.state.busy, false); assert.equal(ai.state.status, 'Waiting for permission');
  finish('{"action":"click","element":1}'); await running;
  assert.equal(acted, false); assert.equal(ai.state.pending.url, 'https://example.org');
});
