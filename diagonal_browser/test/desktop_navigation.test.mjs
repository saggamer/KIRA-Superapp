import test from 'node:test';
import assert from 'node:assert/strict';
import { TabNavigation, liveContents, countBlockedRequest } from '../desktop_navigation.mjs';

test('blocked-request accounting tolerates closing, missing and disposed webContents', () => {
  const disposed = { get webContents() { throw new Error('Object destroyed'); } };
  const tabs = [{ view: null, blocked: 0 }, { view: {}, blocked: 0 }, { view: disposed, blocked: 0 },
    { view: { webContents: { isDestroyed: () => true, get id() { throw new Error('Already closed'); } } }, blocked: 0 },
    { view: { webContents: { isDestroyed: () => false, id: 7 } }, blocked: 0 }];
  assert.doesNotThrow(() => countBlockedRequest(tabs, { tabId: 7 }));
  for (const request of [undefined, {}, { tabId: -1 }, { tabId: 999 }]) assert.doesNotThrow(() => countBlockedRequest(tabs, request));
  assert.deepEqual(tabs.map(tab => tab.blocked), [0, 0, 0, 0, 1]);
  assert.equal(liveContents(disposed), null);
});
test('home, cached searches and native pages form one navigable per-tab timeline', () => {
  const navigation = new TabNavigation();
  const searchResults = { query: 'test', mode: 'web', results: [{ title: 'Cached', url: 'https://example.com' }] };
  navigation.surface({ kind: 'search', title: 'Search', searchResults });
  navigation.page({ url: 'https://example.com', viewId: 1, nativeIndex: 0 });
  navigation.page({ url: 'https://example.com/next', viewId: 1, nativeIndex: 1 });
  assert.equal(navigation.move(-1).url, 'https://example.com');
  assert.deepEqual(navigation.move(-1).searchResults, searchResults);
  assert.equal(navigation.move(-1).kind, 'home');
  assert.equal(navigation.canGoBack, false);
  assert.equal(navigation.move(1).kind, 'search');
  assert.equal(navigation.move(1).nativeIndex, 0);
  assert.equal(navigation.move(1).nativeIndex, 1);
  assert.equal(navigation.canGoForward, false);
});
test('page history traversal does not append duplicates and new navigation clears Forward', () => {
  const navigation = new TabNavigation();
  navigation.page({ url: 'https://a.test', viewId: 1, nativeIndex: 0 });
  navigation.page({ url: 'https://b.test', viewId: 1, nativeIndex: 1 });
  navigation.page({ url: 'https://a.test', viewId: 1, nativeIndex: 0 });
  assert.equal(navigation.entries.length, 3); assert.equal(navigation.index, 1);
  navigation.page({ url: 'https://c.test', viewId: 1, nativeIndex: 1 });
  assert.equal(navigation.canGoForward, false);
  assert.equal(navigation.entries.at(-1).url, 'https://c.test');
});
