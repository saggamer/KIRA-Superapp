import test from 'node:test';
import assert from 'node:assert/strict';
import { inspectUrl } from '../page_inspection.mjs';

test('URL inspection exposes structure and technical flags', () => {
  const info = inspectUrl('http://example.com/path?utm_source=x&k=1#section');
  assert.equal(info.hostname, 'example.com');
  assert.deepEqual(info.queryKeys, ['utm_source', 'k']);
  assert.deepEqual(info.flags, ['Unencrypted HTTP']);
});
