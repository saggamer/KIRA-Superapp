import test from 'node:test';
import assert from 'node:assert/strict';
import { DiagonalBrowser } from '../browser.mjs';

test('preview never launches a browser and never changes action grounding', async () => {
  const browser = new DiagonalBrowser({historyEnabled: false});
  browser.ensure = () => { throw new Error('Preview must not launch'); };
  assert.deepEqual(await browser.preview(), {available: false});
  browser.active = 'fixture';
  browser.lastObservations.set('fixture', 'original-observation');
  browser.previewInfo = {tabId: 'fixture', title: 'Fixture'};
  let shots = 0;
  browser.pages.set('fixture', {
    isClosed: () => false, url: () => 'https://example.test',
    screenshot: async options => {
      shots++;
      assert.equal(options.fullPage, false);
      assert.equal(options.type, 'jpeg');
      assert.equal(options.timeout, 1000);
      return Buffer.from('jpeg-frame');
    },
  });
  const frame = await browser.preview();
  assert.equal(frame.available, true);
  assert.equal(frame.url, 'https://example.test');
  assert.equal(frame.dataUrl, 'data:image/jpeg;base64,' + Buffer.from('jpeg-frame').toString('base64'));
  assert.equal(browser.lastObservations.get('fixture'), 'original-observation');
  assert.equal(shots, 1);
});
