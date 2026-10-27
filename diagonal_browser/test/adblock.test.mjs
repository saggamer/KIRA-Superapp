import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { AdBlocker } from '../adblock.mjs';
import { DiagonalBrowser } from '../browser.mjs';

test('bundled lists block real ad/tracker patterns offline and preserve first-party navigation', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-shield-'));
  try {
    const blocker = await new AdBlocker({ profileDir: dir }).init();
    assert.ok(blocker.status().networkRules > 100000);
    assert.ok(blocker.status().cosmeticRules > 30000);
    const ad = await blocker.check('https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js', 'https://example.com/', 'script');
    assert.equal(ad.blocked, true); assert.ok(ad.rule); assert.equal(ad.networkRequestMade, false);
    assert.equal((await blocker.check('https://www.google-analytics.com/analytics.js', 'https://example.com/', 'script')).blocked, true);
    assert.equal((await blocker.check('https://example.com/main.js', 'https://example.com/', 'script')).blocked, false);
    assert.equal((await blocker.check('https://doubleclick.net/', 'https://example.com/', 'main_frame')).blocked, false);
    assert.equal(blocker.status().blocked, 0);
    await blocker.rule('add', '||diagonal-fixture.test^$script');
    assert.equal((await blocker.check('https://diagonal-fixture.test/ad.js', 'https://example.com/')).blocked, true);
    await blocker.rule('add', '@@||diagonal-fixture.test^$script');
    assert.equal((await blocker.check('https://diagonal-fixture.test/ad.js', 'https://example.com/')).blocked, false);
    await blocker.rule('remove', '@@||diagonal-fixture.test^$script');
    assert.equal((await blocker.check('https://diagonal-fixture.test/ad.js', 'https://example.com/')).blocked, true);
    assert.throws(() => blocker.rule('add', '! this is a comment'), /Invalid/);
    assert.throws(() => blocker.rule('add', 'one\ntwo'), /one filter/);
    await blocker.configure({ site: 'example.com', siteEnabled: false });
    assert.equal((await blocker.check('https://diagonal-fixture.test/ad.js', 'https://example.com/')).blocked, false);
    await blocker.configure({ site: 'docs.example.com', siteEnabled: true });
    assert.equal(blocker.status('https://docs.example.com/').activeOnSite, true);
    assert.equal(blocker.status('https://other.example.com/').activeOnSite, false);
    const restarted = await new AdBlocker({ profileDir: dir }).init();
    assert.equal(restarted.status('https://example.com').activeOnSite, false);
    assert.ok(restarted.status().customRules.includes('||diagonal-fixture.test^$script'));
    await Promise.all([blocker.rule('add', '||concurrent-one.test^'), restarted.rule('add', '||concurrent-two.test^')]);
    assert.ok(blocker.status().customRules.includes('||concurrent-one.test^'));
    assert.ok(blocker.status().customRules.includes('||concurrent-two.test^'));
    await blocker.configure({ site: 'example.com', siteEnabled: true });
    assert.equal(restarted.status('https://example.com').activeOnSite, true);
    const previous = blocker.snapshot.updatedAt;
    const originalFetch = globalThis.fetch;
    try { globalThis.fetch = async () => { throw new Error('offline fixture'); }; await assert.rejects(blocker.update(), /offline fixture/); }
    finally { globalThis.fetch = originalFetch; }
    assert.equal(blocker.snapshot.updatedAt, previous);
    assert.equal((await blocker.check('https://diagonal-fixture.test/ad.js', 'https://example.com/')).blocked, true);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('legacy desktop pause migrates and cosmetic exceptions override hide rules', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-shield-legacy-'));
  try {
    await writeFile(path.join(dir, 'settings.json'), JSON.stringify({ adBlockEnabled: false }));
    const blocker = await new AdBlocker({ profileDir: dir }).init();
    assert.equal(blocker.status().enabled, false);
    await blocker.configure({ enabled: true });
    await blocker.rule('add', 'example.com##.diagonal-fixture-ad');
    const details = { url: 'https://example.com/', hostname: 'example.com', domain: 'example.com' };
    assert.match(blocker.cosmetics(details).styles, /diagonal-fixture-ad/);
    await blocker.rule('add', 'example.com#@#.diagonal-fixture-ad');
    assert.equal(blocker.cosmetics(details).styles.includes('diagonal-fixture-ad'), false);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('Playwright actually blocks requests and hides dynamic ads under strict CSP, with reversible site pauses', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-shield-browser-'));
  let adHits = 0, usefulHits = 0;
  const server = http.createServer((req, res) => {
    if (req.url === '/diagonal-fixture-ad.js') { adHits++; res.setHeader('content-type', 'application/javascript'); res.end('window.adRan=true;'); return; }
    if (req.url === '/useful.js') { usefulHits++; res.setHeader('content-type', 'application/javascript'); res.end(`window.usefulRan=true;setTimeout(()=>{const ad=document.createElement('div');ad.className='diagonal-fixture-ad';ad.textContent='Late sponsored slot';document.body.append(ad);},600);`); return; }
    res.setHeader('content-type', 'text/html');
    res.setHeader('content-security-policy', "default-src 'self'; script-src 'self'; style-src 'self'");
    res.end('<title>Shield fixture</title><h1>Useful content</h1><ins class="adsbygoogle" data-ad-client="fixture-client">Built-in ad slot</ins><div class="diagonal-fixture-ad">Sponsored slot</div><script src="/diagonal-fixture-ad.js"></script><script src="/useful.js"></script>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://shield.localhost:${server.address().port}/`;
  const browser = new DiagonalBrowser({ profileDir: dir, historyEnabled: false });
  try {
    await browser.adblockRule('add', '/diagonal-fixture-ad.js$script');
    await browser.adblockRule('add', 'shield.localhost##.diagonal-fixture-ad');
    const visit = await browser.navigate(url);
    const page = browser.pages.get(visit.tabId);
    await page.waitForFunction(() => document.querySelector('.diagonal-fixture-ad') && getComputedStyle(document.querySelector('.diagonal-fixture-ad')).display === 'none');
    assert.equal(await page.locator('.adsbygoogle').isVisible(), false);
    await page.waitForFunction(() => document.querySelectorAll('.diagonal-fixture-ad').length === 2);
    assert.equal(await page.locator('.diagonal-fixture-ad').nth(1).isVisible(), false);
    assert.equal(adHits, 0); assert.ok(usefulHits > 0); assert.equal(await page.evaluate(() => window.usefulRan), true);
    const state = await browser.adblockStatus(url); assert.ok(state.blocked > 0); assert.match(state.recent[0].rule, /diagonal-fixture-ad/);
    const recording = await browser.recordStart(visit.tabId);
    await browser.pages.get(recording.tabId).waitForFunction(() => getComputedStyle(document.querySelector('.diagonal-fixture-ad')).display === 'none');
    assert.equal(adHits, 0);
    const video = await browser.recordStop(); await rm(video.path, { force: true });
    await browser.adblockSet({ site: 'shield.localhost', siteEnabled: false });
    await browser.reload(visit.tabId);
    assert.ok(adHits > 0); assert.equal(await page.locator('.diagonal-fixture-ad').first().isVisible(), true);
    await browser.adblockSet({ site: 'shield.localhost', siteEnabled: true });
    const hits = adHits; await browser.reload(visit.tabId);
    await page.waitForFunction(() => getComputedStyle(document.querySelector('.diagonal-fixture-ad')).display === 'none');
    assert.equal(adHits, hits);
    const popupReady = page.waitForEvent('popup');
    await page.evaluate(url => window.open(url, 'shield-test-popup'), url);
    const popup = await popupReady; await popup.waitForLoadState('domcontentloaded');
    await popup.waitForFunction(() => getComputedStyle(document.querySelector('.diagonal-fixture-ad')).display === 'none');
    assert.equal(adHits, hits); await popup.close();
    const login = await browser.beginLogin(url, visit.tabId);
    assert.equal(adHits, hits);
    await browser.finishLogin(false);
    assert.ok(login.tabId);
  } finally { await browser.shutdown(); await new Promise(resolve => server.close(resolve)); await rm(dir, { recursive: true, force: true }); }
});
