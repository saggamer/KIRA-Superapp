import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { stat, unlink, mkdtemp, rm } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { DiagonalBrowser, webUrl, artifactPath } from '../browser.mjs';
import { saveEmailProfiles } from '../email_profiles.mjs';

test('browser can visit, interact, capture and record a local page', async () => {
  const server = http.createServer((_req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<title>Browser Fixture</title><input id="name"><button id="go" onclick="document.querySelector(\'#out\').textContent=document.querySelector(\'#name\').value">Go</button><label><input id="agree" type="checkbox">Agree</label><label for="choice">Choice</label><select id="choice"><option value="a">A</option><option value="b">B</option></select><input type="password" value="fixture-secret"><p id="out"></p>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = new DiagonalBrowser({ historyEnabled: false });
  let visualMap;
  const url = `http://127.0.0.1:${server.address().port}/`;
  try {
    const visit = await browser.navigate(url);
    assert.equal(visit.title, 'Browser Fixture');
    assert.ok(visit.observationId);
    assert.equal(visit.elements.find(item => item.name === 'Go')?.role, 'button');
    const password = visit.elements.find(item => item.sensitive);
    assert.equal(password.hasValue, true);
    assert.equal(JSON.stringify(visit).includes('fixture-secret'), false);
    await browser.fill('#name', undefined, 'Kira', visit.tabId);
    await browser.click(undefined, 1, undefined, visit.tabId);
    assert.match((await browser.snapshot(visit.tabId)).text, /Kira/);
    const observed = await browser.observe(visit.tabId);
    assert.equal(observed.elements.find(item => item.name === 'Go')?.role, 'button');
    assert.equal(observed.grid.columns, 12);
    assert.match(observed.elements.find(item => item.name === 'Go')?.gridCell || '', /^[A-H](?:[1-9]|1[0-2])$/);
    visualMap = await browser.observe(visit.tabId, true);
    assert.ok((await stat(visualMap.screenshot)).size > 100);
    await assert.rejects(browser.act({ operation: 'fill', index: 0, value: 'Diagonal', tabId: visit.tabId }), /require an OBSERVATION_ID/);
    const changed = await browser.act({ operation: 'fill', index: 0, value: 'Diagonal', observationId: visualMap.observationId, tabId: visit.tabId });
    assert.equal(changed.elements.find(item => item.index === 0)?.value, 'Diagonal');
    await assert.rejects(browser.act({ operation: 'click', index: 1, observationId: visualMap.observationId, tabId: visit.tabId }), /Stale browser observation/);
    const checked = await browser.act({ operation: 'check', role: 'checkbox', targetName: 'Agree', tabId: visit.tabId });
    assert.ok(checked.observationId);
    await browser.act({ operation: 'select_option', role: 'combobox', targetName: 'Choice', value: 'b', tabId: visit.tabId });
    assert.equal(await browser.pages.get(visit.tabId).locator('#choice').inputValue(), 'b');
    const afterGo = await browser.act({ operation: 'click', role: 'button', targetName: 'Go', tabId: visit.tabId });
    assert.equal(afterGo.text.includes('Diagonal'), true);
    const gridFilled = await browser.act({ operation: 'fill', index: 0, value: 'Grid', observationId: afterGo.observationId, tabId: visit.tabId });
    const go = gridFilled.elements.find(item => item.name === 'Go');
    await assert.rejects(browser.act({ operation: 'click', x: go.x, y: go.y, tabId: visit.tabId }), /require an OBSERVATION_ID/);
    assert.match((await browser.act({ operation: 'click', x: go.x, y: go.y, observationId: gridFilled.observationId, tabId: visit.tabId })).text, /Grid/);
    await browser.act({ operation: 'navigate', url: `${url}next`, tabId: visit.tabId });
    assert.equal((await browser.act({ operation: 'back', tabId: visit.tabId })).url, url);
    assert.equal((await browser.act({ operation: 'forward', tabId: visit.tabId })).url, `${url}next`);
    assert.equal((await browser.act({ operation: 'back', tabId: visit.tabId })).url, url);
    const inspected = await browser.inspect(visit.tabId);
    assert.equal(inspected.page.title, 'Browser Fixture');
    assert.equal(inspected.urlInfo.hostname, '127.0.0.1');
    assert.ok(inspected.page.forms.length === 0);
    const shot = await browser.screenshot(visit.tabId, 'smoke.png');
    assert.ok((await stat(shot.path)).size > 100);
    const recording = await browser.recordStart(visit.tabId);
    await browser.fill(undefined, 0, 'Diagonal', recording.tabId);
    await browser.click(undefined, undefined, 'Go', recording.tabId);
    await browser.navigate(`${url}final`, recording.tabId);
    const video = await browser.recordStop('smoke.webm');
    assert.ok((await stat(video.path)).size > 100);
    assert.equal(video.sourceUrl, url);
    assert.equal(video.lastUrl, `${url}final`);
    assert.equal(video.tabId, visit.tabId);

    const interrupted = await browser.recordStart(visit.tabId);
    await browser.pages.get(interrupted.tabId).close();
    assert.equal(browser.pages.has(interrupted.tabId), false);
    assert.equal(browser.active, visit.tabId);
    const interruptedVideo = await browser.recordStop('closed-tab.webm');
    assert.ok((await stat(interruptedVideo.path)).size > 100);
  } finally {
    await browser.shutdown();
    server.close();
    if (visualMap?.screenshot) await unlink(visualMap.screenshot).catch(() => {});
    await Promise.all(['smoke.png', 'smoke.webm', 'closed-tab.webm'].map(name => unlink(artifactPath(name, name.split('.').at(-1))).catch(() => {})));
  }
});

test('scroll returns visible lower text and a bounded position', async () => {
  const server = http.createServer((_req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<title>Long Page</title><h1>Top heading</h1><div style="height:1800px"></div><h2>Lower heading: Details</h2>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = new DiagonalBrowser({ historyEnabled: false });
  try {
    const first = await browser.navigate(`http://127.0.0.1:${server.address().port}/`);
    assert.equal(first.scroll.atTop, true);
    assert.equal(first.visibleText.includes('Lower heading: Details'), false);
    assert.equal(first.scroll.atEnd, false);
    await assert.rejects(browser.scroll(2000, first.tabId), /PIXELS/);
    let after = first;
    for (let step = 0; step < 4 && !after.scroll.atEnd; step++) after = await browser.scroll(700, first.tabId);
    assert.equal(after.scroll.atEnd, true);
    assert.equal(after.moved, true);
    assert.match(after.visibleText, /Lower heading: Details/);
    const up = await browser.scroll(-700, first.tabId);
    assert.ok(up.scroll.y < after.scroll.y);
  } finally {
    await browser.shutdown(); server.close();
  }
});

test('non-web URLs and artifact traversal are rejected', () => {
  assert.throws(() => webUrl('file:///etc/passwd'));
  assert.throws(() => artifactPath('../escape.png', 'png'));
});

test('saved email nickname fills a sign-in field without submitting', async () => {
  const server = http.createServer((_req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<title>Sign in</title><form id="sign-in"><input type="email" aria-label="Email"><button>Continue</button></form>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-login-test-'));
  const browser = new DiagonalBrowser({ profileDir: dir });
  try {
    await saveEmailProfiles([{ nickname: 'Personal', email: 'person@example.com' }], dir);
    await browser.navigate(`http://127.0.0.1:${server.address().port}/`);
    const result = await browser.useEmailProfile('Personal');
    assert.equal(result.submitted, false);
    assert.equal(result.elements.find(item => item.role === 'textbox')?.hasValue, true);
    assert.equal(JSON.stringify(result).includes('person@example.com'), false);
    assert.equal(await browser.pages.get(result.tabId).locator('input[type="email"]').inputValue(), 'person@example.com');
    assert.equal(await browser.pages.get(result.tabId).locator('form').evaluate(form => form.dataset.submitted || ''), '');
  } finally {
    await browser.shutdown(); server.close(); await rm(dir, { recursive: true, force: true });
  }
});

test('OAuth-style popup is tracked and can return to its opener', async () => {
  const server = http.createServer((req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(req.url === '/popup'
      ? '<title>Provider</title><button onclick="window.opener.postMessage(\'signed-in\', location.origin); window.close()">Authorize</button>'
      : '<title>App</title><button onclick="window.open(\'/popup\', \'oauth\', \'width=480,height=640\')">Sign in</button><p id="status">waiting</p><script>addEventListener("message", event => { if (event.origin === location.origin) document.querySelector("#status").textContent=event.data })</script>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = new DiagonalBrowser({ historyEnabled: false });
  try {
    const app = await browser.navigate(`http://127.0.0.1:${server.address().port}/`);
    const popup = await browser.act({ operation: 'click', role: 'button', targetName: 'Sign in' });
    assert.equal(popup.title, 'Provider');
    assert.notEqual(popup.tabId, app.tabId);
    await browser.act({ operation: 'click', role: 'button', targetName: 'Authorize' });
    await browser.pages.get(app.tabId).waitForFunction(() => document.querySelector('#status').textContent === 'signed-in');
    assert.equal(browser.pages.has(popup.tabId), false);
  } finally { await browser.shutdown(); server.close(); }
});

test('Kira browser restores cookie and local storage auth state', async () => {
  const server = http.createServer((req, res) => {
    res.setHeader('content-type', 'text/html');
    if (req.url === '/login') res.setHeader('set-cookie', 'session=demo; Path=/; SameSite=Lax');
    res.end(req.url === '/login'
      ? '<title>Login</title><script>localStorage.setItem("account", "demo")</script>Signed in'
      : '<title>Check</title><p id="status"></p><script>document.querySelector("#status").textContent=document.cookie+" "+localStorage.getItem("account")</script>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-auth-test-'));
  const first = new DiagonalBrowser({ profileDir: dir, persistAuth: true, loginHeadless: true });
  let second;
  try {
    await first.navigate(`http://127.0.0.1:${server.address().port}/check`);
    const login = await first.beginLogin(`http://127.0.0.1:${server.address().port}/login`);
    assert.equal(login.title, 'Login');
    assert.equal((await first.finishLogin()).saved, true);
    assert.match((await first.navigate(`http://127.0.0.1:${server.address().port}/check`)).text, /session=demo demo/);
    await first.shutdown();
    assert.equal((await stat(path.join(dir, 'kira_auth_state.json'))).mode & 0o777, 0o600);
    second = new DiagonalBrowser({ profileDir: dir, persistAuth: true });
    const resumed = await second.navigate(`http://127.0.0.1:${server.address().port}/check`);
    assert.match(resumed.text, /session=demo demo/);
  } finally {
    await first.shutdown().catch(() => {});
    await second?.shutdown(); server.close(); await rm(dir, { recursive: true, force: true });
  }
});

test('failed video initialization still clears recording state', async () => {
  const browser = new DiagonalBrowser({ historyEnabled: false });
  let closed = false;
  browser.pages.set('source', { url: () => 'https://example.com/' });
  browser.pages.set('recording', { url: () => 'https://example.com/final' });
  browser.active = 'recording';
  browser.recording = {
    context: { close: async () => { closed = true; } },
    page: { video: () => null, url: () => 'https://example.com/final' },
    id: 'recording', sourceId: 'source',
  };
  await assert.rejects(browser.recordStop(), /Video recording was not initialized/);
  assert.equal(closed, true);
  assert.equal(browser.recording, null);
  assert.equal(browser.active, 'source');
  assert.equal(browser.pages.has('recording'), false);
});

test('saved page history survives restart, reads offline, and reopens live sources', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'diagonal-page-history-'));
  let revision = 'Original orchid research';
  let requests = 0;
  const server = http.createServer((_req, res) => {
    requests++;
    res.setHeader('content-type', 'text/html');
    res.end(`<title>Orchid source</title><h1>${revision}</h1><p>${'evidence '.repeat(1700)}unique lower passage</p><input type="password" value="never-save-this-password">`);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}/orchids`;
  const first = new DiagonalBrowser({ profileDir: dir });
  const restarted = new DiagonalBrowser({ profileDir: dir });
  try {
    const visited = await first.navigate(url);
    assert.ok(visited.historyId);
    assert.equal((await first.observe(visited.tabId)).historyId, visited.historyId);
    assert.equal((await first.inspect(visited.tabId)).historyId, visited.historyId);
    assert.equal((await first.searchHistory({ kind: 'page', query: 'unique lower passage' })).total, 1);
    await first.shutdown();
    const cached = await restarted.readHistory(visited.historyId);
    assert.equal(restarted.browser, null);
    assert.match(cached.text, /Original orchid/); assert.equal(cached.text.includes('never-save-this-password'), false);
    const requestCount = requests;
    await restarted.searchHistory({ query: 'orchid research' });
    assert.equal(requests, requestCount);
    const search = await restarted.history.save({ kind: 'search', source: 'llm', query: 'orchid references', results: [{ title: 'Orchid source', url }] });
    assert.equal((await restarted.revisitHistory(search.id)).cached, true);
    assert.equal(restarted.browser, null);
    revision = 'Updated orchid research';
    const reopened = await restarted.revisitHistory(search.id, 0);
    assert.match(reopened.text, /Updated orchid/);
    assert.equal(reopened.revisitedHistoryId, search.id);
    assert.notEqual(reopened.historyId, visited.historyId);
    assert.match((await restarted.readHistory(visited.historyId)).text, /Original orchid/);
    await assert.rejects(restarted.revisitHistory(search.id, 3), /resultIndex/);
    await restarted.shutdown();
    await new Promise(resolve => server.close(resolve));
    assert.match((await restarted.readHistory(visited.historyId)).text, /Original orchid/);
  } finally {
    await restarted.shutdown().catch(() => {});
    await first.shutdown().catch(() => {});
    server.close();
    await rm(dir, { recursive: true, force: true });
  }
});
