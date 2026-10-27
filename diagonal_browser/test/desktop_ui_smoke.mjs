import { _electron as electron } from 'playwright';
import { SearchHistory } from '../search_history.mjs';
import { mkdtemp, rm, writeFile, readFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import http from 'node:http';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'diagonal-desktop-ui-'));
const server = http.createServer((_req, res) => { res.setHeader('content-type', 'text/html'); res.end('<title>Browser layout fixture</title><h1>Working browser page</h1><a href="/next">Next page</a>'); });
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/`;
const history = new SearchHistory(profile);
for (const query of ['Black hole photography', 'Playwright browser controls', 'A longer research query to check how saved search titles fit']) {
  await history.save({ kind: 'search', source: 'llm', query, results: [{ title: 'Saved browser result', url, snippet: 'Saved source evidence.' }] });
}
const calls = path.join(profile, 'search-calls'); const ddgs = path.join(profile, 'ddgs-fixture');
await writeFile(ddgs, `#!${process.execPath}\nrequire('node:fs').appendFileSync(${JSON.stringify(calls)},'search\\n');console.log(JSON.stringify([{title:'Fresh browser result',url:${JSON.stringify(url)},snippet:'Fresh source evidence'}]));\n`, { mode: 0o700 });
const env = { ...process.env, DIAGONAL_PROFILE_DIR: profile, DIAGONAL_DDGS_BIN: ddgs, DIAGONAL_ADBLOCK_AUTO_UPDATE: '0' }; delete env.ELECTRON_RUN_AS_NODE;
let app;
try {
  app = await electron.launch({ args: [path.join(root, 'desktop.mjs')], env, timeout: 60000 });
  const page = await app.firstWindow();
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.locator('.tab.active').waitFor();
  await page.locator('.recent-search').nth(2).waitFor();
  assert.equal((await page.locator('.chrome').boundingBox()).height, 104);
  const actions = ['back', 'forward', 'reload', 'home', 'images', 'screenshot', 'record', 'adblock', 'accounts', 'history', 'inspect', 'media'];
  async function verifyToolbar() {
    const width = await page.evaluate(() => innerWidth);
    for (const id of actions) {
      const box = await page.locator(`#${id}`).boundingBox();
      assert.ok(box && box.x >= 0 && box.x + box.width <= width && box.y + box.height <= 104, `${id} must stay inside the toolbar`);
    }
    assert.ok((await page.locator('#address-form').boundingBox()).width >= 140);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  }
  await verifyToolbar();
  const artifacts = path.join(root, 'artifacts'); await mkdir(artifacts, { recursive: true });
  await page.screenshot({ path: path.join(artifacts, 'diagonal-ui-home.png') });
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(800, 700));
  await page.waitForFunction(() => innerWidth === 800); await verifyToolbar();
  await page.screenshot({ path: path.join(artifacts, 'diagonal-ui-compact.png') });
  await page.locator('.recent-search').first().click();
  await page.getByText(/historical results/).waitFor();
  await assert.rejects(readFile(calls), { code: 'ENOENT' });
  assert.equal(await page.locator('#reload').isEnabled(), true);
  await page.getByRole('button', { name: 'Reload', exact: true }).click();
  await page.getByRole('button', { name: 'Fresh browser result', exact: true }).waitFor();
  assert.match(await readFile(calls, 'utf8'), /search/);
  await page.locator('.tab.active .tab-close').click();
  await page.locator('.home-content').waitFor();
  await page.keyboard.press('Meta+Shift+T');
  await page.getByRole('button', { name: 'Fresh browser result', exact: true }).waitFor();
  await page.getByLabel('Address and search', { exact: true }).fill(url);
  await page.getByLabel('Address and search', { exact: true }).press('Enter');
  await page.waitForFunction(() => document.querySelector('#page-context-label').textContent.includes('127.0.0.1'));
  const bounds = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].contentView.children.map(view => view.getBounds()));
  assert.ok(bounds.some(box => box.y === 104 && box.width === 800), 'Native page must align below the new toolbar');
  for (const [name, title] of [['Ad blocker settings', 'Privacy shield'], ['Search history', 'Search history'], ['Email nicknames', 'Email nicknames'], ['Inspect current page', 'URL & page inspector']]) {
    await page.getByRole('button', { name, exact: true }).click();
    await page.getByRole('heading', { name: title, exact: true }).waitFor();
    assert.equal(await page.locator('.chrome').getByRole('button', { name, exact: true }).getAttribute('aria-pressed'), 'true');
    await page.keyboard.press('Escape');
    await page.locator('#panel').waitFor({ state: 'hidden' });
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ desktopUI: 'passed', checks: ['wide/compact toolbar', 'recent search restore', 'live result refresh', 'reopen closed tab', 'native view alignment', 'tool panels', 'Escape dismissal'], screenshots: ['artifacts/diagonal-ui-home.png', 'artifacts/diagonal-ui-compact.png'] }));
} finally { await app?.close(); await new Promise(resolve => server.close(resolve)); await rm(profile, { recursive: true, force: true }); }
