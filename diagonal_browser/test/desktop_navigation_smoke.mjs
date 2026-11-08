import { _electron as electron } from 'playwright';
import { mkdtemp, rm, writeFile, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import os from 'node:os';
import http from 'node:http';
import assert from 'node:assert/strict';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'diagonal-navigation-'));
const server = http.createServer((request, response) => {
  response.setHeader('content-type', 'text/html');
  response.end(`<title>${request.url}</title><h1>${request.url}</h1><input aria-label="Draft"><a href="/two">Next</a><script>setTimeout(()=>{const img=new Image();img.src='https://ads.doubleclick.net/navigation-fixture.gif';document.body.append(img)},100)</script>`);
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
const origin = `http://127.0.0.1:${server.address().port}`;
const calls = path.join(profile, 'search-calls'); const ddgs = path.join(profile, 'ddgs.cjs');
await writeFile(ddgs, `#!${process.execPath}\nrequire('node:fs').appendFileSync(${JSON.stringify(calls)},'search\\n');const respond=()=>console.log(JSON.stringify([{title:'Navigation result',url:${JSON.stringify(origin + '/one')},snippet:'Cached evidence'}]));if(process.argv[2]==='Delayed navigation')setTimeout(respond,250);else respond();\n`, { mode: 0o700 });
const env = { ...process.env, DIAGONAL_PROFILE_DIR: profile, DIAGONAL_DDGS_BIN: ddgs, DIAGONAL_ADBLOCK_AUTO_UPDATE: '0' }; delete env.ELECTRON_RUN_AS_NODE;
let app;
try {
  app = await electron.launch({ args: [path.join(root, 'desktop.mjs')], env, timeout: 60000 });
  const page = await app.firstWindow();
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  const action = (name, value) => page.evaluate(({ name, value }) => window.diagonal.action(name, value), { name, value });
  const state = () => action('state');
  const active = async () => { const value = await state(); return value.tabs.find(tab => tab.id === value.activeId); };
  const site = code => app.evaluate(async ({ BrowserWindow }, code) => {
    const view = BrowserWindow.getAllWindows()[0].contentView.children.find(view => view.webContents?.getURL().startsWith('http://127.0.0.1'));
    return view.webContents.executeJavaScript(code);
  }, code);
  async function loaded(url) {
    await page.waitForFunction(async url => { const value = await window.diagonal.action('state'); const tab = value.tabs.find(item => item.id === value.activeId); return tab.url === url && !tab.loading; }, url);
    for (let attempt = 0; attempt < 100; attempt++) {
      const ready = await app.evaluate(({ BrowserWindow }, url) => BrowserWindow.getAllWindows()[0].contentView.children.some(view => view.webContents?.getURL() === url && !view.webContents.isLoading()), url);
      if (ready) return;
      await new Promise(resolve => setTimeout(resolve, 25));
    }
    assert.fail(`The native page did not finish loading ${url}`);
  }
  await page.locator('.tab.active').waitFor();
  assert.equal(await page.locator('#back').isDisabled(), true);
  await action('navigate', origin + '/one'); await loaded(origin + '/one');
  await site(`document.querySelector('input').value = 'Preserve my draft'; location.href = '/two'`);
  await loaded(origin + '/two');
  await page.locator('#back').click(); await loaded(origin + '/one');
  assert.equal(await page.locator('#forward').isEnabled(), true);
  assert.equal(await site(`document.querySelector('input').value`), 'Preserve my draft');
  await page.locator('#forward').click(); await loaded(origin + '/two');
  await action('search', { query: 'Navigation fixture' });
  const count = (await readFile(calls, 'utf8')).split('\n').filter(Boolean).length;
  await page.locator('#back').click(); await loaded(origin + '/two');
  await page.locator('#forward').click();
  await page.getByRole('button', { name: 'Navigation result', exact: true }).waitFor();
  assert.equal((await readFile(calls, 'utf8')).split('\n').filter(Boolean).length, count, 'History must restore cached results without another search');
  await page.locator('#home').click(); await page.locator('.home-content').waitFor();
  await page.locator('#back').click(); await page.getByRole('button', { name: 'Navigation result', exact: true }).waitFor();
  await page.locator('#forward').click(); await page.locator('.home-content').waitFor();
  await page.locator('#back').click();
  await action('navigate', origin + '/branch'); await loaded(origin + '/branch');
  assert.equal(await page.locator('#forward').isDisabled(), true, 'New navigation must clear Forward');
  const tabId = (await active()).id;
  await action('newTab'); assert.equal((await active()).canGoBack, false);
  await action('selectTab', tabId); assert.equal((await active()).canGoBack, true);
  // A destroyed native view must not crash publication or future blocked requests.
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].contentView.children.find(view => view.webContents?.getURL().startsWith('http://127.0.0.1')).webContents.close());
  await page.locator('#back').click(); await page.getByRole('button', { name: 'Navigation result', exact: true }).waitFor();
  await page.locator('#back').click(); await loaded(origin + '/two');
  await page.waitForFunction(async () => (await window.diagonal.action('state')).blockedCount > 0);
  await action('navigate', origin + '/stress'); await loaded(origin + '/stress');
  for (let index = 0; index < 4; index++) {
    await action('newTab', origin + '/closing-' + index); await loaded(origin + '/closing-' + index);
    await action('closeTab', (await active()).id);
  }
  await page.evaluate(() => { window.fixtureSearch = window.diagonal.action('search', { query: 'Delayed navigation' }); });
  await action('home');
  await page.evaluate(() => window.fixtureSearch);
  assert.equal((await active()).searchResults, null, 'A finished search must not replace Home after navigating away');
  assert.equal((await active()).url, '');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ desktopNavigation: 'passed', checks: ['native Back/Forward', 'form state preserved', 'cached searches', 'Home history', 'forward branch reset', 'independent tabs', 'destroyed view recovery', 'blocked requests while closing tabs'] }));
} finally { await app?.close(); await new Promise(resolve => server.close(resolve)); await rm(profile, { recursive: true, force: true }); }
