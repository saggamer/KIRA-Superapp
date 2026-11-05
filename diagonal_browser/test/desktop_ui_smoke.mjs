import { _electron as electron } from 'playwright';
import { SearchHistory } from '../search_history.mjs';
import { mkdtemp, rm, writeFile, readFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import http from 'node:http';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';
import { supportsKiraPopup } from '../ai_platform.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'diagonal-desktop-ui-'));
let modelReplies = []; const modelRequests = [];
const server = http.createServer(async (req, res) => {
  if (req.url === '/v1/chat/completions') {
    let raw = ''; for await (const chunk of req) raw += chunk;
    const request = JSON.parse(raw); modelRequests.push(request);
    const reply = modelReplies.shift() || { action: 'answer', text: 'The current website is ready.' };
    const action = typeof reply === 'function' ? await reply(request) : reply;
    res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ choices: [{ message: { content: JSON.stringify(action) } }] })); return;
  }
  res.setHeader('content-type', 'text/html'); res.end('<title>Browser layout fixture</title><h1>Working browser page</h1><a href="/next">Next page</a><label>Find on page<input aria-label="Find on page"></label><button onclick="document.querySelector(\'h1\').textContent=\'Kira clicked successfully\'">Apply</button><input type="password" value="secret-fixture-password"><a href="https://example.org">External source</a><button onclick="window.open(\'https://example.org/popup\')">Scripted popup</button><button onclick="location.href=\'https://example.org/redirect\'">Scripted navigation</button>');
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/`;
const history = new SearchHistory(profile);
for (const query of ['Black hole photography', 'Playwright browser controls', 'A longer research query to check how saved search titles fit']) {
  await history.save({ kind: 'search', source: 'llm', query, results: [{ title: 'Saved browser result', url, snippet: 'Saved source evidence.' }] });
}
const calls = path.join(profile, 'search-calls'); const ddgs = path.join(profile, 'ddgs-fixture');
await writeFile(ddgs, `#!${process.execPath}\nrequire('node:fs').appendFileSync(${JSON.stringify(calls)},'search\\n');console.log(JSON.stringify([{title:'Fresh browser result',url:${JSON.stringify(url)},snippet:'Fresh source evidence'}]));\n`, { mode: 0o700 });
const env = { ...process.env, DIAGONAL_PROFILE_DIR: profile, DIAGONAL_DDGS_BIN: ddgs, DIAGONAL_ADBLOCK_AUTO_UPDATE: '0', KIRA_MODEL_BASE_URL: url + 'v1', KIRA_MODEL_NAME: 'kira-test-fixture' }; delete env.ELECTRON_RUN_AS_NODE;
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
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Shift+T' : 'Control+Shift+T');
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
  if (supportsKiraPopup()) {
    await page.getByRole('button', { name: 'Kira AI mode', exact: true }).click();
    await page.locator('#kira-popup').waitFor();
    const popupBox = await page.locator('#kira-popup').boundingBox();
    assert.ok(popupBox.x >= 410 && popupBox.x + popupBox.width <= 800);
    const aiBounds = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].contentView.children.map(view => view.getBounds()));
    assert.ok(aiBounds.some(box => box.width === 410), 'Website must remain visible beside popup');
    function observe(request) { return JSON.parse(request.messages.at(-1).content.split('\n').slice(1).join('\n')); }
    modelReplies = [request => ({ action: 'fill', element: observe(request).elements.find(item => item.label === 'Find on page').element, text: 'Kira entered this' }), request => ({ action: 'click', element: observe(request).elements.find(item => item.label === 'Apply').element }), { action: 'answer', text: 'I filled the field and clicked Apply on this page.' }];
    await page.getByLabel('Ask Kira', { exact: true }).fill('Use this website'); await page.getByRole('button', { name: 'Send to Kira' }).click();
    await page.getByText('I filled the field and clicked Apply on this page.', { exact: true }).waitFor();
    const actual = await app.evaluate(async ({ webContents }) => { const site = webContents.getAllWebContents().find(item => item.getURL().startsWith('http://127.0.0.1')); return site.executeJavaScript('({title:document.querySelector("h1").textContent,value:document.querySelector("input").value})'); });
    assert.equal(actual.title, 'Kira clicked successfully'); assert.equal(actual.value, 'Kira entered this');
    assert.ok(!JSON.stringify(modelRequests).includes('secret-fixture-password'), 'Input secrets must not be sent to model');
    modelReplies = [request => ({ action: 'fill', element: observe(request).elements.find(item => item.type === 'password').element, text: 'Must not fill' })];
    await page.getByLabel('Ask Kira', { exact: true }).fill('Try the password field'); await page.getByRole('button', { name: 'Send to Kira' }).click();
    await page.getByText(/Enter sensitive credentials or payment details yourself/).waitFor();
    const password = await app.evaluate(async ({ webContents }) => { const site = webContents.getAllWebContents().find(item => item.getURL().startsWith('http://127.0.0.1')); return site.executeJavaScript('document.querySelector("input[type=password]").value'); });
    assert.equal(password, 'secret-fixture-password');
    const before = await page.locator('.tab').count();
    for (const label of ['Scripted popup', 'Scripted navigation']) {
      modelReplies = [request => ({ action: 'click', element: observe(request).elements.find(item => item.label === label).element }), { action: 'answer', text: `${label} stayed on the page.` }];
      await page.getByLabel('Ask Kira', { exact: true }).fill(`Try ${label}`); await page.getByRole('button', { name: 'Send to Kira' }).click();
      await page.locator('#kira-permission').waitFor();
      assert.equal(await page.locator('.tab').count(), before);
      assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().length), 1, 'Page script must not create an unapproved popup');
      assert.match(await page.locator('#address').inputValue(), /127\.0\.0\.1/);
      await page.getByRole('button', { name: 'Stay here', exact: true }).click();
      await page.getByText(`${label} stayed on the page.`, { exact: true }).waitFor();
    }
    let releaseModel;
    modelReplies = [() => new Promise(resolve => { releaseModel = resolve; })];
    await page.getByLabel('Ask Kira', { exact: true }).fill('Wait before doing anything'); await page.getByRole('button', { name: 'Send to Kira' }).click();
    for (let attempt = 0; !releaseModel && attempt < 100; attempt++) await new Promise(resolve => setTimeout(resolve, 20));
    assert.ok(releaseModel, 'Model request should be in flight');
    await page.getByRole('button', { name: 'Stop', exact: true }).click();
    releaseModel({ action: 'search', query: 'Must never run after Stop' });
    await page.waitForTimeout(150);
    assert.equal(await page.locator('#kira-permission').isVisible(), false); assert.equal(await page.locator('.tab').count(), before);
    modelReplies = [{ action: 'search', query: 'Compare browser details', reason: 'Find another source' }];
    await page.getByLabel('Ask Kira', { exact: true }).fill('Research more'); await page.getByRole('button', { name: 'Send to Kira' }).click();
    await page.locator('#kira-permission').waitFor(); assert.equal(await page.locator('.tab').count(), before);
    await page.screenshot({ path: path.join(artifacts, 'diagonal-kira-permission.png') });
    await page.getByRole('button', { name: 'Stay here', exact: true }).click(); await page.getByText('The current website is ready.', { exact: true }).waitFor();
    assert.equal(await page.locator('.tab').count(), before);
    modelReplies = [{ action: 'search', query: 'Approved research', reason: 'Find another source' }, { action: 'answer', text: 'I searched in the approved tab.' }];
    await page.getByLabel('Ask Kira', { exact: true }).fill('Try another search'); await page.getByRole('button', { name: 'Send to Kira' }).click();
    await page.locator('#kira-permission').waitFor(); assert.equal(await page.locator('.tab').count(), before);
    await page.getByRole('button', { name: 'Allow once', exact: true }).click(); await page.getByText('I searched in the approved tab.', { exact: true }).waitFor();
    assert.equal(await page.locator('.tab').count(), before + 1);
    await page.screenshot({ path: path.join(artifacts, 'diagonal-kira-popup.png') });
    await page.getByRole('button', { name: 'Close Kira', exact: true }).click(); await page.locator('#kira-popup').waitFor({ state: 'hidden' });
  } else {
    assert.equal(await page.locator('#kira-toggle').isVisible(), false);
    const denied = await page.evaluate(async () => { try { await window.diagonal.action('aiToggle'); return false; } catch (error) { return error.message.includes('MLX edition'); } });
    assert.equal(denied, true, 'Non-MLX platforms must reject popup commands');
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ desktopUI: 'passed', checks: ['wide/compact toolbar', 'recent search restore', 'live result refresh', 'reopen closed tab', 'native view alignment', 'tool panels', 'Escape dismissal', 'Kira page fill and click', 'sensitive field guard', 'scripted popup/navigation permissions', 'search allow/deny', 'Stop cancels late responses'], screenshots: ['artifacts/diagonal-ui-home.png', 'artifacts/diagonal-ui-compact.png', 'artifacts/diagonal-kira-popup.png', 'artifacts/diagonal-kira-permission.png'] }));
} catch (error) {
  if (app) {
    const page = await app.firstWindow();
    console.error('Desktop notification:', await page.locator('#toast').innerText().catch(() => 'unavailable'));
    console.error('Desktop pages:', await app.evaluate(({ webContents }) => webContents.getAllWebContents().map(item => ({ url: item.getURL(), loading: item.isLoading(), title: item.getTitle() }))));
  }
  throw error;
} finally { await app?.close(); await new Promise(resolve => server.close(resolve)); await rm(profile, { recursive: true, force: true }); }
