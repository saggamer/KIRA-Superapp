import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile, rename, chmod } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractPage, inspectUrl } from './page_inspection.mjs';
import { loadEmailProfiles, emailProfileByNickname, defaultProfileDir } from './email_profiles.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(process.env.DIAGONAL_ARTIFACTS_DIR || path.join(here, 'artifacts'));
const localDdgs = path.join(os.homedir(), 'Downloads/chatgpt_antigravity_bridge/bin/ddgs');
const ddgs = process.env.DIAGONAL_DDGS_BIN || (existsSync(localDdgs) ? localDdgs : 'ddgs');
const chrome = process.env.DIAGONAL_CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const importPlaywright = () => import('playwright');
const actionableSelector = 'a[href], button, input, textarea, select, [role="button"], [role="link"], [role="textbox"], [role="checkbox"], [role="radio"], [role="combobox"]';

export function webUrl(input) {
  const url = new URL(input);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Only HTTP and HTTPS pages are supported');
  return url.href;
}

export function artifactPath(name, ext) {
  const basename = name || `${new Date().toISOString().replace(/[:.]/g, '-')}-${randomUUID().slice(0, 8)}.${ext}`;
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(basename) || !basename.endsWith(`.${ext}`)) {
    throw new Error(`Artifact name must be a simple .${ext} filename`);
  }
  return path.join(root, basename);
}

function run(command, args, timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs);
    child.stdout.on('data', chunk => { stdout += chunk; if (stdout.length > 2_000_000) child.kill('SIGKILL'); });
    child.stderr.on('data', chunk => { stderr += chunk; if (stderr.length > 20_000) child.kill('SIGKILL'); });
    child.on('error', error => { clearTimeout(timer); reject(error); });
    child.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) reject(new Error(stderr.trim() || `Command exited ${code}`));
      else resolve(stdout);
    });
  });
}

export class DiagonalBrowser {
  constructor({ profileDir = defaultProfileDir, persistAuth = false, loginHeadless = false } = {}) { this.browser = null; this.context = null; this.pages = new Map(); this.lastObservations = new Map(); this.active = null; this.recording = null; this.profileDir = profileDir; this.persistAuth = persistAuth; this.loginHeadless = loginHeadless; this.loginBrowser = null; this.loginContext = null; this.loginSourceId = null; this.authSaveTimer = null; this.authSaving = Promise.resolve(); }

  async ensure() {
    if (this.browser) return;
    const { chromium } = await importPlaywright();
    let storageState;
    if (this.persistAuth) {
      try { storageState = JSON.parse(await readFile(path.join(this.profileDir, 'kira_auth_state.json'), 'utf8')); }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
    }
    this.browser = await chromium.launch({ executablePath: chrome, headless: process.env.DIAGONAL_HEADED !== '1' });
    this.context = await this.browser.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: false, ...(storageState ? { storageState } : {}) });
    this.context.setDefaultTimeout(10000);
    this.context.setDefaultNavigationTimeout(20000);
    this.context.on('page', page => this._trackPage(page));
  }

  _queueAuthSave() {
    if (!this.persistAuth || !this.context) return;
    clearTimeout(this.authSaveTimer);
    this.authSaveTimer = setTimeout(() => { this._saveAuthState().catch(error => console.error('Could not save browser session:', error.message)); }, 800);
  }

  async _saveAuthState() {
    if (!this.persistAuth || !this.context) return;
    this.authSaving = this.authSaving.catch(() => {}).then(async () => {
      await mkdir(this.profileDir, { recursive: true });
      const target = path.join(this.profileDir, 'kira_auth_state.json');
      const temporary = path.join(this.profileDir, `.kira_auth_state-${randomUUID()}.tmp`);
      await writeFile(temporary, JSON.stringify(await this.context.storageState({ indexedDB: true })), { mode: 0o600 });
      await rename(temporary, target);
      await chmod(target, 0o600);
    });
    return this.authSaving;
  }

  _trackPage(page) {
    const existing = [...this.pages].find(([, known]) => known === page);
    if (existing) return existing[0];
    const id = randomUUID().slice(0, 8);
    this.pages.set(id, page);
    this.active = id;
    page.on('load', () => { if (page.context() === this.context) this._queueAuthSave(); });
    page.on('close', () => { this.pages.delete(id); this.lastObservations.delete(id); if (this.active === id) this.active = this.pages.keys().next().value || null; });
    return id;
  }

  async page(id) {
    await this.ensure();
    const key = id || this.active;
    if (key && this.pages.has(key)) return [key, this.pages.get(key)];
    if (id) throw new Error(`Unknown tab: ${id}`);
    const page = await this.context.newPage();
    return [this._trackPage(page), page];
  }

  async search(query, limit = 5) {
    if (!query?.trim()) throw new Error('Search query is required');
    const output = await run(ddgs, [query, '--max-results', String(Math.max(1, Math.min(limit, 10))), '--format', 'json']);
    const results = JSON.parse(output);
    if (!Array.isArray(results) || results.some(item => item.error)) throw new Error(results?.[0]?.error || 'DDG returned invalid results');
    return { query, results };
  }

  async navigate(url, tabId) {
    const safeUrl = webUrl(url);
    const [id, page] = await this.page(tabId);
    const response = await page.goto(safeUrl, { waitUntil: 'domcontentloaded' });
    this.active = id;
    return { ...(await this.observe(id)), status: response?.status() ?? null };
  }

  async newTab(url) {
    if (url) webUrl(url);
    await this.ensure();
    if (this.recording) throw new Error('Finish the current recording before opening another tab');
    const page = await this.context.newPage();
    const id = this._trackPage(page);
    if (url) return this.navigate(url, id);
    return { tabId: id, url: page.url() };
  }

  async back(tabId) {
    const [id, page] = await this.page(tabId);
    await page.goBack({ waitUntil: 'domcontentloaded' });
    return this.observe(id);
  }

  async forward(tabId) {
    const [id, page] = await this.page(tabId);
    await page.goForward({ waitUntil: 'domcontentloaded' });
    return this.observe(id);
  }

  async reload(tabId) {
    const [id, page] = await this.page(tabId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    return this.observe(id);
  }

  async selectTab(tabId) {
    if (!tabId || !this.pages.has(tabId)) throw new Error(`Unknown tab: ${tabId}`);
    this.active = tabId;
    return this.observe(tabId);
  }

  async scroll(pixels = 600, tabId) {
    const [id, page] = await this.page(tabId);
    if (!Number.isInteger(pixels) || pixels < -1600 || pixels > 1600 || pixels === 0) {
      throw new Error('Scroll PIXELS must be an integer from -1600 to 1600, excluding zero');
    }
    const before = await page.evaluate(() => window.scrollY);
    await page.mouse.wheel(0, pixels);
    await page.waitForTimeout(120);
    const result = await this.observe(id);
    return { ...result, pixels, moved: result.scroll.y !== before };
  }

  async waitFor(selector, tabId) {
    const [id, page] = await this.page(tabId);
    await page.locator(selector).first().waitFor({ state: 'visible' });
    return { ...(await this.observe(id)), selector, visible: true };
  }

  async tabs() {
    await this.ensure();
    const result = [];
    for (const [tabId, page] of this.pages) result.push({ tabId, url: page.url(), title: await page.title().catch(() => ''), active: tabId === this.active });
    return { tabs: result };
  }

  async snapshot(tabId) {
    const [id, page] = await this.page(tabId);
    const items = await page.locator('a[href], button, input, textarea, select').evaluateAll(nodes => nodes.slice(0, 100).map((node, index) => ({
      index, tag: node.tagName.toLowerCase(), role: node.getAttribute('role'), text: (node.innerText || node.getAttribute('aria-label') || node.getAttribute('placeholder') || '').trim().slice(0, 120),
      href: node instanceof HTMLAnchorElement ? node.href : null
    })));
    return { tabId: id, url: page.url(), title: await page.title(), text: (await page.locator('body').innerText()).slice(0, 25000), elements: items };
  }

  async observe(tabId, screenshot = false) {
    const [id, page] = await this.page(tabId);
    const viewport = page.viewportSize() || { width: 1440, height: 900 };
    const elements = await page.locator(actionableSelector).evaluateAll((nodes, viewport) => nodes.map((node, index) => {
      const box = node.getBoundingClientRect();
      const style = getComputedStyle(node);
      if (!box.width || !box.height || style.visibility === 'hidden' || style.display === 'none' || box.right < 0 || box.bottom < 0 || box.left > viewport.width || box.top > viewport.height) return null;
      const x = Math.max(0, Math.min(viewport.width - 1, Math.round(box.left + box.width / 2)));
      const y = Math.max(0, Math.min(viewport.height - 1, Math.round(box.top + box.height / 2)));
      return {
        index, role: node.getAttribute('role') || (node instanceof HTMLInputElement
          ? ({ checkbox: 'checkbox', radio: 'radio', button: 'button', submit: 'button' })[node.type] || 'textbox'
          : ({ A: 'link', BUTTON: 'button', TEXTAREA: 'textbox', SELECT: 'combobox' })[node.tagName] || node.tagName.toLowerCase()),
        name: (node.getAttribute('aria-label') || node.labels?.[0]?.innerText || node.innerText || node.getAttribute('placeholder') || node.getAttribute('title') || '').trim().slice(0, 100),
        x, y, gridCell: `${String.fromCharCode(65 + Math.min(7, Math.floor(y / viewport.height * 8)))}${Math.min(12, Math.floor(x / viewport.width * 12) + 1)}`,
        bounds: { x: Math.round(box.x), y: Math.round(box.y), width: Math.round(box.width), height: Math.round(box.height) },
        href: node instanceof HTMLAnchorElement ? node.href : undefined,
        options: node instanceof HTMLSelectElement
          ? [...node.options].slice(0, 30).map(option => ({ label: option.label.trim().slice(0, 80), value: option.value.slice(0, 80) }))
          : undefined,
        ...(node instanceof HTMLSelectElement ? { value: node.value.slice(0, 80) } : {}),
        ...(() => {
          if (!(node instanceof HTMLInputElement || node instanceof HTMLTextAreaElement)) return {};
          const sensitive = /^(password|email|hidden)$/i.test(node.type || '') ||
            /password|secret|token|api.?key|auth|credit|card|email/i.test(
              `${node.name || ''} ${node.id || ''} ${node.autocomplete || ''}`);
          return sensitive
            ? { sensitive: true, hasValue: !!node.value, valueLength: node.value.length }
            : { value: node.value.slice(0, 100) };
        })(),
      };
    }).filter(Boolean).slice(0, 80), viewport);
    const result = {
      tabId: id, url: page.url(), title: await page.title(), viewport, grid: { columns: 12, rows: 8 },
      text: (await page.locator('body').innerText().catch(() => '')).slice(0, 9000),
      elements,
    };
    const viewportEvidence = await page.evaluate(() => {
      const root = document.scrollingElement || document.documentElement;
      const y = Math.round(window.scrollY);
      const maxY = Math.max(0, root.scrollHeight - window.innerHeight);
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      const lines = [];
      let node;
      while ((node = walker.nextNode()) && lines.join('\n').length < 5500) {
        const content = node.textContent?.replace(/\s+/g, ' ').trim();
        if (!content) continue;
        const parent = node.parentElement;
        if (!parent || parent.closest('script,style,noscript,[aria-hidden="true"]')) continue;
        const style = getComputedStyle(parent);
        if (style.display === 'none' || style.visibility === 'hidden') continue;
        const range = document.createRange();
        range.selectNodeContents(node);
        if ([...range.getClientRects()].some(rect => rect.width > 0 && rect.height > 0 &&
            rect.bottom > 0 && rect.top < window.innerHeight &&
            rect.right > 0 && rect.left < window.innerWidth)) lines.push(content);
      }
      return { visibleText: lines.join('\n').slice(0, 5500),
        scroll: { y, maxY: Math.round(maxY), atTop: y <= 1, atEnd: y >= maxY - 2 } };
    });
    Object.assign(result, viewportEvidence);
    result.observationId = randomUUID().slice(0, 12);
    this.lastObservations.set(id, result.observationId);
    if (screenshot) {
      await page.evaluate(({ elements, viewport }) => {
        document.getElementById('__diagonal_visual_map__')?.remove();
        const overlay = document.createElement('div');
        overlay.id = '__diagonal_visual_map__';
        overlay.style.cssText = 'position:fixed;inset:0;z-index:2147483647;pointer-events:none;font:700 12px system-ui,sans-serif;color:#fff';
        const mark = (left, top, width, height, background, label = '') => {
          const el = document.createElement('div');
          el.style.cssText = `position:absolute;left:${left}px;top:${top}px;width:${width}px;height:${height}px;background:${background};pointer-events:none;`;
          el.textContent = label; overlay.append(el);
        };
        for (let col = 1; col < 12; col++) mark(col * viewport.width / 12, 0, 1, viewport.height, '#ffd09955');
        for (let row = 1; row < 8; row++) mark(0, row * viewport.height / 8, viewport.width, 1, '#ffd09955');
        for (let col = 0; col < 12; col++) mark(col * viewport.width / 12 + 4, 4, 22, 18, '#17110de8', String(col + 1));
        for (let row = 0; row < 8; row++) mark(4, row * viewport.height / 8 + 24, 22, 18, '#17110de8', String.fromCharCode(65 + row));
        elements.forEach(item => {
          const outline = document.createElement('div');
          outline.style.cssText = `position:absolute;left:${item.bounds.x}px;top:${item.bounds.y}px;width:${item.bounds.width}px;height:${item.bounds.height}px;border:2px solid #edb575;border-radius:3px;box-shadow:0 0 0 1px #0008;pointer-events:none`;
          overlay.append(outline);
          const tag = document.createElement('div');
          tag.style.cssText = `position:absolute;left:${Math.max(0, Math.min(viewport.width - 34, item.bounds.x))}px;top:${Math.max(0, Math.min(viewport.height - 22, item.bounds.y - 19))}px;padding:2px 5px;border:1px solid #fff;border-radius:4px;background:#17110de8;color:#ffd099;box-shadow:0 1px 5px #000;pointer-events:none`;
          tag.textContent = String(item.index); overlay.append(tag);
        });
        document.documentElement.append(overlay);
      }, { elements, viewport });
      try { result.screenshot = (await this.screenshot(id, undefined, false)).path; }
      finally { await page.evaluate(() => document.getElementById('__diagonal_visual_map__')?.remove()).catch(() => {}); }
    }
    return result;
  }

  async act({ operation, tabId, url, role, targetName, selector, index, text, value, key, pixels, x, y, observationId } = {}) {
    if (!operation) throw new Error('An operation is required');
    if (operation === 'navigate') return this.navigate(url, tabId);
    if (operation === 'new_tab') { const tab = await this.newTab(url); return tab.observationId ? tab : this.observe(tab.tabId); }
    if (operation === 'select_tab') return this.selectTab(tabId);
    if (operation === 'close_tab') { await this.close(tabId); return this.tabs(); }
    if (operation === 'back') return this.back(tabId);
    if (operation === 'forward') return this.forward(tabId);
    if (operation === 'reload') return this.reload(tabId);
    if (operation === 'scroll') return this.scroll(pixels ?? 600, tabId);
    if (Number.isInteger(index) || (Number.isFinite(x) && Number.isFinite(y))) {
      if (!observationId) {
        throw new Error('INDEX and X/Y actions require an OBSERVATION_ID from the latest observation.');
      }
      const selectedId = tabId || this.active;
      if (this.lastObservations.get(selectedId) !== observationId) {
        throw new Error('Stale browser observation. Observe the tab again before using INDEX or X/Y.');
      }
    }
    if (operation === 'click') {
      if (Number.isFinite(x) && Number.isFinite(y)) {
        const [id, page] = await this.page(tabId);
        this.active = id;
        const viewport = page.viewportSize();
        if (x < 0 || y < 0 || x >= viewport.width || y >= viewport.height) throw new Error('Click coordinates must be inside the viewport');
        await page.mouse.click(x, y);
        return this.observe(this.active);
      }
      return this.click(selector, index, text, tabId, role, targetName);
    }
    if (operation === 'fill') return this.fill(selector, index, value ?? '', tabId, role, targetName);
    if (operation === 'press') return this.press(key, tabId);
    if (['hover', 'check', 'uncheck', 'select_option'].includes(operation)) {
      const [id, page] = await this.page(tabId);
      this.active = id;
      const target = this._target(page, selector, index, text, role, targetName);
      if (operation === 'hover') await target.hover();
      if (operation === 'check') await target.check();
      if (operation === 'uncheck') await target.uncheck();
      if (operation === 'select_option') await target.selectOption(value);
      return this.observe(id);
    }
    throw new Error(`Unsupported browser operation: ${operation}`);
  }

  async listEmailProfiles() { return { profiles: await loadEmailProfiles(this.profileDir) }; }

  async useEmailProfile(nickname, tabId) {
    const profile = emailProfileByNickname(await loadEmailProfiles(this.profileDir), nickname);
    const [id, page] = await this.page(tabId);
    const candidates = page.locator('input[type="email"], input[autocomplete="email"], input[autocomplete="username"], input[name*="email" i], input[id*="email" i]');
    for (const input of await candidates.all()) {
      if (await input.isVisible() && await input.isEnabled()) {
        await input.fill(profile.email);
        return { nickname: profile.nickname, submitted: false, ...(await this.observe(id)) };
      }
    }
    throw new Error('No visible email or username field was found on this page');
  }

  async beginLogin(url, tabId) {
    const safeUrl = webUrl(url);
    await this.ensure();
    if (this.loginBrowser) throw new Error('A visible login is already in progress');
    const sourceId = tabId || this.active;
    if (tabId && !this.pages.has(tabId)) throw new Error(`Unknown tab: ${tabId}`);
    const { chromium } = await importPlaywright();
    const loginBrowser = await chromium.launch({ executablePath: chrome, headless: this.loginHeadless });
    try {
      const state = await this.context.storageState({ indexedDB: true });
      const loginContext = await loginBrowser.newContext({ viewport: { width: 1440, height: 900 }, storageState: state, acceptDownloads: false });
      loginContext.setDefaultTimeout(10000);
      loginContext.setDefaultNavigationTimeout(20000);
      this.loginBrowser = loginBrowser;
      this.loginContext = loginContext;
      this.loginSourceId = sourceId;
      loginContext.on('page', page => this._trackPage(page));
      const page = await loginContext.newPage();
      const id = this._trackPage(page);
      await page.goto(safeUrl, { waitUntil: 'domcontentloaded' });
      return { ...(await this.observe(id)), visible: !this.loginHeadless, message: 'Complete the website sign-in and any passkey or consent prompt in the visible Chrome window, then call finish_login.' };
    } catch (error) {
      await loginBrowser.close();
      this.loginBrowser = null; this.loginContext = null; this.loginSourceId = null;
      throw error;
    }
  }

  async finishLogin(save = true) {
    if (!this.loginBrowser || !this.loginContext) throw new Error('No visible login is in progress');
    const activePage = this.pages.get(this.active);
    const lastUrl = activePage?.context() === this.loginContext ? activePage.url() : null;
    const sourceId = this.loginSourceId;
    try {
      if (save) {
        const state = await this.loginContext.storageState({ indexedDB: true });
        await this.context.setStorageState(state);
        await this._saveAuthState();
      }
    } finally {
      await this.loginBrowser.close();
      this.loginBrowser = null; this.loginContext = null; this.loginSourceId = null;
      this.active = sourceId && this.pages.has(sourceId) ? sourceId : (this.pages.keys().next().value || null);
    }
    return { saved: save, lastUrl, activeTabId: this.active, message: save ? 'Browser login state saved for Kira. Revisit the site to verify that it is signed in.' : 'Visible login cancelled without saving state.' };
  }

  async inspect(tabId) {
    const [id, page] = await this.page(tabId);
    return { tabId: id, urlInfo: inspectUrl(page.url()), page: await page.evaluate(extractPage) };
  }

  _target(page, selector, index, text, role, targetName) {
    if (Number.isInteger(index)) return page.locator(actionableSelector).nth(index);
    if (role && targetName) return page.getByRole(role, { name: targetName, exact: true }).first();
    if (selector) return page.locator(selector).first();
    if (text) return page.getByText(text, { exact: true }).first();
    throw new Error('Provide a CSS selector, snapshot index, or exact text');
  }

  async click(selector, index, text, tabId, role, targetName) {
    const [id, page] = await this.page(tabId);
    this.active = id;
    await this._target(page, selector, index, text, role, targetName).click();
    if (page.isClosed()) return { tabId: id, closed: true, activeTabId: this.active };
    return this.observe(this.active);
  }

  async fill(selector, index, value, tabId, role, targetName) {
    const [id, page] = await this.page(tabId);
    this.active = id;
    await this._target(page, selector, index, undefined, role, targetName).fill(value);
    return { ...(await this.observe(id)), valueLength: value.length };
  }

  async press(key, tabId) {
    const [id, page] = await this.page(tabId);
    this.active = id;
    await page.keyboard.press(key);
    return this.observe(id);
  }

  async screenshot(tabId, name, fullPage = true) {
    const [id, page] = await this.page(tabId);
    const saved = artifactPath(name, 'png');
    await mkdir(root, { recursive: true });
    await page.screenshot({ path: saved, fullPage, animations: 'disabled' });
    return { tabId: id, path: saved, url: page.url() };
  }

  async recordStart(tabId) {
    if (this.recording) throw new Error('A recording is already active');
    const [id, page] = await this.page(tabId);
    await mkdir(root, { recursive: true });
    const state = await this.context.storageState();
    const context = await this.browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: state, recordVideo: { dir: root, size: { width: 1440, height: 900 } } });
    context.setDefaultTimeout(10000);
    let recordingPage;
    try {
      recordingPage = await context.newPage();
      if (page.url().startsWith('http')) await recordingPage.goto(page.url(), { waitUntil: 'domcontentloaded' });
    } catch (error) {
      await context.close();
      throw error;
    }
    const recordingId = randomUUID().slice(0, 8);
    this.pages.set(recordingId, recordingPage); this.active = recordingId;
    this.recording = { context, page: recordingPage, id: recordingId, sourceId: id };
    recordingPage.on('close', () => {
      this.pages.delete(recordingId);
      if (this.active === recordingId) this.active = id;
    });
    return { tabId: recordingId, url: recordingPage.url(), message: 'Recording started. Use this tabId for subsequent actions; stop_recording saves the video.' };
  }

  async recordStop(name) {
    if (!this.recording) throw new Error('No recording is active');
    const { context, page, id, sourceId } = this.recording;
    const saved = artifactPath(name, 'webm');
    const video = page.video();
    const lastUrl = page.url();
    const sourceUrl = this.pages.get(sourceId)?.url() ?? null;
    try {
      await context.close();
      if (!video) throw new Error('Video recording was not initialized. Run npm run setup in diagonal_browser.');
      await video.saveAs(saved);
      await video.delete().catch(() => {});
    } finally {
      this.pages.delete(id);
      this.active = this.pages.has(sourceId) ? sourceId : (this.pages.keys().next().value || null);
      this.recording = null;
    }
    return { path: saved, lastUrl, tabId: sourceId, sourceUrl };
  }

  async close(tabId) {
    const [id, page] = await this.page(tabId);
    if (this.recording && [this.recording.id, this.recording.sourceId].includes(id)) throw new Error('Stop recording before closing this tab');
    await page.close();
    return { closed: id };
  }

  async shutdown() {
    if (this.recording) await this.recordStop().catch(() => {});
    await this.loginBrowser?.close();
    clearTimeout(this.authSaveTimer);
    await this._saveAuthState();
    await this.browser?.close();
  }
}
