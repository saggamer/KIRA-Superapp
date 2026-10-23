import { app, BrowserWindow, WebContentsView, Menu, clipboard, dialog, ipcMain, session, shell } from 'electron';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { mkdir, readdir, stat, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { artifactPath } from './browser.mjs';
import { blockedHost } from './blocking.mjs';
import { extractPage, inspectUrl } from './page_inspection.mjs';
import { loadEmailProfiles, saveEmailProfiles, emailProfileByNickname } from './email_profiles.mjs';
import { SearchHistory } from './search_history.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const localDdgs = path.join(os.homedir(), 'Downloads/chatgpt_antigravity_bridge/bin/ddgs');
const ddgs = process.env.DIAGONAL_DDGS_BIN || (existsSync(localDdgs) ? localDdgs : 'ddgs');
const imagePython = process.env.DIAGONAL_IMAGE_PYTHON || path.join(here, '.venv/bin/python');
const ffmpeg = process.env.DIAGONAL_FFMPEG_BIN || 'ffmpeg';
const toolbarHeight = 136;
const artifactsDir = path.dirname(artifactPath('gallery.png', 'png'));
const profileDir = process.env.DIAGONAL_DESKTOP_SMOKE === '1'
  ? path.join(app.getPath('temp'), `diagonal-browser-smoke-${process.pid}`)
  : process.env.DIAGONAL_PROFILE_DIR || path.join(here, '.profile');
const searchHistory = new SearchHistory(profileDir);
mkdirSync(profileDir, { recursive: true });
mkdirSync(path.join(profileDir, 'session'), { recursive: true });
let window;
let tabs = [];
let activeId = null;
let recording = null;
let panel = null;
let inspection = null;
let blockedCount = 0;
let adBlockEnabled = true;
let emailProfiles = [];
try { adBlockEnabled = JSON.parse(readFileSync(path.join(profileDir, 'settings.json'), 'utf8')).adBlockEnabled !== false; }
catch { /* Default to blocking known ad and tracker hosts. */ }

app.setName('Diagonal Browser');
app.setPath('userData', profileDir);
app.setPath('sessionData', path.join(profileDir, 'session'));
const hasInstanceLock = app.requestSingleInstanceLock();
if (!hasInstanceLock) app.quit();
app.on('second-instance', () => {
  if (window && !window.isDestroyed()) {
    if (window.isMinimized()) window.restore();
    window.show();
    window.focus();
  }
});

function current() { return tabs.find(tab => tab.id === activeId); }
function goBack() { const history = current()?.view?.webContents.navigationHistory; if (history?.canGoBack()) history.goBack(); }
function goForward() { const history = current()?.view?.webContents.navigationHistory; if (history?.canGoForward()) history.goForward(); }
function publicState() {
  return {
    tabs: tabs.map(({ id, title, url, loading, searchResults, blocked, view }) => ({
      id, title, url, loading, searchResults, blocked,
      canGoBack: view?.webContents.navigationHistory.canGoBack() || false,
      canGoForward: view?.webContents.navigationHistory.canGoForward() || false,
    })),
    activeId,
    recording: recording ? { tabId: recording.tabId, path: recording.path } : null,
    panel, inspection, adBlockEnabled, blockedCount, emailProfiles,
  };
}
function publish() {
  if (window && !window.isDestroyed()) window.webContents.send('diagonal:state', publicState());
}
function bounds() {
  const [width, height] = window.getContentSize();
  return { x: 0, y: toolbarHeight, width, height: Math.max(1, height - toolbarHeight) };
}
function showActive() {
  for (const tab of tabs) {
    if (!tab.view) continue;
    if (tab.id === activeId && !panel) {
      window.contentView.addChildView(tab.view);
      tab.view.setBounds(bounds());
      tab.view.webContents.focus();
    } else {
      window.contentView.removeChildView(tab.view);
    }
  }
  publish();
}
function safeUrl(input) {
  const candidate = /^[a-z][a-z\d+.-]*:\/\//i.test(input) ? input : `https://${input}`;
  const url = new URL(candidate);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Only HTTP and HTTPS sites can be opened');
  return url.href;
}
function looksLikeUrl(input) {
  return /^(https?:\/\/|localhost(?::\d+)?(?:\/|$)|127\.0\.0\.1(?::\d+)?(?:\/|$)|[^\s]+\.[^\s]+(?:\/|$))/i.test(input);
}
function newTab(url) {
  if (recording) throw new Error('Stop recording before opening another tab');
  const tab = { id: randomUUID().slice(0, 8), title: 'New tab', url: '', loading: false, searchResults: null, view: null, blocked: 0 };
  tabs.push(tab);
  activeId = tab.id;
  panel = null;
  showActive();
  if (url) navigate(url, tab.id);
  return tab.id;
}

async function rememberDesktopPage(tab, view) {
  if (tab.view !== view || view.webContents.isDestroyed()) return;
  try {
    const page = await view.webContents.executeJavaScript('({ url: location.href, title: document.title, text: (document.body?.innerText || "").slice(0, 25001) })');
    if (!/^https?:\/\//i.test(page.url) || tab.view !== view || tab.url !== page.url) return;
    if (!tab.historyVisit || tab.historyVisit.url !== page.url) tab.historyVisit = { id: randomUUID(), createdAt: new Date().toISOString(), url: page.url };
    await searchHistory.save({ ...tab.historyVisit, ...page, kind: 'page', source: 'desktop' });
  } catch (error) { console.error('Could not save page history:', error.message); }
}

async function revisitHistory(value) {
  const entry = await searchHistory.get(value?.historyId);
  if (entry.kind === 'page') return navigate(entry.url);
  if (value.resultIndex !== undefined) {
    if (!Number.isInteger(value.resultIndex) || value.resultIndex < 0 || !entry.results[value.resultIndex]?.url) throw new Error('Unknown saved result');
    return navigate(entry.results[value.resultIndex].url);
  }
  const tab = current();
  if (!tab) throw new Error('Tab not found');
  if (recording) throw new Error('Stop recording before restoring a saved search');
  if (tab.view) { window.contentView.removeChildView(tab.view); tab.view.webContents.close(); tab.view = null; }
  tab.url = ''; tab.title = `Saved search: ${entry.query}`;
  tab.searchResults = { query: entry.query, results: entry.results, mode: entry.mode, savedAt: entry.createdAt, historyId: entry.id };
  panel = null; showActive();
}

function attachView(tab) {
  if (tab.view) return tab.view;
  const view = new WebContentsView({ webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true } });
  tab.view = view;
  view.webContents.setWindowOpenHandler(({ url }) => {
    if (!/^https?:\/\//i.test(url) && url !== 'about:blank') return { action: 'deny' };
    return {
      action: 'allow',
      overrideBrowserWindowOptions: {
        width: 560, height: 720, minWidth: 400, minHeight: 480,
        title: 'Sign in · Diagonal', backgroundColor: '#15110d', autoHideMenuBar: true,
        webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true, session: session.defaultSession },
      },
    };
  });
  view.webContents.on('did-create-window', child => {
    child.webContents.on('will-navigate', event => {
      if (!/^https?:\/\//i.test(event.url)) event.preventDefault();
    });
    child.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  });
  view.webContents.on('will-navigate', event => {
    if (!/^https?:\/\//i.test(event.url)) event.preventDefault();
  });
  view.webContents.on('will-redirect', event => {
    if (!/^https?:\/\//i.test(event.url)) event.preventDefault();
  });
  view.webContents.on('did-start-loading', () => { tab.loading = true; publish(); });
  view.webContents.on('did-stop-loading', () => { tab.loading = false; publish(); void rememberDesktopPage(tab, view); });
  view.webContents.on('did-navigate', (_event, url) => { tab.url = url; tab.historyVisit = null; publish(); });
  view.webContents.on('did-navigate-in-page', (_event, url, isMainFrame) => { if (isMainFrame) { tab.url = url; tab.historyVisit = null; publish(); void rememberDesktopPage(tab, view); } });
  view.webContents.on('page-title-updated', (_event, title) => { tab.title = title || tab.url; publish(); });
  view.webContents.on('did-fail-load', (_event, code, description, url, isMainFrame) => {
    if (isMainFrame && code !== -3) { tab.loading = false; tab.title = `Could not load ${url || tab.url}`; tab.error = description; publish(); }
  });
  return view;
}
async function navigate(input, id = activeId) {
  const tab = tabs.find(item => item.id === id);
  if (!tab) throw new Error('Tab not found');
  const value = String(input || '').trim();
  if (!value) return;
  if (!looksLikeUrl(value)) return search(value, id);
  const url = safeUrl(value);
  panel = null;
  tab.searchResults = null;
  tab.url = url;
  tab.title = new URL(url).hostname;
  const view = attachView(tab);
  if (id === activeId) showActive();
  await view.webContents.loadURL(url);
  publish();
}
function runSearchCommand(command, args, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let output = '', error = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs);
    child.stdout.on('data', data => { output += data; if (output.length > 5_000_000) child.kill('SIGKILL'); });
    child.stderr.on('data', data => { error += data; if (error.length > 20_000) child.kill('SIGKILL'); });
    child.on('error', cause => { clearTimeout(timer); reject(cause); });
    child.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) reject(new Error(error.trim() || `Search exited ${code}`));
      else { try { resolve(JSON.parse(output)); } catch (cause) { reject(new Error('Search returned invalid data', { cause })); } }
    });
  });
}
async function search(query, id = activeId, mode = 'web', filters = {}) {
  const tab = tabs.find(item => item.id === id);
  if (!tab) throw new Error('Tab not found');
  const cleanQuery = String(query || '').trim().slice(0, 300);
  if (!cleanQuery) return;
  if (!['web', 'images'].includes(mode)) throw new Error('Unknown search mode');
  if (mode === 'images' && !existsSync(imagePython)) throw new Error('Image search is not installed. Run SETUP_DIAGONAL_BROWSER.command.');
  const safeFilters = {
    size: ['Large', 'Wallpaper'].includes(filters?.size) ? filters.size : undefined,
    color: filters?.color === 'Monochrome' ? 'Monochrome' : undefined,
    type_image: ['gif', 'transparent'].includes(filters?.type_image) ? filters.type_image : undefined,
    layout: ['Wide', 'Tall', 'Square'].includes(filters?.layout) ? filters.layout : undefined,
  };
  const args = mode === 'images'
    ? [path.join(here, 'image_search.py'), cleanQuery, '--max-results', '36',
      ...Object.entries(safeFilters).filter(([, value]) => value).flatMap(([key, value]) => [`--${key.replace('_', '-')}`, value])]
    : [cleanQuery, '--max-results', '8', '--format', 'json'];
  const results = await runSearchCommand(mode === 'images' ? imagePython : ddgs, args, mode === 'images' ? 30000 : 20000);
  if (!Array.isArray(results)) throw new Error('Search returned invalid results');
  let historyId;
  try { historyId = (await searchHistory.save({ kind: 'search', source: 'desktop', query: cleanQuery, mode, results })).id; }
  catch (error) { console.error('Could not save search history:', error.message); }
  panel = null;
  if (tab.view) { window.contentView.removeChildView(tab.view); tab.view.webContents.close(); tab.view = null; }
  tab.url = '';
  tab.title = `${mode === 'images' ? 'Images' : 'Search'}: ${cleanQuery}`;
  tab.searchResults = { query: cleanQuery, results, mode, filters: safeFilters, historyId };
  if (id === activeId) showActive();
  publish();
}
function submitSearch(value) {
  const query = String(value?.query || '').trim();
  if (!query) return;
  if (looksLikeUrl(query)) return navigate(query);
  return search(query, activeId, value?.mode === 'images' ? 'images' : 'web', value?.filters || {});
}
function selectTab(id) {
  if (!tabs.some(tab => tab.id === id)) throw new Error('Tab not found');
  if (recording && recording.tabId !== id) throw new Error('Stop recording before switching tabs');
  activeId = id;
  panel = null;
  showActive();
}
function closeTab(id) {
  if (recording?.tabId === id) throw new Error('Stop recording before closing this tab');
  const index = tabs.findIndex(tab => tab.id === id);
  if (index < 0) return;
  const [tab] = tabs.splice(index, 1);
  if (tab.view) { window.contentView.removeChildView(tab.view); tab.view.webContents.close(); }
  if (activeId === id) activeId = tabs[Math.min(index, tabs.length - 1)]?.id || null;
  panel = null;
  inspection = null;
  if (!tabs.length) newTab();
  showActive();
}
async function screenshot() {
  const tab = current();
  const image = await (tab?.view ? tab.view.webContents : window.webContents).capturePage();
  const saved = artifactPath(undefined, 'png');
  await mkdir(path.dirname(saved), { recursive: true });
  await writeFile(saved, image.toPNG());
  return saved;
}
async function startRecording() {
  if (recording) throw new Error('Recording is already active');
  if (panel) closePanel();
  const tab = current();
  const saved = artifactPath(undefined, 'webm');
  await mkdir(path.dirname(saved), { recursive: true });
  const child = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', '5', '-vcodec', 'png', '-i', 'pipe:0', '-an', '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuv420p', '-deadline', 'realtime', '-cpu-used', '8', '-y', saved], { stdio: ['pipe', 'ignore', 'pipe'] });
  let error = '';
  child.stderr.on('data', data => { error += data.toString(); });
  const exit = new Promise(resolve => {
    child.on('error', cause => { error += cause.message; resolve(-1); });
    child.on('close', resolve);
  });
  const capture = async () => {
    if (!recording || recording.busy || child.stdin.destroyed) return;
    recording.busy = true;
    try {
      const image = await (tab.view ? tab.view.webContents : window.webContents).capturePage();
      const png = image.resize({ width: 1280, height: 720 }).toPNG();
      child.stdin.write(png);
      recording.frames += 1;
    } finally { if (recording) recording.busy = false; }
  };
  recording = { tabId: tab.id, path: saved, child, exit, frames: 0, busy: false, error: '', encoderError: () => error, timer: null, capture };
  await capture();
  recording.timer = setInterval(() => { capture().catch(err => { if (recording) recording.error = err.message; }); }, 200);
  publish();
  return saved;
}
async function stopRecording() {
  if (!recording) throw new Error('No recording is active');
  const session = recording;
  clearInterval(session.timer);
  while (session.busy) await new Promise(resolve => setTimeout(resolve, 20));
  recording = null;
  session.child.stdin.end();
  const exitCode = await session.exit;
  publish();
  if (exitCode !== 0 || session.frames === 0) throw new Error(session.error || session.encoderError() || 'Video encoder failed');
  return session.path;
}

async function listMedia() {
  await mkdir(artifactsDir, { recursive: true });
  const names = await readdir(artifactsDir);
  const items = await Promise.all(names.filter(name => /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.(png|webm)$/.test(name)).map(async name => {
    const file = path.join(artifactsDir, name);
    const info = await stat(file);
    return { name, kind: name.endsWith('.webm') ? 'video' : 'image', url: pathToFileURL(file).href, path: file, size: info.size, updated: info.mtimeMs };
  }));
  return items.sort((a, b) => b.updated - a.updated).slice(0, 100);
}

async function openPanel(name) {
  if (!['media', 'inspect', 'accounts', 'history'].includes(name)) throw new Error('Unknown panel');
  if (recording) throw new Error('Stop recording before opening a panel');
  if (name === 'inspect') {
    const tab = current();
    inspection = tab?.view ? {
      tabId: tab.id, urlInfo: inspectUrl(tab.view.webContents.getURL()),
      page: await tab.view.webContents.executeJavaScript(`(${extractPage.toString()})()`),
      blockedRequests: tab.blocked,
    } : null;
  }
  panel = name;
  showActive();
  return name === 'media' ? listMedia() : inspection;
}

function closePanel() { panel = null; showActive(); }

async function setAdBlock(enabled) {
  adBlockEnabled = Boolean(enabled);
  await writeFile(path.join(profileDir, 'settings.json'), JSON.stringify({ adBlockEnabled }));
  publish();
  return adBlockEnabled;
}

async function fillEmailProfile(nickname) {
  const profile = emailProfileByNickname(emailProfiles, nickname);
  const tab = current();
  if (!tab?.view) throw new Error('Open a sign-in page before using an email nickname');
  const filled = await tab.view.webContents.executeJavaScript(`(() => {
    const email = ${JSON.stringify(profile.email)};
    const fields = [...document.querySelectorAll('input[type="email"], input[autocomplete="email"], input[autocomplete="username"], input[name*="email" i], input[id*="email" i]')];
    const field = fields.find(node => node.getClientRects().length && !node.disabled);
    if (!field) return false;
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(field, email);
    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.dispatchEvent(new Event('change', { bubbles: true }));
    field.focus();
    return true;
  })()`);
  if (!filled) throw new Error('No visible email or username field was found on this page');
  closePanel();
  return { nickname: profile.nickname, email: profile.email, submitted: false };
}

if (hasInstanceLock) app.whenReady().then(async () => {
  if (process.platform === 'darwin' && process.env.DIAGONAL_WEBAUTHN_KEYCHAIN_GROUP) {
    try {
      app.configureWebAuthn({ touchID: { keychainAccessGroup: process.env.DIAGONAL_WEBAUTHN_KEYCHAIN_GROUP, promptReason: 'sign in to $1' } });
    } catch (error) {
      console.error('Touch ID passkeys could not be enabled:', error.message);
    }
  }
  session.defaultSession.on('select-webauthn-account', (_event, details, callback) => {
    const accounts = details.accounts.slice(0, 8);
    const buttons = [...accounts.map(item => item.displayName || item.name || 'Passkey'), 'Cancel'];
    dialog.showMessageBox(window, {
      type: 'question', title: 'Choose a passkey',
      message: `Choose a passkey for ${details.relyingPartyId}`,
      buttons, cancelId: buttons.length - 1, noLink: true,
    }).then(({ response }) => callback(accounts[response]?.credentialId)).catch(() => callback());
  });
  emailProfiles = await loadEmailProfiles(profileDir).catch(() => []);
  app.dock?.setIcon(path.join(here, 'ui/assets/diagonal-wordmark.png'));
  let blockPublishTimer;
  session.defaultSession.webRequest.onBeforeRequest({ urls: ['*://*/*'] }, (details, callback) => {
    const blocked = adBlockEnabled && blockedHost(details.url, details.resourceType);
    if (blocked) {
      blockedCount += 1;
      const tab = tabs.find(item => item.view?.webContents.id === details.webContentsId);
      if (tab) tab.blocked += 1;
      if (!blockPublishTimer) blockPublishTimer = setTimeout(() => { blockPublishTimer = null; publish(); }, 200);
    }
    callback({ cancel: !!blocked });
  });
  window = new BrowserWindow({ width: 1440, height: 900, minWidth: 800, minHeight: 560, title: 'Diagonal Browser', backgroundColor: '#080706', webPreferences: { preload: path.join(here, 'ui/preload.cjs'), nodeIntegration: false, contextIsolation: true, sandbox: true } });
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', event => event.preventDefault());
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'Diagonal Browser', submenu: [{ role: 'about' }, { role: 'quit' }] },
    { label: 'File', submenu: [
      { label: 'New Tab', accelerator: 'CommandOrControl+T', click: () => newTab() },
      { label: 'Close Tab', accelerator: 'CommandOrControl+W', click: () => closeTab(activeId) },
    ] },
    { label: 'Navigate', submenu: [
      { label: 'Address Bar', accelerator: 'CommandOrControl+L', click: () => window.webContents.send('diagonal:focus-address') },
      { label: 'Back', accelerator: 'Alt+Left', click: goBack },
      { label: 'Forward', accelerator: 'Alt+Right', click: goForward },
      { label: 'Reload', accelerator: 'CommandOrControl+R', click: () => current()?.view?.webContents.reload() },
    ] },
    { role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' },
  ]));
  window.on('resize', () => { for (const tab of tabs) if (tab.id === activeId && tab.view) tab.view.setBounds(bounds()); });
  window.on('close', event => {
    if (!recording) return;
    event.preventDefault();
    stopRecording().finally(() => window.close());
  });
  window.on('closed', () => { if (recording) { clearInterval(recording.timer); recording.child.stdin.end(); } for (const tab of tabs) tab.view?.webContents.close(); });
  await window.loadFile(path.join(here, 'ui/index.html'));
  newTab();
  if (process.env.DIAGONAL_DESKTOP_SMOKE === '1') {
    try {
      const homeShot = await screenshot();
      await navigate('https://example.com');
      await current().view.webContents.executeJavaScript("document.head.append(Object.assign(document.createElement('script'), { src: 'https://ads.doubleclick.net/diagonal-smoke.js' }))");
      await new Promise(resolve => setTimeout(resolve, 150));
      if (!current().blocked) throw new Error('Desktop ad blocker did not intercept a known ad host');
      const pageShot = await screenshot();
      await startRecording();
      await new Promise(resolve => setTimeout(resolve, 650));
      const video = await stopRecording();
      const inspected = await openPanel('inspect');
      if (!inspected?.page?.title) throw new Error('Desktop page inspection returned no title');
      closePanel();
      await window.webContents.executeJavaScript("document.querySelector('#media').click()");
      let mediaCount = 0;
      for (let attempt = 0; attempt < 20 && !mediaCount; attempt += 1) {
        await new Promise(resolve => setTimeout(resolve, 100));
        mediaCount = await window.webContents.executeJavaScript("document.querySelectorAll('.media-card').length");
      }
      if (!mediaCount) throw new Error('Capture library did not display media');
      await window.webContents.executeJavaScript("document.querySelector('.media-card').click()");
      const videoPlayable = await window.webContents.executeJavaScript(`new Promise(resolve => {
        const player = document.querySelector('.viewer video');
        if (!player) return resolve(false);
        if (player.readyState > 0) return resolve(true);
        player.addEventListener('loadedmetadata', () => resolve(true), { once: true });
        player.addEventListener('error', () => resolve(false), { once: true });
        setTimeout(() => resolve(false), 4000);
      })`);
      if (!videoPlayable) throw new Error('Capture library could not play the saved WebM');
      closePanel();
      await search('Diagonal Browser');
      console.log(JSON.stringify({ homeShot, pageShot, video, inspectedTitle: inspected.page.title, blocked: blockedCount, mediaCount, videoPlayable, results: current().searchResults.results.length }));
      app.quit();
    } catch (error) {
      console.error(error);
      app.exit(1);
    }
  }
});

ipcMain.handle('diagonal:action', async (event, action, value) => {
  if (!window || event.sender !== window.webContents) throw new Error('Unauthorized browser command');
  switch (action) {
    case 'state': return publicState();
    case 'newTab': return newTab(value);
    case 'selectTab': return selectTab(value);
    case 'closeTab': return closeTab(value);
    case 'navigate': return navigate(value);
    case 'submitSearch': return submitSearch(value);
    case 'search': return search(value?.query, activeId, value?.mode === 'images' ? 'images' : 'web', value?.filters || {});
    case 'home': {
      const tab = current();
      if (tab.view) { window.contentView.removeChildView(tab.view); tab.view.webContents.close(); tab.view = null; }
      tab.url = ''; tab.title = 'New tab'; tab.searchResults = null; panel = null; inspection = null; showActive(); return;
    }
    case 'back': return goBack();
    case 'forward': return goForward();
    case 'reload': return current()?.view?.webContents.reload();
    case 'screenshot': return screenshot();
    case 'startRecording': return startRecording();
    case 'stopRecording': return stopRecording();
    case 'showArtifact': return shell.showItemInFolder(value);
    case 'copyText': clipboard.writeText(String(value || '')); return true;
    case 'openPanel': return openPanel(value);
    case 'closePanel': return closePanel();
    case 'listMedia': return listMedia();
    case 'searchHistory': return searchHistory.search(value || {});
    case 'readHistory': return searchHistory.read(value?.historyId, value || {});
    case 'revisitHistory': return revisitHistory(value);
    case 'setAdBlock': return setAdBlock(value);
    case 'saveEmailProfiles': emailProfiles = await saveEmailProfiles(value, profileDir); publish(); return emailProfiles;
    case 'useEmailProfile': return fillEmailProfile(value);
    case 'copyInspection': if (inspection) clipboard.writeText(JSON.stringify(inspection, null, 2)); return;
    case 'exportInspection': {
      if (!inspection) throw new Error('Inspect a page first');
      const saved = artifactPath(undefined, 'json');
      await mkdir(artifactsDir, { recursive: true });
      await writeFile(saved, JSON.stringify(inspection, null, 2));
      return saved;
    }
    default: throw new Error('Unknown browser command');
  }
});
app.on('window-all-closed', () => app.quit());
