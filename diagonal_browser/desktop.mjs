import { app, BrowserWindow, WebContentsView, Menu, clipboard, dialog, ipcMain, session, shell } from 'electron';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { mkdir, readdir, stat, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { artifactPath } from './browser.mjs';
import { AdBlocker } from './adblock.mjs';
import { protectElectronSession } from './adblock_electron.mjs';
import { extractPage, inspectUrl } from './page_inspection.mjs';
import { loadEmailProfiles, saveEmailProfiles, emailProfileByNickname } from './email_profiles.mjs';
import { SearchHistory } from './search_history.mjs';
import { AiSession } from './ai_session.mjs';
import { KiraModel } from './ai_model.mjs';
import { pageSnapshot, pageAction } from './ai_page.mjs';
import { supportsKiraPopup } from './ai_platform.mjs';
import { TabNavigation, liveContents, countBlockedRequest } from './desktop_navigation.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const localDdgs = path.join(os.homedir(), 'Downloads/chatgpt_antigravity_bridge/bin/ddgs');
const ddgs = process.env.DIAGONAL_DDGS_BIN || (existsSync(localDdgs) ? localDdgs : 'ddgs');
const imagePython = process.env.DIAGONAL_IMAGE_PYTHON || path.join(here, '.venv/bin/python');
const ffmpeg = process.env.DIAGONAL_FFMPEG_BIN || 'ffmpeg';
const toolbarHeight = 104;
const artifactsDir = path.dirname(artifactPath('gallery.png', 'png'));
const profileDir = process.env.DIAGONAL_DESKTOP_SMOKE === '1'
  ? path.join(app.getPath('temp'), `diagonal-browser-smoke-${process.pid}`)
  : process.env.DIAGONAL_PROFILE_DIR || path.join(here, '.profile');
const searchHistory = new SearchHistory(profileDir);
mkdirSync(profileDir, { recursive: true });
mkdirSync(path.join(profileDir, 'session'), { recursive: true });
let window;
let tabs = [];
const closedTabs = [];
let activeId = null;
let recording = null;
let panel = null;
let inspection = null;
const adblock = new AdBlocker({ profileDir, onBlocked: (_item, request) => {
  countBlockedRequest(tabs, request);
  if (!blockPublishTimer) blockPublishTimer = setTimeout(() => { blockPublishTimer = null; publish(); }, 200);
} });
let blockPublishTimer;
let emailProfiles = [];
const kiraModel = new KiraModel(path.dirname(here));
let aiScope = null;
const ai = new AiSession({
  model: (messages, signal) => kiraModel.complete(messages, signal),
  observe: observeAiPage,
  act: actOnAiPage,
  research: async action => {
    if (recording) throw new Error('Stop recording before opening a research tab');
    const id = newTab();
    aiScope = { id, origin: null };
    if (action.action === 'search') await search(action.query, id, 'web', {}, 'llm');
    else { aiScope.origin = new URL(action.url).origin; await navigate(action.url, id); }
    return observeAiPage();
  },
  changed: () => { if (window && !window.isDestroyed()) { for (const tab of tabs) if (tab.id === activeId && liveContents(tab.view)) tab.view.setBounds(bounds()); publish(); } },
});
function bindAiPage() { const tab = current(); aiScope = tab ? { id: tab.id, origin: tab.url ? new URL(tab.url).origin : null } : null; }
async function observeAiPage() {
  const tab = current();
  if (!tab || tab.id !== aiScope?.id) throw new Error('The active tab changed. Send a new message to use this page.');
  if (!isWebsite(tab) || !liveContents(tab.view)) return { title: tab.title, search: tab.searchResults || null, text: 'No website is open. Ask permission to search or open a website.' };
  if (new URL(tab.view.webContents.getURL()).origin !== aiScope.origin) throw new Error('The website changed. Send a new message to use this website.');
  return tab.view.webContents.executeJavaScriptInIsolatedWorld(999, [{ code: `(${pageSnapshot.toString()})()` }]);
}
async function actOnAiPage(action, observation) {
  if (action.action === 'read') return { summary: 'Read the current page' };
  const tab = current();
  if (!ai.state.open || tab?.id !== aiScope?.id || !tab?.view || tab.view.webContents.getURL() !== observation.url || new URL(observation.url).origin !== aiScope.origin) throw new Error('The page changed. Send a new message before continuing.');
  const result = await tab.view.webContents.executeJavaScriptInIsolatedWorld(999, [{ code: `(() => { try { return (${pageAction.toString()})(${JSON.stringify(action)}); } catch (error) { return { error: error.message }; } })()` }]);
  if (result?.error) throw new Error(result.error);
  await new Promise(resolve => setTimeout(resolve, 300));
  return result;
}
function aiNavigation(event, url, tab) {
  if (event.isMainFrame === false) return;
  if (!ai.state.open || tab.id !== activeId || tab.id !== aiScope?.id) return;
  try {
    if (new URL(url).origin === aiScope.origin) return;
    event.preventDefault(); if (!ai.state.pending) ai.request({ action: 'open', url: safeUrl(url), reason: 'The page requested navigation to another website' });
  } catch { event.preventDefault(); }
}

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
function goBack() { return traverseTab(-1); }
function goForward() { return traverseTab(1); }
function isWebsite(tab) { return !!tab?.url && !tab.searchResults; }
function retireView(tab) {
  const view = tab.view; const contents = liveContents(view);
  tab.view = null;
  if (view && window && !window.isDestroyed()) window.contentView.removeChildView(view);
  contents?.close();
}
function setSurface(tab, entry, { remember = true } = {}) {
  tab.navigationEpoch = (tab.navigationEpoch || 0) + 1;
  liveContents(tab.view)?.stop();
  tab.restoring = null;
  tab.url = ''; tab.title = entry.title; tab.searchResults = entry.searchResults ? structuredClone(entry.searchResults) : null;
  tab.loading = false; tab.historyVisit = null; tab.error = null;
  if (remember) tab.navigation.surface(entry);
  panel = null; inspection = null;
  if (tab.id === activeId) showActive(); else publish();
}
function commitPage(tab, view, url) {
  const contents = liveContents(view);
  if (tab.view !== view || !contents || !tabs.includes(tab) || !isWebsite(tab)) return;
  tab.url = url; tab.searchResults = null; tab.historyVisit = null;
  tab.navigation.page({ url, title: tab.title, nativeIndex: contents.navigationHistory.getActiveIndex(), viewId: contents.id }, { replace: tab.navigation.current === tab.restoring });
  tab.restoring = null;
  if (tab.id === activeId) showActive(); else publish();
}
async function traverseTab(direction) {
  if (recording) throw new Error('Stop recording before navigating through history');
  if (ai.state.busy || ai.state.pending) ai.stop();
  const tab = current(); const entry = tab?.navigation.move(direction);
  if (!entry) return;
  tab.navigationEpoch = (tab.navigationEpoch || 0) + 1;
  panel = null; inspection = null;
  liveContents(tab.view)?.stop();
  if (entry.kind !== 'page') { setSurface(tab, entry, { remember: false }); return; }
  tab.url = entry.url; tab.title = entry.title; tab.searchResults = null; tab.error = null;
  const contents = liveContents(attachView(tab));
  const history = contents.navigationHistory;
  const nativeEntry = entry.viewId === contents.id ? history.getEntryAtIndex(entry.nativeIndex) : null;
  tab.loading = history.getActiveIndex() !== entry.nativeIndex || contents.getURL() !== entry.url;
  tab.restoring = tab.loading ? entry : null;
  showActive();
  if (nativeEntry?.url === entry.url && (history.getActiveIndex() !== entry.nativeIndex || contents.getURL() === entry.url)) {
    if (history.getActiveIndex() !== entry.nativeIndex) history.goToIndex(entry.nativeIndex);
  } else {
    try { await contents.loadURL(entry.url); }
    catch (error) { if (error.errno !== -3 && error.code !== 'ERR_ABORTED') throw error; }
    finally { if (tab.restoring === entry) tab.restoring = null; }
  }
  publish();
}
function publicState() {
  const shield = adblock.status(current()?.url);
  return {
    tabs: tabs.map(({ id, title, url, loading, searchResults, blocked, navigation }) => ({
      id, title, url, loading, searchResults, blocked,
      canGoBack: !recording && navigation.canGoBack,
      canGoForward: !recording && navigation.canGoForward,
    })),
    activeId,
    recording: recording ? { tabId: recording.tabId, path: recording.path } : null,
    panel, inspection, adBlockEnabled: shield.enabled, adBlockSiteEnabled: shield.siteEnabled, blockedCount: shield.blocked, emailProfiles,
    ai: { ...ai.state, available: supportsKiraPopup(), model: kiraModel.label() },
  };
}
function publish() {
  if (window && !window.isDestroyed()) window.webContents.send('diagonal:state', publicState());
}
function bounds() {
  const [width, height] = window.getContentSize();
  return { x: 0, y: toolbarHeight, width: Math.max(1, width - (ai.state.open ? 390 : 0)), height: Math.max(1, height - toolbarHeight) };
}
function showActive() {
  for (const tab of tabs) {
    const contents = liveContents(tab.view);
    if (!contents) continue;
    if (tab.id === activeId && !panel && isWebsite(tab)) {
      window.contentView.addChildView(tab.view);
      tab.view.setBounds(bounds());
      contents.focus();
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
  const tab = { id: randomUUID().slice(0, 8), title: 'New tab', url: '', loading: false, searchResults: null, view: null, blocked: 0, navigation: new TabNavigation() };
  tabs.push(tab);
  activeId = tab.id;
  panel = null;
  showActive();
  if (url) navigate(url, tab.id);
  return tab.id;
}

async function rememberDesktopPage(tab, view) {
  if (tab.view !== view || !liveContents(view) || !isWebsite(tab)) return;
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
  setSurface(tab, { kind: 'search', title: `Saved search: ${entry.query}`, searchResults: { query: entry.query, results: entry.results, mode: entry.mode, savedAt: entry.createdAt, historyId: entry.id } });
}

function attachView(tab) {
  if (liveContents(tab.view)) return tab.view;
  if (tab.view) retireView(tab);
  const view = new WebContentsView({ webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true } });
  tab.view = view;
  view.webContents.setWindowOpenHandler(({ url }) => {
    if (ai.state.open && tab.id === activeId && tab.id === aiScope?.id) {
      if (/^https?:\/\//i.test(url) && !ai.state.pending) ai.request({ action: 'open', url, reason: 'This page requested a new tab' });
      return { action: 'deny' };
    }
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
    else aiNavigation(event, event.url, tab);
  });
  view.webContents.on('will-redirect', event => {
    if (!/^https?:\/\//i.test(event.url)) event.preventDefault();
    else aiNavigation(event, event.url, tab);
  });
  view.webContents.on('destroyed', () => { if (tab.view === view) { tab.view = null; tab.loading = false; publish(); } });
  view.webContents.on('did-start-loading', () => { if (tab.view === view && isWebsite(tab)) { tab.loading = true; publish(); } });
  view.webContents.on('did-stop-loading', () => { if (tab.view === view) { tab.loading = liveContents(view)?.isLoading() || false; publish(); void rememberDesktopPage(tab, view); } });
  view.webContents.on('did-navigate', (_event, url) => commitPage(tab, view, url));
  view.webContents.on('did-navigate-in-page', (_event, url, isMainFrame) => { if (isMainFrame) { commitPage(tab, view, url); void rememberDesktopPage(tab, view); } });
  view.webContents.on('page-title-updated', (_event, title) => {
    if (tab.view !== view || !isWebsite(tab)) return;
    tab.title = title || tab.url;
    if (tab.navigation.current.kind === 'page') tab.navigation.current.title = tab.title;
    publish();
  });
  view.webContents.on('did-fail-load', (_event, code, description, url, isMainFrame) => {
    if (tab.view === view && isWebsite(tab) && isMainFrame && code !== -3) { tab.loading = false; tab.title = `Could not load ${url || tab.url}`; tab.error = description; publish(); }
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
  tab.navigationEpoch = (tab.navigationEpoch || 0) + 1;
  tab.restoring = null;
  tab.searchResults = null;
  tab.url = url;
  tab.title = new URL(url).hostname;
  tab.loading = true;
  const view = attachView(tab);
  if (id === activeId) showActive();
  try { await view.webContents.loadURL(url); }
  catch (error) { if (error.errno !== -3 && error.code !== 'ERR_ABORTED') throw error; }
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
async function search(query, id = activeId, mode = 'web', filters = {}, source = 'desktop') {
  const tab = tabs.find(item => item.id === id);
  if (!tab) throw new Error('Tab not found');
  const cleanQuery = String(query || '').trim().slice(0, 300);
  if (!cleanQuery) return;
  if (!['web', 'images'].includes(mode)) throw new Error('Unknown search mode');
  if (mode === 'images' && !existsSync(imagePython)) throw new Error('Image search is not installed. Run SETUP_DIAGONAL_BROWSER.command.');
  const navigationEpoch = tab.navigationEpoch = (tab.navigationEpoch || 0) + 1;
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
  if (!tabs.includes(tab) || tab.navigationEpoch !== navigationEpoch) return;
  let historyId;
  try { historyId = (await searchHistory.save({ kind: 'search', source, query: cleanQuery, mode, results })).id; }
  catch (error) { console.error('Could not save search history:', error.message); }
  if (!tabs.includes(tab) || tab.navigationEpoch !== navigationEpoch) return;
  panel = null;
  setSurface(tab, { kind: 'search', title: `${mode === 'images' ? 'Images' : 'Search'}: ${cleanQuery}`, searchResults: { query: cleanQuery, results, mode, filters: safeFilters, historyId } });
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
  if (tab.url || tab.searchResults) { closedTabs.push({ url: tab.url, title: tab.title, searchResults: structuredClone(tab.searchResults) }); if (closedTabs.length > 20) closedTabs.shift(); }
  retireView(tab);
  if (activeId === id) activeId = tabs[Math.min(index, tabs.length - 1)]?.id || null;
  panel = null;
  inspection = null;
  if (!tabs.length) newTab();
  showActive();
}
async function reopenClosedTab() {
  if (recording) throw new Error('Stop recording before restoring a tab');
  const saved = closedTabs.pop(); if (!saved) return;
  const id = newTab();
  if (saved.url) await navigate(saved.url, id);
  else { const tab = tabs.find(item => item.id === id); setSurface(tab, { kind: 'search', title: saved.title, searchResults: saved.searchResults }); }
  return id;
}
function reloadTab() {
  const tab = current();
  if (tab?.searchResults) return search(tab.searchResults.query, tab.id, tab.searchResults.mode, tab.searchResults.filters || {});
  return isWebsite(tab) ? liveContents(tab.view)?.reload() : undefined;
}
async function screenshot() {
  const tab = current();
  const image = await (isWebsite(tab) ? liveContents(tab.view) || window.webContents : window.webContents).capturePage();
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
      const image = await (isWebsite(tab) ? liveContents(tab.view) || window.webContents : window.webContents).capturePage();
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
  if (!['media', 'inspect', 'accounts', 'history', 'shield'].includes(name)) throw new Error('Unknown panel');
  if (recording) throw new Error('Stop recording before opening a panel');
  if (name === 'inspect') {
    const tab = current();
    const contents = isWebsite(tab) ? liveContents(tab.view) : null;
    const page = contents ? await contents.executeJavaScript(`(${extractPage.toString()})()`) : null;
    inspection = page ? {
      tabId: tab.id, urlInfo: inspectUrl(page.url || tab.url),
      page,
      blockedRequests: tab.blocked,
    } : null;
  }
  panel = name;
  showActive();
  return name === 'media' ? listMedia() : inspection;
}

function closePanel() { panel = null; showActive(); }

async function setAdBlock(enabled) {
  const state = await adblock.configure({ enabled: Boolean(enabled) });
  publish();
  return state.enabled;
}

async function fillEmailProfile(nickname) {
  const profile = emailProfileByNickname(emailProfiles, nickname);
  const tab = current();
  if (!isWebsite(tab) || !liveContents(tab.view)) throw new Error('Open a sign-in page before using an email nickname');
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
  await protectElectronSession(session.defaultSession, adblock);
  void adblock.autoUpdate();
  window = new BrowserWindow({ width: 1440, height: 900, minWidth: 800, minHeight: 560, title: 'Diagonal Browser', backgroundColor: '#080706', webPreferences: { preload: path.join(here, 'ui/preload.cjs'), nodeIntegration: false, contextIsolation: true, sandbox: true } });
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', event => event.preventDefault());
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'Diagonal Browser', submenu: [{ role: 'about' }, { role: 'quit' }] },
    { label: 'File', submenu: [
      { label: 'New Tab', accelerator: 'CommandOrControl+T', click: () => newTab() },
      { label: 'Close Tab', accelerator: 'CommandOrControl+W', click: () => closeTab(activeId) },
      { label: 'Reopen Closed Tab', accelerator: 'CommandOrControl+Shift+T', click: () => { void reopenClosedTab(); } },
    ] },
    { label: 'Navigate', submenu: [
      { label: 'Address Bar', accelerator: 'CommandOrControl+L', click: () => window.webContents.send('diagonal:focus-address') },
      { label: 'Back', accelerator: 'Alt+Left', click: () => { void goBack().catch(error => console.error('Navigation:', error.message)); } },
      { label: 'Forward', accelerator: 'Alt+Right', click: () => { void goForward().catch(error => console.error('Navigation:', error.message)); } },
      { label: 'Reload', accelerator: 'CommandOrControl+R', click: reloadTab },
      { label: 'Search History', accelerator: 'CommandOrControl+Shift+H', click: () => { void openPanel('history'); } },
      ...Array.from({ length: 9 }, (_, index) => ({ label: `Select Tab ${index + 1}`, accelerator: `CommandOrControl+${index + 1}`, click: () => { const tab = index === 8 ? tabs.at(-1) : tabs[index]; if (tab) selectTab(tab.id); } })),
    ] },
    { role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' },
  ]));
  window.on('resize', () => { for (const tab of tabs) if (tab.id === activeId && liveContents(tab.view)) tab.view.setBounds(bounds()); });
  window.on('close', event => {
    if (!recording) return;
    event.preventDefault();
    stopRecording().finally(() => window.close());
  });
  window.on('closed', () => { ai.close(); kiraModel.dispose(); if (recording) { clearInterval(recording.timer); recording.child.stdin.end(); } for (const tab of tabs) liveContents(tab.view)?.close(); });
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
      console.log(JSON.stringify({ homeShot, pageShot, video, inspectedTitle: inspected.page.title, blocked: adblock.blocked, mediaCount, videoPlayable, results: current().searchResults.results.length }));
      app.quit();
    } catch (error) {
      console.error(error);
      app.exit(1);
    }
  }
});

ipcMain.handle('diagonal:action', async (event, action, value) => {
  if (!window || event.sender !== window.webContents) throw new Error('Unauthorized browser command');
  if (typeof action !== 'string') throw new Error('Invalid browser command');
  if (action.startsWith('ai') && !supportsKiraPopup()) throw new Error('Kira AI mode is available only in the macOS Apple Silicon MLX edition');
  if (['newTab', 'reopenClosedTab', 'selectTab', 'closeTab', 'navigate', 'submitSearch', 'search', 'home', 'back', 'forward', 'reload', 'revisitHistory', 'openPanel'].includes(action) && (ai.state.busy || ai.state.pending)) ai.stop();
  switch (action) {
    case 'aiToggle': {
      if (ai.state.open) ai.close();
      else { panel = null; bindAiPage(); ai.open(); }
      showActive(); window.webContents.focus(); return;
    }
    case 'aiClose': ai.close(); showActive(); return;
    case 'aiStop': ai.stop(); return;
    case 'aiSend': if (!ai.state.busy && !ai.state.pending) bindAiPage(); return ai.send(value);
    case 'aiPermission': return ai.decide(value?.id, value?.allowed);
    case 'state': return publicState();
    case 'newTab': return newTab(value);
    case 'reopenClosedTab': return reopenClosedTab();
    case 'selectTab': return selectTab(value);
    case 'closeTab': return closeTab(value);
    case 'navigate': return navigate(value);
    case 'submitSearch': return submitSearch(value);
    case 'search': return search(value?.query, activeId, value?.mode === 'images' ? 'images' : 'web', value?.filters || {});
    case 'home': {
      const tab = current();
      setSurface(tab, { kind: 'home', title: 'New tab' }); return;
    }
    case 'back': return goBack();
    case 'forward': return goForward();
    case 'reload': return reloadTab();
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
    case 'adblockStatus': return adblock.status(current()?.url);
    case 'adblockSet': { const result = await adblock.configure(value || {}); publish(); return result; }
    case 'adblockRule': { const result = await adblock.rule(value?.operation, value?.rule); publish(); return result; }
    case 'adblockCheck': return adblock.check(value?.url, value?.pageUrl || current()?.url, value?.resourceType || 'script');
    case 'adblockUpdate': { const result = await adblock.update(); publish(); return result; }
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
