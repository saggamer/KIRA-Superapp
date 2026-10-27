import { FiltersEngine, Request, ENGINE_VERSION, adsAndTrackingLists, parseFilters } from '@ghostery/adblocker';
import { readFileSync, statSync } from 'node:fs';
import { mkdir, writeFile, rename, unlink, rmdir } from 'node:fs/promises';
import { gzipSync, gunzipSync } from 'node:zlib';
import { randomUUID, createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultProfileDir } from './email_profiles.mjs';
import { historyUrl } from './search_history.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const bundledSnapshot = path.join(here, 'filters/ads-and-tracking.snapshot.json.gz');
const types = { mainFrame: 'main_frame', subFrame: 'sub_frame', xhr: 'xmlhttprequest', fetch: 'xmlhttprequest', document: 'main_frame', stylesheet: 'stylesheet', script: 'script', image: 'image', font: 'font', media: 'media', websocket: 'websocket', ping: 'ping', other: 'other' };
const defaults = { enabled: true, cosmetic: true, autoUpdate: true, pausedSites: [], enabledSites: [], customRules: [] };
const stamp = file => { const info = statSync(file); return `${info.mtimeMs}:${info.ino}:${info.size}`; };

async function lockSettings(dir) {
  await mkdir(dir, { recursive: true, mode: 0o700 });
  const lock = path.join(dir, '.settings-lock'); const deadline = Date.now() + 4000;
  while (true) {
    try { await mkdir(lock, { mode: 0o700 }); return () => rmdir(lock).catch(() => {}); }
    catch (error) {
      if (error.code !== 'EEXIST') throw error;
      try { if (Date.now() - statSync(lock).mtimeMs > 30000) { await rmdir(lock).catch(() => {}); continue; } } catch {}
      if (Date.now() >= deadline) throw new Error('Another browser is saving ad blocker settings; try again');
      await new Promise(resolve => setTimeout(resolve, 20));
    }
  }
}

export function siteHostname(value) {
  const url = new URL(String(value).includes('://') ? value : `https://${value}`);
  if (!['https:', 'http:'].includes(url.protocol) || !url.hostname || url.username || url.password) throw new Error('Use an HTTP(S) website URL or hostname');
  return url.hostname.toLowerCase().replace(/\.$/, '');
}

async function atomicWrite(file, value) {
  await mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
  const temporary = `${file}.${randomUUID()}.tmp`;
  try { await writeFile(temporary, value, { mode: 0o600 }); await rename(temporary, file); }
  finally { await unlink(temporary).catch(() => {}); }
}

export async function downloadAdblockSnapshot(fetchImpl = fetch) {
  const resourcesUrl = 'https://raw.githubusercontent.com/ghostery/adblocker/master/packages/adblocker/assets/ublock-origin/resources.json';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 25000);
  try {
    const documents = await Promise.all([...adsAndTrackingLists, resourcesUrl].map(async url => {
      const response = await fetchImpl(url, { signal: controller.signal, redirect: 'error' });
      if (!response.ok) throw new Error(`Filter download failed (${response.status}): ${url}`);
      // Bound every response, including chunked responses without Content-Length.
      const reader = response.body.getReader(); let size = 0; const chunks = [];
      try {
        while (true) { const { done, value } = await reader.read(); if (done) break; size += value.length; if (size > 12_000_000) throw new Error('Filter list exceeded size limit'); chunks.push(Buffer.from(value)); }
      } finally { reader.releaseLock(); }
      return { url, text: Buffer.concat(chunks).toString('utf8') };
    }));
    const resources = documents.pop();
    const filters = documents.map(item => item.text).join('\n');
    const engine = FiltersEngine.parse(filters);
    engine.updateResources(resources.text, createHash('sha256').update(resources.text).digest('hex'));
    const counts = engine.getFilters();
    if (counts.networkFilters.length < 10000 || counts.cosmeticFilters.length < 1000) throw new Error('Downloaded filter lists are incomplete; keeping the last working snapshot');
    const snapshot = { format: 1, engineVersion: ENGINE_VERSION, updatedAt: new Date().toISOString(), sources: documents, resources,
      networkRules: counts.networkFilters.length, cosmeticRules: counts.cosmeticFilters.length, compiled: Buffer.from(engine.serialize()).toString('base64') };
    return gzipSync(JSON.stringify(snapshot), { level: 9 });
  } finally { clearTimeout(timer); controller.abort(); }
}

export class AdBlocker {
  constructor({ profileDir = defaultProfileDir, snapshotPath = bundledSnapshot, onBlocked, topUrlForRequest } = {}) {
    this.dir = path.join(profileDir, 'adblock'); this.profileDir = profileDir; this.snapshotPath = snapshotPath;
    this.settings = { ...defaults, pausedSites: [], enabledSites: [], customRules: [] }; this.onBlocked = onBlocked; this.topUrlForRequest = topUrlForRequest;
    this.engine = null; this.snapshot = null; this.baseBytes = null; this.recent = []; this.blocked = 0; this.redirected = 0; this.cosmeticInjections = 0;
    this.lastSync = 0; this.settingsStamp = null; this.snapshotStamp = null; this.rulesKey = ''; this.pending = Promise.resolve(); this.updating = null;
  }

  async init() {
    if (this.engine) return this;
    let file = path.join(this.dir, 'snapshot.json.gz');
    try { this._loadSnapshot(file); this.snapshotStamp = stamp(file); }
    catch { this._loadSnapshot(this.snapshotPath); }
    this._sync(true);
    if (!this.autoTimer) { this.autoTimer = setInterval(() => { void this.autoUpdate(); }, 3600000); this.autoTimer.unref?.(); }
    return this;
  }

  _loadSnapshot(file) {
    const value = JSON.parse(gunzipSync(readFileSync(file)).toString('utf8'));
    if (value.format !== 1 || value.engineVersion !== ENGINE_VERSION) throw new Error('Incompatible ad blocker snapshot');
    const bytes = Buffer.from(value.compiled, 'base64');
    const engine = FiltersEngine.deserialize(bytes);
    this.baseBytes = bytes; this.engine = engine; this.snapshot = { updatedAt: value.updatedAt, networkRules: value.networkRules, cosmeticRules: value.cosmeticRules, sources: value.sources.map(item => item.url) };
    this.rulesKey = ''; this._applyRules();
  }

  _applyRules() {
    const key = JSON.stringify(this.settings.customRules);
    if (!this.baseBytes || key === this.rulesKey) return;
    const next = FiltersEngine.deserialize(this.baseBytes);
    next.updateFromDiff({ added: this.settings.customRules });
    this.engine = next; this.rulesKey = key;
  }

  _sync(force = false) {
    if (!force && Date.now() - this.lastSync < 250) return;
    this.lastSync = Date.now();
    try {
      const file = path.join(this.dir, 'settings.json'); const version = stamp(file);
      if (version !== this.settingsStamp) {
        const value = JSON.parse(readFileSync(file, 'utf8'));
        this.settings = { enabled: value.enabled !== false, cosmetic: value.cosmetic !== false, autoUpdate: value.autoUpdate !== false,
          pausedSites: (value.pausedSites || []).slice(0, 500).map(siteHostname), enabledSites: (value.enabledSites || []).slice(0, 500).map(siteHostname), customRules: (value.customRules || []).slice(0, 500).map(String) };
        this.settingsStamp = version; this._applyRules();
      }
    } catch (error) {
      if (error.code === 'ENOENT' && this.settingsStamp === null) {
        try { this.settings.enabled = JSON.parse(readFileSync(path.join(this.profileDir, 'settings.json'), 'utf8')).adBlockEnabled !== false; } catch {}
      }
    }
    try {
      const file = path.join(this.dir, 'snapshot.json.gz'); const version = stamp(file);
      if (version !== this.snapshotStamp) { this._loadSnapshot(file); this.snapshotStamp = version; }
    } catch { /* Keep the last working engine during offline use or a bad update. */ }
  }

  enabledFor(url) {
    this._sync();
    if (!this.settings.enabled) return false;
    return this.siteEnabled(url);
  }

  siteEnabled(url) {
    let host; try { host = siteHostname(url); } catch { return true; }
    let length = -1, enabled = true;
    for (const [sites, value] of [[this.settings.pausedSites, false], [this.settings.enabledSites, true]]) {
      for (const site of sites) if ((host === site || host.endsWith(`.${site}`)) && site.length > length) { length = site.length; enabled = value; }
    }
    return enabled;
  }

  match(request, { topUrl, record = true } = {}) {
    this._sync();
    topUrl ||= this.topUrlForRequest?.(request) || request.sourceUrl;
    if (request.isMainFrame() || !this.enabledFor(topUrl) || !this.engine) return { match: false, redirect: undefined, exception: undefined, filter: undefined };
    const result = this.engine.match(request);
    if (record && (result.match || result.redirect)) {
      this.blocked++; if (result.redirect) this.redirected++;
      const item = { time: new Date().toISOString(), url: historyUrl(request.url).slice(0, 2000), pageUrl: historyUrl(topUrl).slice(0, 2000), type: request.type,
        action: result.redirect ? 'replaced' : 'blocked', rule: result.filter?.toString().slice(0, 500) || 'Filter list match' };
      this.recent.push(item); if (this.recent.length > 100) this.recent.shift();
      this.onBlocked?.(item, request);
    }
    return result;
  }

  cosmetics(details, topUrl = details.url) {
    this._sync();
    if (!this.settings.cosmetic || !this.enabledFor(topUrl) || !this.engine) return { active: false, styles: '', scripts: [], extended: [] };
    const result = this.engine.getCosmeticsFilters(details);
    if (result.styles || result.scripts.length) this.cosmeticInjections++;
    return result;
  }

  status(url) {
    this._sync(true);
    let hostname; try { if (url) hostname = siteHostname(url); } catch {}
    return { ...this.settings, engine: 'Ghostery', engineVersion: ENGINE_VERSION, ready: !!this.engine, ...this.snapshot,
      ...(hostname ? { site: hostname, siteEnabled: this.siteEnabled(url), activeOnSite: this.enabledFor(url) } : {}),
      blocked: this.blocked, redirected: this.redirected, cosmeticInjections: this.cosmeticInjections,
      recent: this.recent.filter(item => !hostname || (() => { try { return siteHostname(item.pageUrl) === hostname; } catch { return false; } })()).slice(-15).reverse(),
      countersScope: 'this browser process', updating: !!this.updating, updateError: this.updateError || null,
    };
  }

  configure(changes = {}) {
    return this._mutate(() => {
      for (const key of ['enabled', 'cosmetic', 'autoUpdate']) if (changes[key] !== undefined) {
        if (typeof changes[key] !== 'boolean') throw new Error(`${key} must be true or false`);
        this.settings[key] = changes[key];
      }
      if (changes.siteEnabled !== undefined) {
        if (typeof changes.siteEnabled !== 'boolean' || !changes.site) throw new Error('siteEnabled requires a site URL or hostname');
        const host = siteHostname(changes.site);
        if (changes.siteEnabled) {
          this.settings.pausedSites = this.settings.pausedSites.filter(site => site !== host);
          if (!this.siteEnabled(host) && !this.settings.enabledSites.includes(host)) {
            if (this.settings.enabledSites.length >= 500) throw new Error('The site exception limit is 500');
            this.settings.enabledSites.push(host);
          }
        } else {
          this.settings.enabledSites = this.settings.enabledSites.filter(site => site !== host);
          if (!this.settings.pausedSites.includes(host)) {
            if (this.settings.pausedSites.length >= 500) throw new Error('The paused site limit is 500');
            this.settings.pausedSites.push(host);
          }
        }
      }
    }, changes.site);
  }

  rule(operation, rule) {
    if (!['add', 'remove'].includes(operation) || typeof rule !== 'string' || rule.length > 4000 || !rule.trim() || /[\r\n]/.test(rule)) throw new Error('Use add/remove and one filter rule of up to 4,000 characters');
    rule = rule.trim();
    const parsed = parseFilters(rule);
    if (parsed.networkFilters.length + parsed.cosmeticFilters.length !== 1) throw new Error('Invalid or unsupported adblock filter rule');
    return this._mutate(() => {
      if (operation === 'remove') this.settings.customRules = this.settings.customRules.filter(item => item !== rule);
      else if (!this.settings.customRules.includes(rule)) {
        if (this.settings.customRules.length >= 500) throw new Error('The custom rule limit is 500');
        this.settings.customRules.push(rule);
      }
    });
  }

  _mutate(fn, url) {
    const operation = this.pending.catch(() => {}).then(async () => {
      await this.init(); const release = await lockSettings(this.dir);
      try {
        this._sync(true); const before = structuredClone(this.settings);
        try { fn(); await atomicWrite(path.join(this.dir, 'settings.json'), JSON.stringify(this.settings)); }
        catch (error) { this.settings = before; throw error; }
        this.settingsStamp = null; this._sync(true);
        return { ...this.status(url), reloadRequired: true, message: 'Settings saved. Reload open pages to fully apply or remove cosmetic filters and scriptlets.' };
      } finally { await release(); }
    });
    this.pending = operation; return operation;
  }

  async check(url, pageUrl = url, resourceType = 'script') {
    await this.init(); this._sync(true);
    for (const value of [url, pageUrl]) if (!/^https?:\/\//i.test(value)) throw new Error('Use HTTP(S) URLs for the request and source page');
    const request = Request.fromRawDetails({ url, sourceUrl: pageUrl, type: types[resourceType] || resourceType });
    const result = this.match(request, { topUrl: pageUrl, record: false });
    return { url, pageUrl, resourceType, blocked: result.match, replaced: !!result.redirect, enabled: this.enabledFor(pageUrl),
      rule: result.filter?.toString() || null, exception: result.exception?.toString() || null, networkRequestMade: false };
  }

  update() {
    if (this.updating) return this.updating;
    this.updating = (async () => {
      try {
        const bytes = await downloadAdblockSnapshot();
        await atomicWrite(path.join(this.dir, 'snapshot.json.gz'), bytes);
        this.snapshotStamp = null; this._sync(true); this.updateError = null;
        return { ...this.status(), updating: false, reloadRequired: true };
      } catch (error) { this.updateError = error.message; throw error; }
      finally { this.updating = null; }
    })();
    return this.updating;
  }

  async autoUpdate() {
    this._sync(true);
    if (process.env.DIAGONAL_ADBLOCK_AUTO_UPDATE === '0' || !this.settings.autoUpdate || Date.now() - Date.parse(this.snapshot.updatedAt) < 86400000) return;
    const attempt = path.join(this.dir, 'last-update-attempt');
    try { if (Date.now() - statSync(attempt).mtimeMs < 3600000) return; } catch {}
    try { await atomicWrite(attempt, new Date().toISOString()); await this.update(); }
    catch { /* Offline operation keeps the bundled or last good lists. */ }
  }

  dispose() { clearInterval(this.autoTimer); this.autoTimer = null; }
}
