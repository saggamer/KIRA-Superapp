import { Request } from '@ghostery/adblocker';

// Runs before page scripts in every document. Sends only DOM features used by
// cosmetic filters; no page text, input values, cookies, or credentials.
function cosmeticMonitor() {
  if (!/^https?:$/.test(location.protocol)) return;
  if (window.__diagonalShieldMonitor === document) return;
  window.__diagonalShieldMonitor = document;
  window.__diagonalShieldReady = false;
  const seen = { classes: new Set(), ids: new Set(), hrefs: new Set() };
  const pendingNodes = new Set([document]);
  const styles = new Set(); let first = true, timer, busy = false, dirty = true;
  const sheet = new CSSStyleSheet(); document.adoptedStyleSheets = [...document.adoptedStyleSheets, sheet];
  let combined = '';
  async function flush() {
    if (busy) { dirty = true; return; }
    busy = true; dirty = false;
    try {
      const features = { classes: [], ids: [], hrefs: [], first };
      const nodes = new Set();
      for (const root of pendingNodes) {
        if (root.nodeType === Node.ELEMENT_NODE) nodes.add(root);
        for (const node of root.querySelectorAll?.('[id],[class],[href]') || []) { nodes.add(node); if (nodes.size >= 4000) break; }
        if (nodes.size >= 4000) break;
      }
      pendingNodes.clear();
      for (const node of nodes) {
        const add = (key, value) => { if (value && value.length <= 512 && !seen[key].has(value) && seen[key].size < 10000 && features[key].length < 1000) { seen[key].add(value); features[key].push(value); } };
        add('ids', node.id); for (const cls of node.classList) add('classes', cls);
        add('hrefs', node.getAttribute('href'));
      }
      if (!first && !features.classes.length && !features.ids.length && !features.hrefs.length) return;
      first = false;
      const result = await window.__diagonalShield(features);
      if (result?.active && result.styles && !styles.has(result.styles)) {
        styles.add(result.styles); combined += `\n${result.styles}`;
        // Constructed sheets apply under strict style CSP without changing it.
        sheet.replaceSync(combined);
      }
    } catch { /* A detached frame or unsupported selector must not break a site. */ }
    finally { window.__diagonalShieldReady = true; busy = false; if (dirty) schedule(); }
  }
  function schedule() { dirty = true; if (!timer) timer = setTimeout(() => { timer = null; flush(); }, 150); }
  new MutationObserver(records => {
    for (const record of records) {
      if (record.type === 'attributes') pendingNodes.add(record.target);
      else for (const node of record.addedNodes) if (node.nodeType === Node.ELEMENT_NODE) pendingNodes.add(node);
      if (pendingNodes.size >= 200) break;
    }
    if (pendingNodes.size) schedule();
  }).observe(document, { subtree: true, childList: true, attributes: true, attributeFilter: ['id', 'class', 'href'] });
  document.addEventListener('DOMContentLoaded', () => { pendingNodes.add(document); schedule(); }, { once: true });
  schedule();
}

export async function protectPlaywrightContext(context, blocker) {
  await blocker.init();
  const executedScripts = new WeakMap();
  await context.exposeBinding('__diagonalShield', async ({ page, frame }, features) => {
    if (!/^https?:\/\//i.test(frame.url()) || !features || typeof features !== 'object') return { active: false };
    const safe = key => Array.isArray(features[key]) ? features[key].filter(value => typeof value === 'string' && value.length <= 512).slice(0, 1000) : [];
    const request = Request.fromRawDetails({ url: frame.url() });
    const result = blocker.cosmetics({ url: frame.url(), hostname: request.hostname, domain: request.domain,
      classes: safe('classes'), ids: safe('ids'), hrefs: safe('hrefs'), getBaseRules: features.first === true,
      getInjectionRules: features.first === true, getRulesFromHostname: features.first === true, getRulesFromDOM: true, getExtendedRules: false,
    }, page.url());
    let scripts = executedScripts.get(frame);
    if (!scripts || features.first) { scripts = new Set(); executedScripts.set(frame, scripts); }
    for (const script of result.scripts.slice(0, 60)) if (!scripts.has(script)) {
      scripts.add(script);
      await frame.evaluate(script).catch(() => {});
    }
    return { active: result.active, styles: result.styles };
  });
  await context.addInitScript(cosmeticMonitor);
  // Fast window.open navigations can commit before context init scripts attach.
  // The document guard keeps this fallback from installing duplicate observers.
  context.on('page', page => {
    const inject = frame => frame.evaluate(cosmeticMonitor).catch(() => {});
    page.on('domcontentloaded', () => { void inject(page.mainFrame()); });
    page.on('framenavigated', frame => { void inject(frame); });
    void inject(page.mainFrame());
  });
  await context.route('**/*', async route => {
    const original = route.request(); let frame;
    try { frame = original.frame(); } catch { return route.continue(); }
    const topUrl = frame.page().url();
    if (original.isNavigationRequest() && !frame.parentFrame()) return route.continue();
    const type = original.resourceType() === 'document' ? 'sub_frame' : original.resourceType() === 'xhr' || original.resourceType() === 'fetch' ? 'xmlhttprequest' : original.resourceType();
    const request = Request.fromRawDetails({ url: original.url(), sourceUrl: /^https?:/.test(frame.url()) ? frame.url() : topUrl, type, _originalRequestDetails: original });
    const result = blocker.match(request, { topUrl });
    if (result.redirect) {
      const base64 = result.redirect.contentType.endsWith(';base64');
      return route.fulfill({ body: base64 ? Buffer.from(result.redirect.body, 'base64') : result.redirect.body,
        contentType: result.redirect.contentType.replace(/;base64$/, '') });
    }
    if (result.match) return route.abort('blockedbyclient');
    return route.continue(result.rewrite?.url ? { url: result.rewrite.url } : {});
  });
}
