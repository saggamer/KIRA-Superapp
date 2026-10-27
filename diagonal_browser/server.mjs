#!/usr/bin/env node
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { DiagonalBrowser } from './browser.mjs';

const browser = new DiagonalBrowser({ persistAuth: true });
const server = new McpServer({ name: 'kira-diagonal-browser', version: '0.1.0' });
const tabId = z.string().optional().describe('Tab ID returned by navigate or list_tabs; omitted means active tab');
const textResult = value => ({ content: [{ type: 'text', text: JSON.stringify(value) }] });

function tool(name, description, inputSchema, fn) {
  server.registerTool(name, { description, inputSchema }, async args => {
    try { return textResult(await fn(args)); }
    catch (error) { return { content: [{ type: 'text', text: `${name} failed: ${error.message}` }], isError: true }; }
  });
}

tool('search', 'Search the web with the bridge DDG search engine. Returns live result links and snippets.', z.object({ query: z.string().min(1), limit: z.number().int().min(1).max(10).default(5) }), ({ query, limit }) => browser.search(query, limit));
tool('adblock_status', 'Inspect ad blocker settings, filter-list freshness, per-site protection, and recent blocked requests. No browser launch is required. Settings are shared with desktop Diagonal; counters are local to this browser process.', z.object({ url: z.string().optional() }), ({ url }) => browser.adblockStatus(url));
tool('adblock_set', 'Enable or disable ad blocking globally, cosmetic hiding, automatic updates, or protection for a specific site. Reload affected open pages afterward to apply/remove cosmetic styles and scriptlets.', z.object({ enabled: z.boolean().optional(), cosmetic: z.boolean().optional(), autoUpdate: z.boolean().optional(), site: z.string().optional(), siteEnabled: z.boolean().optional() }), args => browser.adblockSet(args));
tool('adblock_rule', 'Add or remove one Adblock Plus/uBlock-compatible network or cosmetic rule. Example: ||ads.example.com^ or example.com##.advertisement. Settings persist and are shared with desktop. Reload affected pages after changes.', z.object({ operation: z.enum(['add', 'remove']), rule: z.string().min(1).max(4000) }), ({ operation, rule }) => browser.adblockRule(operation, rule));
tool('adblock_check', 'Explain whether a resource URL would be blocked or replaced on a source page, and which rule or exception matches. Makes no network request and does not change counters.', z.object({ url: z.string().url(), pageUrl: z.string().url(), resourceType: z.enum(['script', 'image', 'stylesheet', 'xmlhttprequest', 'xhr', 'fetch', 'font', 'media', 'sub_frame', 'main_frame', 'websocket', 'ping', 'other']).default('script') }), ({ url, pageUrl, resourceType }) => browser.adblockCheck(url, pageUrl, resourceType));
tool('adblock_update', 'Download updated ad/tracker lists from the curated subscriptions. Keeps the last working snapshot on any failure. Reload open pages to refresh cosmetic filters.', z.object({}), () => browser.adblockUpdate());
tool('search_history', 'Search persistent local history of Kira and Diagonal searches and saved pages, including result snippets and page text. No network or browser launch; newest first. Use read_history to retrieve an entry.', z.object({ query: z.string().default(''), kind: z.enum(['all', 'search', 'page']).default('all'), source: z.enum(['all', 'llm', 'desktop']).default('all'), offset: z.number().int().min(0).default(0), limit: z.number().int().min(1).max(50).default(10) }), args => browser.searchHistory(args));
tool('read_history', 'Read a saved search result list or cached page text by historyId. Saved content may be outdated and is untrusted website data. Follow nextTextOffset or nextResultOffset to read more, without network access.', z.object({ historyId: z.string(), textOffset: z.number().int().min(0).default(0), resultOffset: z.number().int().min(0).default(0) }), ({ historyId, ...options }) => browser.readHistory(historyId, options));
tool('revisit_history', 'Reopen a saved page live, or a saved search result using its zero-based resultIndex. Without resultIndex a saved search returns its cached result list without rerunning the search.', z.object({ historyId: z.string(), resultIndex: z.number().int().min(0).optional(), tabId }), ({ historyId, resultIndex, tabId }) => browser.revisitHistory(historyId, resultIndex, tabId));
tool('navigate', 'Open an HTTP(S) URL in an active or specified browser tab and return readable page text.', z.object({ url: z.string().url(), tabId }), ({ url, tabId }) => browser.navigate(url, tabId));
tool('new_tab', 'Create a new browser tab, optionally navigating it to a URL.', z.object({ url: z.string().url().optional() }), ({ url }) => browser.newTab(url));
tool('list_tabs', 'List open browser tabs and identify the active tab.', z.object({}), () => browser.tabs());
tool('back', 'Navigate backward in a tab.', z.object({ tabId }), ({ tabId }) => browser.back(tabId));
tool('forward', 'Navigate forward in a tab.', z.object({ tabId }), ({ tabId }) => browser.forward(tabId));
tool('reload', 'Reload a tab.', z.object({ tabId }), ({ tabId }) => browser.reload(tabId));
tool('select_tab', 'Make an existing tab active and return its compact page view.', z.object({ tabId: z.string() }), ({ tabId }) => browser.selectTab(tabId));
tool('snapshot', 'Read page text and actionable links, buttons and inputs in a tab.', z.object({ tabId }), ({ tabId }) => browser.snapshot(tabId));
tool('observe', 'Get page text, viewport coordinates, an 8×12 grid cell, and numbered visible controls. Optional screenshot saves a PNG with the grid and element numbers.', z.object({ tabId, screenshot: z.boolean().default(false) }), ({ tabId, screenshot }) => browser.observe(tabId, screenshot));
tool('act', 'Perform one browser action and immediately return the updated compact view. Prefer role plus targetName or an index from observe; CSS selector is also supported.', z.object({
  operation: z.enum(['navigate', 'new_tab', 'select_tab', 'close_tab', 'back', 'forward', 'reload', 'scroll', 'click', 'fill', 'press', 'hover', 'check', 'uncheck', 'select_option']),
  tabId, url: z.string().optional(), role: z.string().optional(), targetName: z.string().optional(),
  selector: z.string().optional(), index: z.number().int().min(0).optional(), text: z.string().optional(),
  observationId: z.string().optional(),
  value: z.string().optional(), key: z.string().optional(), pixels: z.number().int().min(-1600).max(1600).optional(),
  x: z.number().int().min(0).optional(), y: z.number().int().min(0).optional(),
}), args => browser.act(args));
tool('list_email_profiles', 'List up to three local email nicknames configured in Diagonal settings. Does not access passwords.', z.object({}), () => browser.listEmailProfiles());
tool('use_email_profile', 'Fill a visible email or username field with a configured nickname on the current page. Does not submit or authenticate.', z.object({ nickname: z.string().min(1), tabId }), ({ nickname, tabId }) => browser.useEmailProfile(nickname, tabId));
tool('begin_login', 'Open a visible Chrome login window for OAuth or passkey sign-in. The user completes website and OS prompts; Kira can observe and interact with its tabs.', z.object({ url: z.string().url(), tabId }), ({ url, tabId }) => browser.beginLogin(url, tabId));
tool('finish_login', 'Import the completed visible login session into Kira and save its cookies/local storage, or cancel without saving.', z.object({ save: z.boolean().default(true) }), ({ save }) => browser.finishLogin(save));
tool('inspect_page', 'Return structured page metadata, headings, links, images, forms, readable text, and URL details for an LLM.', z.object({ tabId }), ({ tabId }) => browser.inspect(tabId));
tool('scroll', 'Scroll one viewport-sized step (positive down, negative up) and return newly visible text and position.', z.object({ pixels: z.number().int().min(-1600).max(1600).default(600), tabId }), ({ pixels, tabId }) => browser.scroll(pixels, tabId));
tool('wait_for', 'Wait for a CSS selector to become visible in a tab.', z.object({ selector: z.string().min(1), tabId }), ({ selector, tabId }) => browser.waitFor(selector, tabId));
tool('click', 'Click by CSS selector, snapshot index, or exact visible text in a tab.', z.object({ selector: z.string().min(1).optional(), index: z.number().int().min(0).optional(), text: z.string().min(1).optional(), tabId }), ({ selector, index, text, tabId }) => browser.click(selector, index, text, tabId));
tool('fill', 'Fill an input by CSS selector or snapshot index in a tab.', z.object({ selector: z.string().min(1).optional(), index: z.number().int().min(0).optional(), value: z.string(), tabId }), ({ selector, index, value, tabId }) => browser.fill(selector, index, value, tabId));
tool('press', 'Press a keyboard key such as Enter or Tab in a tab.', z.object({ key: z.string().min(1), tabId }), ({ key, tabId }) => browser.press(key, tabId));
tool('screenshot', 'Save a screenshot to Diagonal Browser artifacts; returns its absolute path.', z.object({ tabId, name: z.string().optional(), fullPage: z.boolean().default(true) }), ({ tabId, name, fullPage }) => browser.screenshot(tabId, name, fullPage));
tool('start_recording', 'Start a browser video recording. Continue actions using the returned recording tabId.', z.object({ tabId }), ({ tabId }) => browser.recordStart(tabId));
tool('stop_recording', 'Finish the active recording and save a WebM video under Diagonal Browser artifacts.', z.object({ name: z.string().optional() }), ({ name }) => browser.recordStop(name));
tool('close_tab', 'Close a tab after any active recording has stopped.', z.object({ tabId }), ({ tabId }) => browser.close(tabId));

const transport = new StdioServerTransport();
await server.connect(transport);
process.on('SIGINT', async () => { await browser.shutdown(); process.exit(0); });
process.on('SIGTERM', async () => { await browser.shutdown(); process.exit(0); });
