import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../ui.html', import.meta.url), 'utf8');
const code = html.slice(html.indexOf('        function stopBrowserPreview('),
                        html.indexOf('        function imagePathsInMessage('));
const element = () => ({children: [], style: {}, attributes: {}, setAttribute(key, value) {this.attributes[key] = value;}, append(...children) {this.children.push(...children);},
    remove() {this.removed = true;}});
let calls = 0;
const context = vm.createContext({
    browserPreviewCard: null, browserPreviewTimer: null, browserPreviewRunning: false, browserPreviewEpoch: 0,
    currentChatId: 'chat', document: {hidden: false, createElement: element, getElementById: () => null},
    chatContainer: {insertBefore(panel) {context.panel = panel;}},
    window: {pywebview: {api: {get_browser_preview: async chat => {
        assert.equal(chat, 'chat'); calls++;
        return {active: true, available: true, title: 'Fixture', url: 'https://example.test',
            dataUrl: 'data:image/jpeg;base64,AAAA'};
    }}}},
    setTimeout(fn) {context.tick = fn; return 1;}, clearTimeout() {context.tick = null;},
});
vm.runInContext(code, context);
context.startBrowserPreview({active: true, chat_id: 'chat', action: 'navigate'});
await new Promise(resolve => setImmediate(resolve));
assert.equal(calls, 1);
assert.equal(context.browserPreviewCard.image.src, 'data:image/jpeg;base64,AAAA');
assert.equal(context.browserPreviewCard.url.textContent, 'https://example.test');
const card = context.browserPreviewCard;
context.startBrowserPreview({active: true, chat_id: 'chat', action: 'observe'});
assert.equal(context.browserPreviewCard, card);
// Hidden and collapsed previews must not capture frames or load the browser.
context.document.hidden = true;
await context.tick();
assert.equal(calls, 1);
context.document.hidden = false;
card.panel.open = false;
await context.tick();
assert.equal(calls, 1);
card.panel.open = true;
let resolveFrame;
context.window.pywebview.api.get_browser_preview = () => new Promise(resolve => {resolveFrame = resolve;});
const pending = context.refreshBrowserPreview(context.browserPreviewEpoch);
context.stopBrowserPreview();
resolveFrame({active: true, available: true, title: 'Stale frame', dataUrl: 'data:image/jpeg;base64,BBBB'});
await pending;
assert.equal(card.title.textContent, 'Fixture');
assert.equal(context.tick, null);
assert.equal(card.hint.textContent, 'Browser preview paused');
assert.equal(card.state.textContent, 'Paused');
assert.equal(card.panel.attributes['data-state'], 'paused');
context.stopBrowserPreview(true);
assert.equal(card.panel.removed, true);
console.log('Browser preview UI: frames, collapsed/hidden throttling, completion and stale response checks passed.');
