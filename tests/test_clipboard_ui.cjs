const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'ui.html'), 'utf8');
for (const script of html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)) new Function(script[1]);
const helpers = html.slice(html.indexOf('        function enqueueAttachmentImport('), html.indexOf('        function focusComposer('));
const listeners = html.slice(html.indexOf('        function isChatPasteTarget('), html.indexOf("        voiceLiveInput.addEventListener('input'"));

// Exercise the production picker synchronously: awaiting the bridge loses the user gesture.
const pickerSource = html.slice(html.indexOf('        function focusComposer('), html.indexOf('        async function sendMessage('));
let focused = 0, clicked = 0, menuFocused = 0, hidden = true, expanded;
const composer = {focus() { focused++; }};
const doc = {
    activeElement: null,
    querySelector: () => null,
    getElementById: id => id === 'attachment-menu' ? {
        classList: {contains: () => hidden, toggle: (_, value) => { hidden = value; }},
        querySelector: () => ({focus() { menuFocused++; }}),
    } : {setAttribute: (_, value) => { expanded = value; }},
};
const picker = vm.createContext({document: doc, userInput: composer, attachmentInput: {click() { clicked++; }}});
vm.runInContext(pickerSource, picker);
picker.toggleAttachmentMenu();
assert.equal(hidden, false);
assert.equal(expanded, 'true');
assert.equal(menuFocused, 1);
assert.equal(picker.openAttachmentPicker(), undefined);
assert.equal(clicked, 1);
assert.equal(hidden, true);
assert.equal(expanded, 'false');
picker.focusComposer();
assert.equal(focused, 1);
doc.activeElement = {matches: () => true};
picker.focusComposer();
assert.equal(focused, 1, 'Must not steal focus from a settings input');
doc.activeElement = null;
doc.querySelector = () => ({});
picker.focusComposer();
assert.equal(focused, 1, 'Must not steal focus from an open modal');

function fixture(nativeFiles = []) {
    const handlers = {}, imported = [], errors = [], timers = [];
    let nativeCalls = 0;
    const input = {value: '', selectionStart: 0, selectionEnd: 0,
        setSelectionRange() {}, dispatchEvent() {}, closest() { return this; }};
    const context = vm.createContext({
        Promise, Event: class {}, Uint8Array, currentChatId: 'chat-1', isBackendReady: true,
        attachmentImports: Promise.resolve(), attachmentPasteSequence: 0,
        userInput: input, attachmentInput: {value: ''},
        composerWrap: {contains: node => node === input}, chatContainer: {contains: () => false},
        document: {addEventListener: (type, fn) => { handlers[type] = fn; }},
        setTimeout: fn => timers.push(fn), updateSystemStatus() {}, focusComposer() {},
        renderErrorMessage: text => errors.push(text),
        addImportedAttachments: files => imported.push(...files),
        bytesToBase64: data => Buffer.from(data).toString('base64'),
        window: {pywebview: {api: {
            paste_attachments: async () => { nativeCalls++; return {ok: true, files: nativeFiles}; },
            import_attachment: async name => ({ok: true, name, path: '/attachments/' + name}),
        }}},
    });
    vm.runInContext(helpers + listeners, context);
    return {context, input, imported, errors, timers, handlers, calls: () => nativeCalls,
        async paste(text = '', files = []) {
            let prevented = false;
            handlers.paste({target: input, clipboardData: {files, items: [], getData: () => text},
                preventDefault() { prevented = true; }});
            if (!prevented) input.value = text;
            await context.attachmentImports;
            return prevented;
        },
    };
}

(async () => {
    const browser = fixture();
    browser.handlers.keydown({target: browser.input, metaKey: true, key: 'v'});
    assert.equal(await browser.paste('', [{name: 'report.pdf', size: 3, arrayBuffer: async () => Buffer.from('pdf')}]), true);
    browser.timers.forEach(fn => fn());
    assert.equal(browser.calls(), 0);
    assert.equal(browser.imported[0].name, 'report.pdf');

    const plain = fixture();
    assert.equal(await plain.paste('ordinary text'), false);
    assert.equal(plain.input.value, 'ordinary text');
    assert.equal(plain.imported.length, 0);

    const finder = fixture([{ok: true, path: '/attachments/report.docx', name: 'report.docx'}]);
    await finder.paste('report.docx');
    assert.equal(finder.imported.length, 1);
    assert.equal(finder.input.value, '');

    const fallback = fixture([{ok: true, path: '/attachments/report.pdf'}]);
    fallback.handlers.keydown({target: fallback.input, metaKey: true, key: 'v'});
    fallback.timers.forEach(fn => fn());
    await fallback.context.attachmentImports;
    assert.equal(fallback.imported.length, 1);

    const switched = fixture([{ok: true, path: '/attachments/old-chat.pdf'}]);
    const pending = switched.paste();
    switched.context.currentChatId = 'chat-2';
    await pending;
    assert.equal(switched.imported.length, 0);
    for (const run of [browser, plain, finder, fallback, switched]) assert.equal(run.errors.length, 0);
    console.log('PASS: browser files, plain text, Finder filenames, native Cmd+V fallback, chat isolation');
})().catch(error => { console.error(error); process.exitCode = 1; });
