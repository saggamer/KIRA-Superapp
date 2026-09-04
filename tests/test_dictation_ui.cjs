const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '..', 'ui.html'), 'utf8');
const source = html.slice(html.indexOf('        let dictationActive ='), html.indexOf('        async function configureRealtimeVoice('));
function setup() {
    let finish, cancelled = 0;
    const errors = [];
    const attrs = {};
    const preview = {textContent: '', classList: {add() {}, remove() {}}};
    const input = {value: 'Existing draft', dispatchEvent() {}};
    const context = vm.createContext({
        isBackendReady: true, currentChatId: 'a', userInput: input, Event: class {},
        document: {getElementById: id => id === 'dictation-preview' ? preview : ({setAttribute(k, v) {attrs[k] = v;}, classList: {add() {}, remove() {}}})},
        focusComposer() {}, renderErrorMessage(e) {errors.push(e);},
        window: {pywebview: {api: {
            dictate_once: () => new Promise(resolve => {finish = resolve;}),
            cancel_dictation: async () => {cancelled++;},
        }}},
    });
    vm.runInContext(source, context);
    return {context, input, errors, attrs, preview, finish: result => finish(result), cancelled: () => cancelled};
}
(async () => {
    const normal = setup();
    const turn = normal.context.toggleDictation();
    normal.context.renderDictationPartial({request_id: 1, content: 'live words'});
    assert.equal(normal.preview.textContent, 'live words');
    assert.equal(normal.input.value, 'Existing draft');
    normal.context.renderDictationPartial({request_id: 0, content: 'stale words'});
    assert.equal(normal.preview.textContent, 'live words');
    normal.input.value += ' typed while listening';
    normal.finish({ok: true, text: 'dictated text'});
    await turn;
    assert.equal(normal.input.value, 'Existing draft typed while listening dictated text');
    assert.equal(normal.attrs['aria-pressed'], 'false');
    assert.equal(normal.errors.length, 0);
    const cancel = setup();
    const pending = cancel.context.toggleDictation();
    await cancel.context.toggleDictation();
    cancel.context.renderDictationPartial({request_id: 1, content: 'cancelled partial'});
    assert.notEqual(cancel.preview.textContent, 'cancelled partial');
    cancel.finish({ok: true, text: 'late result'});
    await pending;
    assert.equal(cancel.cancelled(), 1);
    assert.equal(cancel.input.value, 'Existing draft');
    const switched = setup();
    const oldChat = switched.context.toggleDictation();
    switched.context.currentChatId = 'b';
    switched.finish({ok: true, text: 'wrong chat'});
    await oldChat;
    assert.equal(switched.input.value, 'Existing draft');
    const missing = setup();
    const failed = missing.context.toggleDictation();
    missing.finish({ok: false, error: 'Missing model'});
    await failed;
    assert.deepEqual(missing.errors, ['Missing model']);
    assert.equal(missing.attrs['aria-pressed'], 'false');
    console.log('PASS: dictation inserts drafts, cancels, isolates chats, reports failures; no send/TTS API used');
})().catch(error => {console.error(error); process.exitCode = 1;});
