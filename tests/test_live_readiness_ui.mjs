import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../ui.html', import.meta.url), 'utf8');
const code = html.slice(html.indexOf('        function renderKiraLiveReadiness('),
                        html.indexOf('        async function startKiraLiveConversation('));
const element = () => ({append() {}, appendChild() {}, classList: {toggle() {}}, textContent: ''});
const statuses = [{state: 'starting', conversation_ready: false, checks: [], blockers: []},
                  {state: 'ready', conversation_ready: true, checks: [], blockers: []}];
const context = vm.createContext({
    kiraLiveStatus: {}, kiraLiveReadinessTimer: null, kiraLiveStartEpoch: 1,
    isBackendReady: true, liveMonitorOpen: true, isVoiceActive: false,
    liveReadiness: element(), voiceListenButton: {disabled: true}, voiceLiveSubtitle: element(),
    document: {createElement: element, getElementById: element},
    window: {pywebview: {api: {get_kira_live_status: async () => statuses.shift()}}},
    setTimeout(fn) { context.pendingRetry = fn; return 1; },
    clearTimeout() { context.pendingRetry = null; },
    setVoiceLiveState(state, caption) { context.display = {state, caption}; },
});
vm.runInContext(code, context);
await vm.runInContext('refreshKiraLiveStatus()', context);
assert.equal(context.display.state, 'loading');
assert.equal(context.voiceListenButton.disabled, true);
assert.equal(typeof context.pendingRetry, 'function');
await context.pendingRetry();
assert.equal(context.display.state, 'ready');
assert.equal(context.voiceListenButton.disabled, false);
assert.equal(context.pendingRetry, null);

statuses.push({state: 'building', conversation_ready: false,
    checks: [{label: 'Listener', ready: false, detail: 'Checkpoint is missing'}], blockers: ['Listener']});
await vm.runInContext('refreshKiraLiveStatus()', context);
assert.equal(context.display.state, 'error');
assert.match(context.display.caption, /Listener: Checkpoint is missing/);
assert.equal(context.pendingRetry, null);
assert.equal(context.voiceListenButton.disabled, true);

// A late response cannot reopen or alter a closed/replaced Live surface.
let resolveStatus;
context.window.pywebview.api.get_kira_live_status = () => new Promise(resolve => {resolveStatus = resolve;});
const pending = vm.runInContext('refreshKiraLiveStatus()', context);
context.liveMonitorOpen = false;
context.kiraLiveStartEpoch++;
resolveStatus({state: 'ready', conversation_ready: true});
await pending;
assert.equal(context.display.state, 'error');
assert.equal(context.pendingRetry, null);
console.log('Live readiness UI: startup recovery, blocker details and stale-response checks passed.');
