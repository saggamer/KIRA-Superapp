import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../ui.html', import.meta.url), 'utf8');
const section = (start, end) => html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));
const nodes = new Map();
const element = () => ({innerHTML: '', textContent: '', style: {}, classList: {remove() {}},
    remove() { nodes.delete(this.id); }});
const context = vm.createContext({
    executionProgress: {}, modelSelector: {value: 'orchestrator_v15'},
    document: {getElementById: id => nodes.get(id), createElement: element},
    chatContainer: {appendChild(node) {
        nodes.set(node.id, node);
        for (const id of ['execution-tree', 'execution-progress-label', 'execution-progress-stage',
            'execution-progress-percent', 'execution-progress-fill', 'execution-branches']) nodes.set(id, element());
    }},
    removeWelcome() {}, scrollToBottom() {}, escapeHtml: value => value,
    setVoiceLiveState(state, text) { context.voiceText = text; },
});
vm.runInContext(section('        function modelLabelForSelection(', '        const baseSlashCommands'), context);
vm.runInContext(section('        function showLoadingIndicator(', '        function renderAIMessage('), context);
vm.runInContext(section('        function syncVoiceStateFromStatus(', '        function handleVoiceBackendEvent('), context);

for (const [model, label] of [
    ['orchestrator_v15', 'Orchestrator v1.5'],
    ['orchestrator_v15_thinking', 'Orchestrator v1.5'],
    ['orchestrator_v15_orchestral_preview', 'Orchestrator v1.5'],
    ['orchestrator_v15_ppt_preview', 'Horizon 1'],
    ['kira', 'Kira Live 1'], ['orchestrator', 'Orchestrator v1'],
]) {
    context.modelSelector.value = model;
    vm.runInContext('showLoadingIndicator()', context);
    assert.ok(nodes.get('loading-indicator').innerHTML.includes(`${label} is working...`));
    // Changing the selector cannot relabel the task that is already running.
    context.modelSelector.value = 'orchestrator';
    vm.runInContext('renderExecutionProgress({percent: 35})', context);
    assert.equal(nodes.get('execution-progress-label').textContent, `${label} is working...`);
    assert.equal(nodes.get('execution-progress-percent').textContent, '35%');
    vm.runInContext('syncVoiceStateFromStatus("Thinking")', context);
    assert.equal(context.voiceText, `${label} is thinking...`);
    vm.runInContext('renderExecutionProgress({label: "Reading sources"})', context);
    assert.equal(nodes.get('execution-progress-label').textContent, 'Reading sources');
}
console.log('Model status UI: all six selections, progress updates and pinned task labels passed.');
