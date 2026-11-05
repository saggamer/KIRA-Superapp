import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { existsSync } from 'node:fs';
import path from 'node:path';
import os from 'node:os';

// Uses the same configured model server or isolated local worker as Kira Superapp.
export class KiraModel {
  constructor(root) { this.root = root; this.pending = new Map(); this.counter = 0; }
  label() { return process.env.KIRA_MODEL_NAME || 'Kira · local Orchestrator'; }
  async complete(messages, signal) {
    if (process.env.KIRA_MODEL_BASE_URL) {
      const url = new URL(process.env.KIRA_MODEL_BASE_URL);
      if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || (url.protocol === 'http:' && !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname))) throw new Error('Use HTTPS for remote model servers or HTTP on localhost.');
      if (!process.env.KIRA_MODEL_NAME) throw new Error('Set KIRA_MODEL_NAME for the connected model server.');
      const headers = { 'Content-Type': 'application/json' };
      if (process.env.KIRA_MODEL_API_KEY) headers.Authorization = `Bearer ${process.env.KIRA_MODEL_API_KEY}`;
      const response = await fetch(`${url.href.replace(/\/$/, '')}/chat/completions`, { method: 'POST', headers, redirect: 'error', signal: AbortSignal.any([signal, AbortSignal.timeout(90000)]), body: JSON.stringify({ model: process.env.KIRA_MODEL_NAME, messages, temperature: 0.2, max_tokens: 900, stream: false }) });
      if (!response.ok) throw new Error(`Kira model server returned HTTP ${response.status}. Check its connection.`);
      const result = await response.json();
      const content = result.choices?.[0]?.message?.content;
      if (typeof content !== 'string') throw new Error('Kira model server returned no response');
      return content;
    }
    if (!this.child) {
      const candidates = [process.env.KIRA_ORCHESTRATOR_PATH, ...[this.root, path.join(this.root, 'models'), path.join(os.homedir(), 'Downloads')].flatMap(root => ['orchestrator_v1_fused', 'orchestrator v1 fused', 'Orchestrator V1 fused', 'Orchestrator_V1'].map(name => path.join(root, name)))].filter(Boolean);
      const modelPath = candidates.find(candidate => existsSync(path.join(candidate, 'config.json')) && existsSync(path.join(candidate, 'tokenizer.json')));
      if (!modelPath) throw new Error('Connect a Kira model: set KIRA_ORCHESTRATOR_PATH to your downloaded fused model, or KIRA_MODEL_BASE_URL and KIRA_MODEL_NAME to your model server. Relaunch Diagonal after configuration.');
      const python = process.env.DIAGONAL_KIRA_PYTHON || [path.join(this.root, '.venv-orchestrator-v15/bin/python'), path.join(this.root, '.venv/bin/python')].find(existsSync) || 'python3';
      this.child = spawn(python, ['-u', path.join(this.root, 'model_worker.py')], { cwd: this.root, stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, PYTHONUNBUFFERED: '1' } });
      this.child.stderr.on('data', () => {});
      createInterface({ input: this.child.stdout }).on('line', line => {
        let data; try { data = JSON.parse(line); } catch { return; }
        const pending = this.pending.get(String(data.id)); if (!pending) return;
        this.pending.delete(String(data.id)); data.ok ? pending.resolve(data) : pending.reject(new Error(data.error || 'Local Kira model failed'));
      });
      const child = this.child;
      child.on('error', () => { if (this.child === child) this.dispose(); }); child.on('exit', () => { if (this.child === child) this.dispose(); });
      this.loading = this.request({ type: 'load', brain: 'orchestrator', model_path: modelPath }, signal, 180000);
    }
    try {
      await this.loading;
      if (signal.aborted) throw new Error('Stopped');
      const prompt = '<bos>' + messages.map(message => `<|turn>${message.role === 'assistant' ? 'model' : 'user'}\n${message.content}<turn|>\n`).join('') + '<|turn>model\n';
      return (await this.request({ type: 'generate', prompt, max_tokens: 900, temperature: 0.2 }, signal)).content;
    } catch (error) { this.dispose(); throw error; }
  }
  request(value, signal, timeout = 90000) {
    return new Promise((resolve, reject) => {
      const id = String(++this.counter);
      const finish = (callback, result) => { clearTimeout(timer); signal.removeEventListener('abort', abort); this.pending.delete(id); callback(result); };
      const abort = () => { finish(reject, new Error('Stopped')); this.dispose(); };
      const timer = setTimeout(() => { finish(reject, new Error('Kira model timed out')); this.dispose(); }, timeout);
      this.pending.set(id, { resolve: result => finish(resolve, result), reject: error => finish(reject, error) });
      signal.addEventListener('abort', abort, { once: true });
      if (signal.aborted) return abort();
      this.child.stdin.write(JSON.stringify({ ...value, id }) + '\n', error => { if (error) finish(reject, new Error('Kira worker disconnected')); });
    });
  }
  dispose() {
    const child = this.child; this.child = null; this.loading = null;
    for (const pending of [...this.pending.values()]) pending.reject(new Error('Kira model worker stopped'));
    this.pending.clear(); if (child && !child.killed) child.kill();
  }
}
