#!/usr/bin/env python3
"""Model-in-loop read-only MCP call against an isolated Notes-like app."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault('webview', types.SimpleNamespace())
from interface import KiraBrain


class BenchBrain(KiraBrain):
    def _scheduled_task_loop(self):
        pass

    def _personalization_loop(self):
        pass

    def _worker_loop(self):
        pass

    def _model_worker_request(self, payload, timeout=60):
        value = super()._model_worker_request(payload, timeout=timeout)
        if payload.get('type') == 'propose_v15':
            self.proposals.extend(value.get('proposals', []))
        return value


def main():
    output = ROOT / 'training_runs/orchestrator_v15/mcp_live_notes_20261015.json'
    with tempfile.TemporaryDirectory(prefix='kira-mcp-live-') as directory:
        config = Path(directory) / 'kira_mcp_app.json'
        fixture = ROOT / 'tests/fixtures/mcp_notes_server.py'
        config.write_text(json.dumps({'mcpServers': {'test-notes': {
            'command': sys.executable, 'args': [str(fixture)], 'env': {},
        }}}))
        brain = BenchBrain()
        brain.proposals = []
        brain.mcp_config_targets = {'kira_app': str(config),
                                    'kira_drafts': str(Path(directory) / 'drafts')}
        request = ('Use the MCP test-notes.search_notes tool to find my charger '
                   'reminder, then tell me what the note says.')
        started = monotonic()
        try:
            answer = brain._run_v15_profile(request, 'mcp-live-notes-test', 'diagonal')
            error = None
        except Exception as exc:
            answer = ''
            error = type(exc).__name__ + ': ' + str(exc)
        finally:
            brain.is_running = False
            brain._stop_model_worker()
        row = {'scope': 'isolated Notes-like MCP app; real stdio SDK transport, no user data',
               'passed': not error and 'charger' in answer.casefold() and
                         'passport' in answer.casefold(),
               'seconds': round(monotonic()-started, 2), 'answer': answer,
               'error': error, 'proposals': brain.proposals}
        output.write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps({key: value for key, value in row.items() if key != 'proposals'}), flush=True)


if __name__ == '__main__':
    main()
