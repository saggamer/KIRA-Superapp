#!/usr/bin/env python3
"""Paired local skill ablation: same weights, isolated browser/history/MCP fixtures."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from http.server import ThreadingHTTPServer
from threading import Thread
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_orchestral_everyday_browser import BenchBrain, Fixture, TASKS, score
from diagonal_browser.search_history import save_history
from orchestrator_v15.contracts import parse_block


def score_history(answer, actions):
    """Require real cached lookup/read, with equivalent natural date formats."""
    parsed = [parse_block(item['action']) for item in actions]
    history_actions = [p[1].get('ACTION') for p in parsed if p and p[0] == 'BROWSER_TOOL']
    date_matches = re.search(r'\b(?:21(?:st)?\s+November|November\s+21(?:st)?)\b', answer, re.I)
    read_results = [item['result'] for item in actions
                    if (parse_block(item['action']) or ('', {}))[1].get('ACTION') == 'read_history']
    return bool(date_matches and 'search_history' in history_actions and 'read_history' in history_actions
                and any('"cached":true' in result.replace(' ', '') and '21 November' in result
                        for result in read_results)
                and not set(history_actions).intersection({'navigate', 'search', 'revisit_history'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=['form', 'scroll', 'compare', 'history', 'mcp'],
                        default=['form', 'scroll', 'compare', 'history', 'mcp'])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    server = ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
    Thread(target=server.serve_forever, daemon=True).start()
    templates = dict(TASKS)
    original = os.environ.get('KIRA_V15_SKILLS')
    with tempfile.TemporaryDirectory(prefix='kira-skills-eval-') as directory:
        root = Path(directory)
        os.environ['DIAGONAL_PROFILE_DIR'] = str(root / 'profile')
        os.environ['DIAGONAL_ARTIFACTS_DIR'] = str(root / 'artifacts')
        save_history(kind='page', url='https://example.test/cedar', title='Cedar launch archive',
                     text='Project Cedar launches on 21 November. This is an archived planning note.')
        config = root / 'mcp.json'
        config.write_text(json.dumps({'mcpServers': {'test-notes': {
            'command': sys.executable, 'args': [str(ROOT / 'tests/fixtures/mcp_notes_server.py')],
            'env': {},
        }}}))
        brain = BenchBrain()
        brain._load_chat_messages = lambda _chat_id: []
        brain._smart_memory_context = lambda *_a, **_k: ''
        brain.mcp_config_targets = {'kira_app': str(config)}
        base = 'http://127.0.0.1:' + str(server.server_address[1])
        try:
            for task in args.tasks:
                request = (templates[task].format(base=base) if task in templates else
                           'Use Diagonal Browser history to find and read the saved Cedar launch page. What is its launch date? Do not open the live web.' if task == 'history' else
                           'Use the MCP test-notes.search_notes tool to find my charger reminder, then tell me what the note says.')
                for enabled in (False, True):
                    os.environ['KIRA_V15_SKILLS'] = 'on' if enabled else 'off'
                    brain.benchmark_actions, brain.benchmark_proposals = [], []
                    start = monotonic()
                    error = None
                    try:
                        answer = brain._run_v15_profile(request, 'skills-isolated', 'diagonal',
                                                        orchestral_candidate=True)
                    except Exception as exc:
                        answer, error = '', f'{type(exc).__name__}: {exc}'
                    actions = brain.benchmark_actions
                    if task in templates:
                        passed = not error and score(task, answer, actions, 0, 0)
                    elif task == 'history':
                        passed = not error and score_history(answer, actions)
                    else:
                        passed = not error and all(word in answer.lower() for word in ('charger', 'passport'))
                    row = {'task': task, 'skills_enabled': enabled, 'passed': bool(passed),
                           'seconds': round(monotonic()-start, 2), 'answer': answer, 'error': error,
                           'skill_audit': getattr(brain, 'last_v15_skill_audit', {}),
                           'actions': actions, 'proposals': brain.benchmark_proposals}
                    results.append(row)
                    args.output.write_text(json.dumps({
                        'scope': 'Paired local development fixtures, same model weights; not a public benchmark or frontier comparison.',
                        'results': results,
                    }, indent=2) + '\n')
                    print(json.dumps({k: row[k] for k in ('task', 'skills_enabled', 'passed', 'seconds', 'error')}), flush=True)
        finally:
            brain.is_running = False
            brain._stop_model_worker()
            server.shutdown()
            if original is None:os.environ.pop('KIRA_V15_SKILLS', None)
            else:os.environ['KIRA_V15_SKILLS'] = original


if __name__ == '__main__':
    main()
