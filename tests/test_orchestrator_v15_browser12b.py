import hashlib
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from actions import KiraActionsMixin
from orchestrator_v15.checkpoints import research_checkpoint
from orchestrator_v15.contracts import reward
from scripts.prepare_orchestrator_v15_browser12b import browser_cases
from scripts.evaluate_orchestrator_v15_browser12b import selected_cases


ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training_data/orchestrator_v15_browser12b'


class BrowserCorpusTests(unittest.TestCase):
    def test_rejected_training_still_has_diagnostic_checkpoint(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            run=Path(directory)
            (run/'status.json').write_text(json.dumps({'stage':'pilot_rejected'}))
            (run/'sft.safetensors').write_bytes(b'checkpoint')
            self.assertEqual(research_checkpoint(run),(run/'sft.safetensors','pilot_rejected'))
            (run/'status.json').write_text(json.dumps({'stage':'sft'}))
            with self.assertRaises(RuntimeError):research_checkpoint(run)

    def test_sealed_evaluation_selection_covers_browser_and_prior_failures(self):
        if not (DATA/'manifest.json').exists():self.skipTest('corpus not built')
        _,cases=selected_cases()
        self.assertEqual(len(cases),113)
        self.assertEqual(len({case['id'] for case in cases}),len(cases))
        families={case['family'] for case in cases}
        self.assertTrue({'cross_branch','browser_stale_recovery',
                         'browser_permission','original_workflow',
                         'original_app_action'}<=families)

    def test_authored_targets_are_typed_and_bridge_dispatchable(self):
        agent=KiraActionsMixin()
        families=set()
        for row in browser_cases(1):
            families.add(row['family'])
            answer=row['messages'][-1]['content']
            self.assertEqual(reward(answer,row['verifier']),1.0,row['id'])
            if answer.startswith('[BROWSER_TOOL]'):
                with patch('diagonal_browser.kira_client.call_browser_tool',
                           return_value={'observationId':'verified'}) as call:
                    agent._run_web_search_tools(answer)
                self.assertEqual(call.call_count,1,row['id'])
                self.assertEqual(call.call_args.args[0],row['verifier']['fields']['ACTION'])
        self.assertEqual(len(families),16)

    def test_built_corpus_hashes_splits_and_database(self):
        if not (DATA/'manifest.json').exists():self.skipTest('corpus not built')
        manifest=json.loads((DATA/'manifest.json').read_text())
        expected_db_sha=(DATA/'corpus.sqlite.sha256').read_text().split()[0]
        self.assertEqual(hashlib.sha256((DATA/'corpus.sqlite').read_bytes()).hexdigest(),
                         expected_db_sha)
        seen={}
        with sqlite3.connect(DATA/'corpus.sqlite') as con:
            self.assertEqual(con.execute('SELECT COUNT(*) FROM source_catalog').fetchone()[0],7)
            self.assertEqual(con.execute('SELECT license_json FROM source_catalog WHERE source=?',
                                         ('kira_diagonal_browser_authored',)).fetchone()[0],
                             '"project-owned"')
            self.assertEqual(json.loads(con.execute('SELECT license_json FROM source_catalog WHERE source=?',
                                                     ('kira_agentic_sft_v1',)).fetchone()[0]),
                             ['cc-by-4.0','project-owned'])
            for split in ('train','valid','test'):
                path=DATA/(split+'.jsonl')
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 manifest['files'][split]['sha256'])
                with path.open() as handle:
                    rows=[json.loads(line) for line in handle]
                self.assertEqual(len(rows),manifest['files'][split]['rows'])
                self.assertGreater(sum(r.get('family')=='browser_permission' for r in rows),0)
                self.assertEqual(con.execute('SELECT count(*) FROM records WHERE split=?',
                                             (split,)).fetchone()[0],len(rows))
                for row in rows:
                    self.assertLessEqual(len(row['token_ids']),1536)
                    self.assertLess(row['answer_start'],len(row['token_ids']))
                    if row['prompt_sha256'] in seen:
                        self.assertEqual(seen[row['prompt_sha256']],split)
                    seen[row['prompt_sha256']]=split


if __name__=='__main__':unittest.main()
