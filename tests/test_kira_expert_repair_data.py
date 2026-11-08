import json
from pathlib import Path
import unittest


class ExpertRepairDataTests(unittest.TestCase):
    def test_balanced_permissions_and_disjoint_inputs(self):
        root = Path(__file__).parents[1] / 'training_data/kira_live_100m_expert_repair_v2'
        splits = {s:[json.loads(line) for line in (root / f'{s}.jsonl').read_text().splitlines()] for s in ('train','valid')}
        for rows in splits.values():
            for source in ('kira_agentic','kira_workflow'):
                labels = [r['permission_target'] for r in rows if r['source']==source]
                self.assertGreater(labels.count(True),0)
                self.assertEqual(labels.count(True),labels.count(False))
            self.assertTrue(all(r['messages'][-1]['role']=='assistant' for r in rows))
        prompts = {s:{json.dumps(r['messages'][:-1],sort_keys=True) for r in rows} for s,rows in splits.items()}
        self.assertFalse(prompts['train'] & prompts['valid'])

    def test_known_failure_probe_not_used_as_a_training_target(self):
        root = Path(__file__).parents[1] / 'training_data/kira_live_100m_expert_repair_v2'
        rows = [json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
        prompts = [m['content'].lower() for r in rows for m in r['messages'] if m['role']=='user']
        self.assertFalse(any('17 times 6' in p or '17 multiplied by 6' in p for p in prompts))

    def test_current_curriculum_matches_native_policy_and_balances_permissions(self):
        from scripts.prepare_kira_100m_expert_repair import build_rows, native_policy
        for split in ('train','valid'):
            # Historical curricula are immutable snapshots. Verify that new
            # curricula use today's policy without rewriting previous training.
            rows = build_rows(split)
            for source in ('kira_agentic','kira_workflow'):
                labels = [r['permission_target'] for r in rows if r['source']==source]
                self.assertEqual(labels.count(True),labels.count(False))
            for row in rows:
                if row['source'] in {'tree','truth','reasoning','conversation'}:
                    self.assertEqual(row['messages'][0]['content'],native_policy(row['source']=='tree'))


if __name__ == '__main__':
    unittest.main()
