import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SpecialistDataTests(unittest.TestCase):
    def test_b_training_preserves_frozen_gate_inference_mode(self):
        import mlx.nn as nn
        from scripts.train_kira_specialist import train_mode
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.variant = 'shared_agentic'
                self.decoder = nn.Linear(2, 2)
                self.routing_branches = [nn.Linear(2, 2)]
        model = Model()
        train_mode(model)
        self.assertTrue(model.decoder.training)
        self.assertFalse(model.routing_branches[0].training)

    def test_compact_splits_disjoint_and_complete(self):
        from scripts.prepare_kira_specialists import signature
        for label in ('A', 'B'):
            seen = set()
            for split in ('train', 'valid', 'test'):
                path = ROOT / f'training_data/kira_specialists_v1/{label}/{split}.jsonl'
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    self.assertEqual(row['messages'][-1]['role'], 'assistant')
                    self.assertTrue(all(isinstance(m['content'], str) and m['content'] for m in row['messages']))
                inputs = {signature(row['messages']) for row in rows}
                self.assertEqual(len(inputs), len(rows))
                self.assertFalse(inputs & seen)
                seen |= inputs

    def test_permissions_balanced_per_split(self):
        for split in ('train', 'valid', 'test'):
            rows = [json.loads(line) for line in (ROOT / f'training_data/kira_specialists_v1/A/{split}.jsonl').read_text().splitlines()]
            for source, field in [('kira_agentic', 'requires_confirmation'), ('kira_workflow', 'must_request_permission')]:
                decisions = [json.loads(r['messages'][-1]['content'])[field] for r in rows if r['source'] == source]
                self.assertEqual(decisions.count(True), decisions.count(False))

    def test_emotion_rubric_rejects_advice_and_offer_without_fulfillment(self):
        from scripts.benchmark_kira_specialists import emotion_score
        row = {'category': 'listen'}
        self.assertEqual(emotion_score(row, 'I hear you. Try breathing deeply.')['task_pass'], 0)
        self.assertEqual(emotion_score(row, 'That sounds draining. I am listening.')['task_pass'], 1)
        row = {'category': 'fulfillment'}
        self.assertEqual(emotion_score(row, 'I can help you draft that.')['task_pass'], 0)
        self.assertEqual(emotion_score(row, 'I am sorry I snapped at you. I value our friendship.')['task_pass'], 1)

    def test_rubric_accepts_valid_paraphrase_and_short_celebration(self):
        from scripts.benchmark_kira_specialists import emotion_score
        self.assertEqual(emotion_score({'category': 'celebrate'}, 'Congratulations!')['task_pass'], 1)
        row = {'category': 'continuity', 'event': 'chess final', 'worry': 'running out of time'}
        self.assertEqual(emotion_score(row, 'It is your chess final, and time is tight; I hear that worry.')['task_pass'], 1)
        self.assertEqual(emotion_score(row, 'Your chess final is important; I hear that worry.')['task_pass'], 0)

    def test_permission_weight_and_workflow_route_labels(self):
        import mlx.core as mx
        from scripts.train_kira_specialist import token_supervision
        class Tokenizer:
            def decode(self, tokens, **kwargs):
                return ''.join(chr(c) for c in tokens)
        # Include each boolean as one token to mirror the real tokenizer.
        class Words:
            def decode(self, tokens, **kwargs):
                return ''.join(['{"must_request_permission": ', 'true', ', "latest_unresolved_status": ', '"failed"', '}'][i] for i in tokens)
        text = '{"must_request_permission": true, "latest_unresolved_status": "failed"}'
        row = {'source': 'kira_workflow', 'messages': [{'role': 'user', 'content': 'Continue safely'}, {'role': 'assistant', 'content': text}]}
        labels, gates, weights = token_supervision(row, Words(), mx.array([[99, 0, 1, 2, 3, 4]]), 1, 'token_routed')
        self.assertEqual(float(weights[0, 1].item()), 8)
        self.assertEqual(int(labels[0, 1].item()), 4)
        self.assertEqual(int(labels[0, 3].item()), 5)


if __name__ == '__main__':
    unittest.main()
