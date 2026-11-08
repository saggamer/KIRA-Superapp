import json
import unittest
from scripts.prepare_kira_unified_repair import build
from scripts.train_kira_unified_repair import eligible, answer_weights
from scripts.benchmark_kira_unified_repair import chat_checks
from kira_live.expert_rl import reward


class UnifiedRepairTests(unittest.TestCase):
    def test_disjoint_small_curriculum_and_exact_targets(self):
        splits = build()
        self.assertEqual({s: len(r) for s, r in splits.items()}, {'train': 80, 'valid': 20, 'test': 24})
        seen = set()
        for rows in splits.values():
            for row in rows:
                prompts = tuple(m['content'] for m in row['messages'] if m['role'] == 'user')
                self.assertNotIn(prompts, seen)
                seen.add(prompts)
                self.assertEqual(row['messages'][-1]['role'], 'assistant')
                if row['mode']:
                    self.assertEqual(reward(row, row['messages'][-1]['content']), 1)

    def test_scope_only_agentic_pairs_and_routers_or_chat(self):
        self.assertTrue(eligible('routing_branches.0.up.2.weight', 'agentic'))
        self.assertTrue(eligible('routing_branches.0.down.5.weight', 'agentic'))
        self.assertTrue(eligible('routing_branches.3.router.weight', 'agentic'))
        for k in ('routing_branches.0.up.0.weight', 'routing_branches.0.gate.bias',
                  'expert_layers.0.up.2.weight', 'model.layers.0.mlp.weight', 'shared_conversation.0.up.weight'):
            self.assertFalse(eligible(k, 'agentic'))
        self.assertTrue(eligible('shared_conversation.0.up.weight', 'conversation'))
        self.assertFalse(eligible('routing_branches.0.router.weight', 'conversation'))

    def test_reference_answers_meet_narrow_conversation_constraints(self):
        rows = [r for r in build()['test'] if not r['mode']]
        for i, row in enumerate(rows):
            self.assertTrue(all(chat_checks(i, row['messages'][-1]['content']).values()), i)

    def test_incomplete_or_generic_answer_does_not_pass(self):
        self.assertFalse(all(chat_checks(2, 'Sure, I can send that.').values()))
        self.assertFalse(all(chat_checks(4, 'Rusty.').values()))
        self.assertFalse(all(chat_checks(9, '<think>Flowers are nice.').values()))
        self.assertFalse(all(chat_checks(10, 'Hi Kofi, I will be fifteen minutes late.', truncated=True).values()))

    def test_fp32_master_retains_sub_bf16_step(self):
        import mlx.core as mx
        import mlx.optimizers as optim
        initial = mx.array([1.0], dtype=mx.bfloat16)
        optimizer = optim.SGD(learning_rate=1e-4)
        master = {'w': initial.astype(mx.float32)}
        for _ in range(60):
            master = optimizer.apply_gradients({'w': mx.array([1.0])}, master)
            mx.eval(master)
        self.assertLess(float(master['w'].astype(mx.bfloat16).item()), 1.0)
        direct = initial
        for _ in range(60):
            direct = (direct.astype(mx.float32)-1e-4).astype(mx.bfloat16)
        self.assertEqual(float(direct.item()), 1.0)

    def test_precision_subset_only_reuses_train_not_test_or_validation(self):
        from scripts.prepare_kira_expert_precision import selected_rows
        train, valid = selected_rows()
        self.assertEqual(len(train), 16)
        self.assertEqual(len(valid), 12)
        self.assertTrue(all(r['mode'] and '-train-' in r['id'] for r in train))
        self.assertEqual(len({r['id'] for r in train}), 16)
        self.assertTrue(all('-valid-' in r['id'] for r in valid))

    def test_precision_weights_only_decisions_and_literal_values(self):
        class CharacterTokenizer:
            def decode(self, values, **_kwargs):
                return ''.join(chr(v) for v in values)
        import mlx.core as mx
        text = '{"recipient": "Zoya", "requires_confirmation": true, "reason": "user_requested_action"}'
        weights = answer_weights(CharacterTokenizer(), mx.array([ord(c) for c in text]), text)
        self.assertEqual(weights[text.index('true')], 8)
        self.assertEqual(weights[text.index('Zoya')], 3)
        self.assertEqual(weights[text.index('recipient')], 1)
        self.assertEqual(weights[text.index('user_requested_action')], 1)


if __name__ == '__main__':
    unittest.main()
