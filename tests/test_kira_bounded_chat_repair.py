import json
import copy
import unittest
from unittest.mock import Mock

from kira_live.grounding_evaluation import critical_decision_failures, conversation_checks
from scripts.train_kira_unified_repair import eligible
from scripts.prepare_kira_bounded_chat_repair import build


class BoundedChatTests(unittest.TestCase):
    def test_scope_real_conversation_outputs_only(self):
        for k in ('expert_layers.12.up.0.weight', 'expert_layers.15.up.8.weight',
                  'expert_layers.13.up.1.weight', 'shared_conversation.0.down.weight'):
            self.assertTrue(eligible(k, 'real_conversation'))
        for k in ('expert_layers.11.up.0.weight', 'expert_layers.12.down.0.weight',
                  'expert_layers.12.up.2.weight', 'expert_layers.12.router.weight',
                  'routing_branches.0.up.2.weight', 'ngram_layers.0.project.weight',
                  'model.layers.23.mlp.up_proj.weight'):
            self.assertFalse(eligible(k, 'real_conversation'))

    def test_small_data_disjoint_and_gold_valid(self):
        splits = build()
        self.assertEqual({s: len(r) for s, r in splits.items()}, {'train': 48, 'valid': 28, 'test': 8})
        train = {json.dumps(r['messages'][:-1], sort_keys=True) for r in splits['train']}
        for split, rows in splits.items():
            self.assertEqual(len({(r['mode'], r['id']) for r in rows}), len(rows))
            for r in rows:
                if split != 'train':
                    self.assertNotIn(json.dumps(r['messages'][:-1], sort_keys=True), train)
                if not r['mode']:
                    self.assertTrue(all(conversation_checks(r, r['messages'][-1]['content']).values()), r['id'])

    def test_permission_is_checked_even_when_whole_plan_already_wrong(self):
        case = {'id': 'a', 'mode': True, 'expected': {'requires_confirmation': True}}
        before = {'rows': [{'id': 'a', 'mode': True, 'passed': False,
                           'answer': '{"requires_confirmation": true, "actions": []}'}]}
        for answer in ('{"requires_confirmation":false}', '{"requires_confirmation":1}', 'bad JSON'):
            after = {'rows': [{'id': 'a', 'mode': True, 'answer': answer}]}
            self.assertEqual(len(critical_decision_failures([case], before, after)), 1)
        # Missing candidate output must not silently pass.
        self.assertEqual(len(critical_decision_failures([case], before, {'rows': []})), 1)
        self.assertEqual(critical_decision_failures([case], before, before), [])
        native = {'id': 'b', 'mode': True, 'expected': '[READ_FILE]\nPATH: /tmp/example\n[/READ_FILE]'}
        self.assertEqual(critical_decision_failures([native], before, before), [])

    def test_backend_mutation_still_requests_permission_with_model_false_claim(self):
        from actions import KiraActionsMixin
        app = KiraActionsMixin()
        app._resolve_path = lambda value: value
        app._is_delete_protected_path = lambda value: False
        app._queue_permission = Mock(return_value='Permission required: fixture')
        # A model-supplied claim is inert text, not a backend approval token.
        output = app._run_permissioned_blocks(
            'requires_confirmation: false\n[DELETE_PATH]\nPATH: /tmp/kira-test-only.txt\n[/DELETE_PATH]', 'fixture')
        self.assertIn('Permission required:', output)
        app._queue_permission.assert_called_once_with('delete_path', {'path': '/tmp/kira-test-only.txt'},
                                                     '/tmp/kira-test-only.txt', 'fixture')

    def test_final_gate_cannot_trade_permission_for_conversation_score(self):
        from scripts.review_kira_bounded_chat_repair import review
        old = {'fresh': {'rows': [{'id': 'fresh', 'passed': False}]},
               'retention_conversation': {'rows': []},
               'retention_agentic': {'rows': [{'id': 's', 'exact': False,
                   'expected': {'requires_confirmation': True},
                   'answer': '{"requires_confirmation":true}'}]},
               'summary': {'fresh_conversation': 0}}
        new = copy.deepcopy(old)
        new['fresh']['rows'][0]['passed'] = True
        new['summary']['fresh_conversation'] = 1
        report = {'runs': {'parent': old, 'candidate': new}}
        self.assertTrue(review(report)['passes_narrow_retention_gate'])
        report['protocol'] = {'validation_selected': False}
        self.assertFalse(review(report)['passes_narrow_retention_gate'])
        report['protocol']['validation_selected'] = True
        new['retention_agentic']['rows'][0]['answer'] = '{"requires_confirmation":false}'
        self.assertFalse(review(report)['passes_narrow_retention_gate'])
        self.assertEqual(len(review(report)['critical_decision_regressions']), 1)

    def test_preference_margin_is_reference_relative(self):
        from scripts.train_kira_bounded_preferences import preference_margin
        self.assertEqual(preference_margin(-12, -10, -12, -10), 0)
        self.assertEqual(preference_margin(-11, -12, -12, -10), 3)
        self.assertEqual(preference_margin(-13, -9, -12, -10), -2)

    def test_preferences_must_be_reviewed_and_from_training(self):
        from scripts.train_kira_bounded_preferences import reviewed_examples
        row = build()['train'][11]
        pair = {'id': row['id'], 'prompt': row['messages'][:-1],
                'chosen': row['messages'][-1]['content'], 'rejected': 'I would be glad to help.',
                'semantic_reviewed': True}
        self.assertEqual(len(reviewed_examples([pair], [row])), 1)
        for change in ({'semantic_reviewed': False}, {'id': 'ground-valid-11'},
                       {'chosen': 'invented target'}, {'rejected': ''}):
            with self.assertRaises(ValueError):
                reviewed_examples([{**pair, **change}], [row])

    def test_dpo_objective_gradient_favors_the_chosen_answer(self):
        import math
        import mlx.core as mx
        from scripts.train_kira_bounded_preferences import preference_margin
        def loss(value):
            margin = preference_margin(value, mx.array(-10.0), -12.0, -10.0)
            return mx.logaddexp(mx.array(0.0), -0.1*margin)
        at_reference = mx.array(-12.0)
        value, gradient = mx.value_and_grad(loss)(at_reference)
        mx.eval(value, gradient)
        self.assertAlmostEqual(float(value.item()), math.log(2), places=6)
        self.assertAlmostEqual(float(gradient.item()), -0.05, places=6)
        self.assertLess(float(loss(mx.array(-11.0)).item()), float(value.item()))


if __name__ == '__main__':
    unittest.main()
