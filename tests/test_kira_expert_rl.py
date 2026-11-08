import json
import unittest
from kira_live.expert_rl import advantages, reward, exact
from scripts.prepare_kira_a_rl import make


class ExpertRLTests(unittest.TestCase):
    def test_rewards_validate_full_targets(self):
        for row in make('train', 'amber', 'Nila', 'lunar craters'):
            answer = row['expected'] if isinstance(row['expected'], str) else json.dumps(row['expected'])
            self.assertAlmostEqual(reward(row, answer), 1)
            self.assertTrue(exact(row, answer))
            self.assertEqual(reward(row, answer, truncated=True), 0)

    def test_permissions_and_booleans_cannot_be_gamed(self):
        case = make('train', 'amber', 'Nila', 'lunar craters')[2]
        for flag in (False, 1, 'true', None):
            value = {**case['expected'], 'requires_confirmation': flag}
            self.assertEqual(reward(case, json.dumps(value)), 0)
            self.assertFalse(exact(case, json.dumps(value)))

    def test_workflow_false_completion_is_zero(self):
        case = make('train', 'amber', 'Nila', 'lunar craters')[4]
        value = {**case['expected'], 'do_not_claim_complete': False}
        self.assertEqual(reward(case, json.dumps(value)), 0)

    def test_no_json_substring_rescue_or_extra_native_text(self):
        cases = make('train', 'amber', 'Nila', 'lunar craters')
        answer = json.dumps(cases[0]['expected'])
        self.assertEqual(reward(cases[0], 'Done: ' + answer), 0)
        self.assertEqual(reward(cases[-1], cases[-1]['expected'] + '\nDone'), 0)

    def test_group_advantages_are_centered_and_skip_equal_rewards(self):
        self.assertEqual(advantages([.5, .5, .5]), [0, 0, 0])
        values = advantages([0, .5, 1])
        self.assertAlmostEqual(sum(values), 0)
        self.assertAlmostEqual(sum(v*v for v in values) / 3, 1)

    def test_split_user_inputs_are_disjoint(self):
        seen = set()
        for split, label in [('train', 'amber'), ('valid', 'cobalt'), ('test', 'jade')]:
            rows = make(split, label, split, f'{label} facts')
            current = {r['messages'][-1]['content'] for r in rows}
            self.assertEqual(len(current), 8)
            self.assertFalse(current & seen)
            seen |= current

    def test_clipping_and_kl_loss_have_correct_behavior(self):
        import mlx.core as mx
        from scripts.train_kira_a_rl import clipped_loss
        old = mx.array([[-1.0, -2.0]])
        self.assertAlmostEqual(float(clipped_loss(old, old, old, 1.0, 2).item()), -1.0)
        shifted = old + 1
        # ratio exp(1) is clipped to 1.2 for positive advantage.
        self.assertAlmostEqual(float(clipped_loss(shifted, old, shifted, 1, 2).item()), -1.2, places=5)
        self.assertGreater(float(clipped_loss(shifted, old, old, 0, 2).item()), 0)


if __name__ == '__main__':
    unittest.main()
