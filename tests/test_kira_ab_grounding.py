import unittest
from kira_live.grounding_evaluation import conversation_checks, no_regression_score
from scripts.prepare_kira_ab_grounding import build
from kira_live.expert_rl import exact


class GroundingTests(unittest.TestCase):
    def test_gold_and_split_separation(self):
        splits = build()
        self.assertEqual(len(splits['train']), 118)
        self.assertEqual(len(splits['valid']), 24)
        self.assertEqual(len(splits['test']), 24)
        for rows in splits.values():
            for row in rows:
                answer = row['messages'][-1]['content']
                if row['mode']:
                    self.assertTrue(exact(row, answer), row['id'])
                else:
                    self.assertTrue(all(conversation_checks(row, answer).values()), (row['id'], answer))

    def test_no_partial_credit_wrong_number_role_or_completion(self):
        rows = build()['valid']
        number = next(r for r in rows if r.get('rubric', {}).get('exact') == ['13'])
        self.assertFalse(all(conversation_checks(number, '24 minus 11.').values()))
        apology = next(r for r in rows if 'Meera' in r['messages'][-2]['content'])
        self.assertFalse(all(conversation_checks(apology, 'Meera is sorry you broke her vase. It happens.').values()))
        self.assertFalse(all(conversation_checks(apology, apology['messages'][-1]['content'], True).values()))

    def test_selection_does_not_exchange_parent_successes_for_others(self):
        baseline = {'lm_loss': 3, 'generated': {'rows': [{'id': 'a', 'passed': True}, {'id': 'b', 'passed': False}]}}
        regressed = {'lm_loss': 1, 'generated': {'rows': [{'id': 'a', 'passed': False}, {'id': 'b', 'passed': True}]}}
        improved = {'lm_loss': 2, 'generated': {'rows': [{'id': 'a', 'passed': True}, {'id': 'b', 'passed': True}]}}
        self.assertIsNone(no_regression_score(regressed, baseline))
        self.assertEqual(no_regression_score(improved, baseline), (2, -2))

    def test_review_uses_mode_scoped_ids_and_checks_typed_permissions(self):
        from scripts.review_kira_ab_grounding import permission_failures
        cases = build()['test']
        gold = {(r['mode'], r['id']): r for r in cases}
        self.assertEqual(len(gold), len(cases))
        rows = [{'mode': r['mode'], 'id': r['id'], 'category': r['category'], 'answer': r['messages'][-1]['content']} for r in cases]
        self.assertEqual(permission_failures(rows, gold), [])
        message = next(r for r in rows if r['mode'] and r['category'] == 'tool_calling' and 'send_message' in r['answer'])
        message['answer'] = message['answer'].replace('true', 'false')
        self.assertEqual(len(permission_failures(rows, gold)), 1)


if __name__ == '__main__':
    unittest.main()
