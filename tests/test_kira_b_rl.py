import unittest
from kira_live.conversation_rl import score, sentences
from scripts.prepare_kira_b_rl import make_data
from scripts.train_kira_b_rl import eligible


class BConversationRLTests(unittest.TestCase):
    def test_data_is_prompt_only_and_splits_are_disjoint(self):
        data = make_data()
        self.assertEqual({k: len(v) for k, v in data.items()}, {'train': 12, 'valid': 6, 'test': 12})
        seen = set()
        for cases in data.values():
            for case in cases:
                self.assertEqual(case['messages'][-1]['role'], 'user')
                key = tuple(m['content'] for m in case['messages'] if m['role'] == 'user')
                self.assertNotIn(key, seen)
                seen.add(key)
                self.assertNotIn('answer', case)
                self.assertNotIn('chosen', case)

    def test_memory_and_identity_rewards_cannot_be_gamed(self):
        cases = make_data()['train']
        self.assertEqual(score(cases[2], 'Clover.')['reward'], 1)
        for wrong in ('Hazel.', 'Hazel and Clover.', 'KIRA Live 1.', '', '<think>Clover</think>'):
            self.assertEqual(score(cases[2], wrong)['reward'], 0)
        self.assertEqual(score(cases[11], 'KIRA Live 1.')['reward'], 1)
        self.assertFalse(score(cases[2], 'Clover', truncated=True)['passed'])

    def test_apology_ownership_and_sentence_requirements(self):
        case = make_data()['train'][4]
        self.assertTrue(score(case, 'Leena, I am sorry I broke your cup. I will replace it.')['passed'])
        self.assertEqual(score(case, 'Leena, I am sorry I broke my cup. I will replace it.')['reward'], 0)
        self.assertFalse(score(case, 'Leena, I am sorry I broke your cup.')['passed'])
        self.assertEqual(score(case, 'Leena is sorry she broke your cup. She will replace it.')['reward'], 0)
        arrival = make_data()['train'][5]
        self.assertEqual(score(arrival, 'Yusuf will arrive twenty minutes late.')['reward'], 0)
        self.assertTrue(score(arrival, 'Yusuf, I will arrive twenty minutes late.')['passed'])

    def test_specific_listening_without_emotional_invention(self):
        case = make_data()['train'][8]
        self.assertTrue(score(case, 'You miss those evening walks with your sister.')['passed'])
        self.assertEqual(score(case, 'Your sister is sad. You miss the walks.')['reward'], 0)
        self.assertFalse(score(case, 'You should try more evening walks with your sister.')['passed'])
        self.assertEqual(score(case, 'Your sister has changed a lot. You miss those evening walks.')['reward'], 0)

    def test_rl_scope_and_sentence_parser(self):
        self.assertTrue(eligible('shared_conversation.0.up.weight'))
        for name in ('routing_branches.0.up.2.weight', 'model.layers.23.mlp.up_proj.weight', 'ngram_layers.0.embeddings.weight'):
            self.assertFalse(eligible(name))
        self.assertEqual(sentences('One sentence. Another sentence.'), 2)
        self.assertEqual(sentences('One sentence.'), 1)


if __name__ == '__main__':
    unittest.main()
