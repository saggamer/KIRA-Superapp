import unittest
from scripts.benchmark_kira_ab_comparison import checks, DIALOGUES, TRANSFER


class ABComparisonTests(unittest.TestCase):
    def test_twelve_unique_cases(self):
        cases = [name for prompts in DIALOGUES.values() for name, _ in prompts]+[name for name, _ in TRANSFER]
        self.assertEqual(len(cases), 12)
        self.assertEqual(len(set(cases)), 12)

    def test_exact_answers_cannot_pass_just_for_brevity(self):
        for case, good, bad in [('identity', 'KIRA Live 1.', 'mine.'),
                                ('arithmetic', '8.', '17.'),
                                ('feeling_memory', 'Relieved.', 'Disappointed.'),
                                ('angle', '30 degrees.', '60 degrees.')]:
            self.assertTrue(all(checks(case, good).values()))
            self.assertFalse(all(checks(case, bad).values()))
            self.assertFalse(all(checks(case, good, truncated=True).values()))

    def test_draft_requires_event_and_speaker_ownership(self):
        self.assertTrue(all(checks('apology', 'Sana, I am sorry for spilling tea on your notebook. I will replace it.').values()))
        for wrong in ('Sana, I am sorry. You used your notebook with tea.',
                      'Sana is sorry she spilled tea on your notebook. She will replace it.',
                      'Sana, I am sorry you spilled tea on your notebook. I will replace it.'):
            self.assertFalse(all(checks('apology', wrong).values()))
        self.assertTrue(all(checks('revised_apology', 'Ravi, I am sorry I tore your comic book. I will replace it.').values()))

    def test_stale_reply_and_thoughts_fail(self):
        self.assertFalse(all(checks('story', 'I am sorry for your notebook.').values()))
        self.assertFalse(all(checks('arithmetic', '<think>8</think>').values()))


if __name__ == '__main__':
    unittest.main()
