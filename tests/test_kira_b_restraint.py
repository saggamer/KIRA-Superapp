import unittest

from scripts.converse_kira_b_repair import DIALOGUES, TRANSFER
from scripts.prepare_kira_b_restraint import TRAIN, VALID, rows


class BRestraintDataTests(unittest.TestCase):
    def test_splits_and_diagnostics_do_not_share_user_prompts(self):
        seen = {p for ps in DIALOGUES.values() for p in ps} | {p for _, p in TRANSFER}
        for entries, split in ((TRAIN, 'train'), (VALID, 'valid')):
            prompts = set()
            for row in rows(entries, split):
                users = [m['content'] for m in row['messages'] if m['role'] == 'user']
                self.assertFalse(set(users) & seen)
                self.assertFalse(set(users) & prompts)
                prompts.update(users)
            seen.update(prompts)

    def test_corrective_data_is_small_varied_and_complete(self):
        self.assertEqual(len(TRAIN), 40)
        self.assertEqual(len(VALID), 8)
        self.assertGreaterEqual(len({r[0] for r in TRAIN}), 10)
        answers = []
        for split, entries in (('train', TRAIN), ('valid', VALID)):
            for row in rows(entries, split):
                roles = [m['role'] for m in row['messages']]
                self.assertEqual(roles[0], 'system')
                self.assertEqual(roles[1:], ['user', 'assistant'] * ((len(roles) - 1) // 2))
                self.assertTrue(all(m['content'].strip() for m in row['messages']))
                answers.append(row['messages'][-1]['content'])
        self.assertEqual(len(answers), len(set(answers)))

    def test_targets_retain_warmth_without_bad_boilerplate(self):
        targets = [a for _, _, a in TRAIN]
        self.assertTrue(any('congratulations' in a.lower() for a in targets))
        self.assertTrue(any('do not have to' in a for a in targets))
        self.assertFalse(any('erase my memory' in a or 'take your word for it' in a for a in targets))


if __name__ == '__main__':
    unittest.main()
