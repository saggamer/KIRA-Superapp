import unittest
from kira_live.public_guard import guard_public_reply
from kira_live.policy import live_text_system_instruction


class ReleasePolishTests(unittest.TestCase):
    def test_identity_is_only_public_kira_name(self):
        for prompt in ('Who are you?', 'What is your name?', 'Which model are you?', 'Identify yourself.'):
            self.assertEqual(guard_public_reply(prompt, 'I am Qwen.'), "I'm KIRA Live 1.")

    def test_provenance_question_is_not_misrepresented(self):
        self.assertEqual(guard_public_reply('What is your donor backbone?', 'Qwen3.5.'), 'Qwen3.5.')
        self.assertIn('identify\n  yourself only as KIRA Live 1', live_text_system_instruction())
