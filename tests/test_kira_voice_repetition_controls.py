import unittest
from kira_live.response_repetition import phrase_continuations, repetition_requested


class VoiceRepetitionTests(unittest.TestCase):
    def test_only_long_phrases_are_penalized(self):
        self.assertEqual(phrase_continuations([(1, 2, 3)]), {})
        table = phrase_continuations([tuple(range(10))])
        self.assertEqual(table[tuple(range(7))], {7})
        self.assertEqual(len(table), 3)

    def test_deliberate_repeat_is_allowed(self):
        self.assertTrue(repetition_requested('Repeat that please.'))
        self.assertFalse(repetition_requested('Find the best camping spot.'))

    def test_live_weather_and_find_requests_use_tree(self):
        from voice import KiraVoiceMixin
        brain = KiraVoiceMixin()
        self.assertTrue(brain._live_tree_requested('Find the best camping spot in India.'))
        self.assertTrue(brain._live_tree_requested('What is the weather right now?'))
        self.assertFalse(brain._live_tree_requested('Find the best spot without using any tools.'))

    def test_copy_penalty_does_not_touch_unrelated_logits(self):
        import mlx.core as mx
        from kira_live.response_repetition import make_reply_copy_penalty
        with mx.stream(mx.cpu):
            process = make_reply_copy_penalty([tuple(range(8))])
            result = process(mx.array(list(range(7))), mx.zeros((1, 10)))
            self.assertEqual(float(result[0, 7].item()), -4.0)
            self.assertEqual(float(result[0, 8].item()), 0.0)


if __name__ == '__main__':
    unittest.main()
