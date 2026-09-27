import unittest
from kira_live.response_prosody import response_style
from kira_live.mlx_voice import KiraIntegratedQwenVoice
from kira_live.emotion import EmotionState
from types import SimpleNamespace


class ResponseProsodyTests(unittest.TestCase):
    def test_compassion_overrides_excited_listener(self):
        self.assertIn("compassionate", response_style("I'm sorry for your loss.", "excited"))

    def test_uncertainty_overrides_celebration(self):
        self.assertIn("honest", response_style("Great news, but I can't verify that.", "joyful"))

    def test_celebration_and_instruction(self):
        self.assertIn("joyful", response_style("Congratulations!", "neutral"))
        self.assertIn("focused", response_style("Let's take the first step.", "neutral"))

    def test_fallback_preserves_acoustic_control(self):
        self.assertEqual(response_style("The answer is four.", "steady"), "steady")

    def test_real_voice_plan_uses_reply_tone_without_changing_words(self):
        text = "Congratulations!"
        tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: text)
        state = EmotionState({"neutral": 1.0}, valence=0, arousal=.5, confidence=.1, observed_at=0)
        plan = KiraIntegratedQwenVoice.build_plan([1], text, tokenizer, state)
        self.assertEqual(plan.public_text, text)
        self.assertEqual(plan.token_ids, (1,))
        self.assertIn("joyful", plan.style_instruction)
