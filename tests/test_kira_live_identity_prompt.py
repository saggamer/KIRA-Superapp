import unittest

from interface import KiraBrain


class KiraLiveIdentityPromptTests(unittest.TestCase):
    def test_live_prompt_separates_public_name_from_donor_provenance(self):
        prompt = KiraBrain.__new__(KiraBrain)._build_system_instruction("kira")
        self.assertIn("introduce yourself as KIRA Live 1", prompt)
        self.assertIn("technical provenance", prompt)
        self.assertIn("do not introduce", prompt)
        self.assertIn("Qwen3.5", prompt)


if __name__ == "__main__":
    unittest.main()
