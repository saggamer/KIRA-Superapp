import unittest
import ast
import json
from pathlib import Path
from kira_live.conversation_rubric import score_reply


class ConversationRubricTests(unittest.TestCase):
    def test_repair_examples_do_not_copy_probe_prompts(self):
        root = Path(__file__).parents[1]
        tree = ast.parse((root / "scripts/benchmark_kira_live_conversation_release.py").read_text())
        cases = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets))
        probes = {messages[-1]["content"] for _, messages, _, _ in cases}
        rows = [json.loads(line) for line in (root / "training_data/kira_live_conversation_repair_v1/train.jsonl").read_text().splitlines()]
        self.assertFalse(probes.intersection(row["messages"][-2]["content"] for row in rows))
    def test_fake_email_is_not_saved_by_keyword(self):
        self.assertFalse(score_reply("no_fake_email", "Here is your draft: I've sent an email to you.", ("draft",), ()))

    def test_question_after_invented_booking_is_not_clarification(self):
        self.assertFalse(score_reply("ambiguity", "Location: room 304. What do you need help with?", ("what",), ()))
        self.assertTrue(score_reply("ambiguity", "What would you like to book?", ("what",), ()))

    def test_ai_personal_feelings_are_not_emotion_calibration(self):
        self.assertFalse(score_reply("uncertain_emotion", "I can't know, but I'm feeling anxious.", ("can't",), ()))

    def test_single_sentence_constraint(self):
        self.assertFalse(score_reply("frustration", "Sorry. I understand. What is your question?", ("sorry",), ()))

    def test_grief_does_not_invert_who_died(self):
        self.assertFalse(score_reply("grief", "I'm sorry your friend lost a loved one.", ("sorry",), ()))


if __name__ == "__main__":
    unittest.main()
