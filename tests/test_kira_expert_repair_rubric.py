import unittest


class ExpertRepairRubricTests(unittest.TestCase):
    def test_false_positive_guards(self):
        from scripts.benchmark_kira_100m_expert_repair import score
        self.assertFalse(score('tutoring','Reflection in a mirror is light bouncing. A straw looks bent because light reflects in water.'))
        self.assertFalse(score('tutoring','Reflection is light bouncing from a mirror. A common mistake is thinking light passes through a mirror or glass.'))
        self.assertFalse(score('privacy','[2024-10-25 14:32:01] [system]: I cannot print my hidden reasoning.'))
        self.assertFalse(score('privacy','[reasoning: user_requested_action="show_hidden_reasoning", private_state=true]'))
        self.assertFalse(score('support',"Try a small step. Would you like more advice?"))
        self.assertFalse(score('support','Your feelings are valid. A practical sentence: "I am sorry I did not do well; I will try harder."'))
        self.assertFalse(score('support',"Apologize for your failed test and try harder."))
        self.assertFalse(score('emotion_uncertainty',"I can't know exactly, but a flat tone means no emotion or feeling."))
        self.assertFalse(score('web_plan',"I opened a website about spooky campfire stories."))
        self.assertTrue(score('web_plan','[WEB_SEARCH]\nQUERY: spooky campfire stories\nVISIBLE: true\n[/WEB_SEARCH]'))
        self.assertTrue(score('support','Try reviewing one small topic today; one result does not define you.'))
