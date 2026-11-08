import unittest
from orchestrator_v15.horizon_creativity import BRIEFS, RUBRIC, validate_review
from tests import test_horizon_experience as experience_tests


class CreativityRubricTests(unittest.TestCase):
    def test_briefs_have_distinct_concepts_and_explicit_constraints(self):
        self.assertEqual(len({brief['id'] for brief in BRIEFS}),6)
        self.assertEqual(len({brief['prompt'] for brief in BRIEFS}),6)
        for brief in BRIEFS:
            self.assertEqual(len(brief['criteria']),4)

    def test_appealing_image_does_not_pass_an_unfulfilled_brief(self):
        result=validate_review([False,True,False,True],dict(brief_fidelity=2,concept_integration=3,visual_coherence=4,composition=4))
        self.assertFalse(result['instruction'])
        self.assertTrue(result['creativity'])
        with self.assertRaises(ValueError):
            validate_review([False,True,False,True],dict(brief_fidelity=4,concept_integration=4,visual_coherence=4,composition=4))

    def test_scores_are_complete_bounded_explicit_judgments(self):
        for scores in ({},dict.fromkeys(RUBRIC,5),dict.fromkeys(RUBRIC,True)):
            with self.assertRaises(ValueError):
                validate_review([True]*4,scores)


class CreativityExperienceTests(unittest.TestCase):
    setUp=experience_tests.HorizonExperienceTests.setUp
    record=experience_tests.HorizonExperienceTests.record

    def test_creative_lesson_transfers_without_edit_mode_leakage(self):
        episode=self.record(mode='creativity',prompt='A forest greenhouse shaped like glass petals',source=None)
        self.memory.review(episode['id'],reviewer='agent_visual',instruction=False,preservation=True,
                           creativity=False,critique='Attractive flower, but no recognizable enclosed building.',
                           policies=['anchor_subject','show_function'])
        plan=self.memory.plan('creativity','A ceramic teapot observatory')
        self.assertEqual(set(plan['policies']),{'anchor_subject','show_function'})
        self.assertEqual(self.memory.plan('editing','A ceramic teapot observatory')['policies'],[])

    def test_creative_replay_needs_both_brief_and_creativity_acceptance(self):
        first=self.record(mode='creativity',source=None,prompt='An observatory teapot')
        self.memory.review(first['id'],reviewer='agent_visual',instruction=False,preservation=True,
                           creativity=False,critique='Telescope and functional teapot not integrated.')
        second_image=self.root/'second.png'
        second_image.write_bytes(b'new creative evidence')
        second=self.record(mode='creativity',source=None,prompt=first['prompt'],parent=first['id'],output=second_image)
        self.memory.review(second['id'],reviewer='agent_visual',instruction=True,preservation=True,
                           creativity=False,critique='Recognizable but creative integration is not established.')
        self.assertEqual(self.memory.replay_pairs(),[])
        self.memory.review(second['id'],reviewer='human',instruction=True,preservation=True,
                           creativity=True,critique='Both the full brief and integrated concept are visible.',supersede=True)
        self.assertEqual(len(self.memory.replay_pairs()),1)
