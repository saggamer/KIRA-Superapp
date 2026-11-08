import unittest
from tests import test_horizon_experience as fixtures
from tests.test_horizon_relationships import observatory


def correct_observations():
    return {'instances':{'teapot':['p'],'spout':['s'],'telescope':['t']},'owners':{'s':'p'},
            'observations':{'optical_path':{'status':'pass','subject':'telescope','predicate':'through','targets':['spout']}}}


class RelationshipExperienceTests(unittest.TestCase):
    setUp=fixtures.HorizonExperienceTests.setUp
    record=fixtures.HorizonExperienceTests.record

    def episode(self, **kwargs):
        return self.record(mode='creativity',source=None,scene_graph=observatory(),**kwargs)

    def test_graph_success_requires_explicit_relationship_evidence(self):
        item=self.episode()
        with self.assertRaises(ValueError):
            self.memory.review(item['id'],reviewer='agent_visual',instruction=True,preservation=True,
                               creativity=True,critique='A pretty scene')
        self.memory.review(item['id'],reviewer='agent_visual',instruction=True,preservation=True,
                           creativity=True,critique='The telescope visibly exits the owned spout.',
                           relationship_observation=correct_observations())
        self.assertTrue(self.memory.get(item['id'])['review']['relationship_assessment']['complete'])

    def test_retry_inherits_graph_and_cannot_switch_the_target_part(self):
        item=self.episode()
        self.memory.review(item['id'],reviewer='agent_visual',instruction=False,preservation=True,
                           critique='Wrong part',policies=['bind_relationships'])
        plan=self.memory.plan('creativity',item['prompt'],retry_of=item['id'])
        self.assertIn("teapot's spout",plan['effective_prompt'])
        other=observatory(); other['relations'][0]['targets']=['teapot']
        with self.assertRaises(ValueError):
            self.memory.plan('creativity',item['prompt'],retry_of=item['id'],scene_graph=other)
        with self.assertRaises(ValueError):
            self.record(mode='creativity',source=None,parent=item['id'],scene_graph=other)

    def test_retrieval_prefers_matching_relation_anchor_over_shared_nouns(self):
        correct=self.episode(prompt='A creative vessel')
        self.memory.review(correct['id'],reviewer='agent_visual',instruction=False,preservation=True,
                           critique='Spout attachment failed',policies=['bind_relationships'])
        graph=observatory(); graph['relations'][0]['targets']=['teapot']
        unrelated=self.record(mode='creativity',source=None,prompt='A creative vessel',scene_graph=graph)
        self.memory.review(unrelated['id'],reviewer='agent_visual',instruction=False,preservation=True,
                           critique='Different target',policies=['bind_relationships'])
        plan=self.memory.plan('creativity','A new object',scene_graph=observatory(),limit=1)
        self.assertEqual(plan['lesson_episodes'],[correct['id']])

    def test_graph_conditioned_successful_repair_exports_its_contract(self):
        parent=self.episode()
        self.memory.review(parent['id'],reviewer='human',instruction=False,preservation=True,critique='Missing telescope')
        better=self.root/'better.png'; better.write_bytes(b'better relationship evidence')
        child=self.record(mode='creativity',source=None,parent=parent['id'],output=better)
        self.memory.review(child['id'],reviewer='human',instruction=True,preservation=True,
                           creativity=True,critique='Exact optical path observed.',relationship_observation=correct_observations())
        pair=self.memory.replay_pairs()[0]
        self.assertEqual(pair['scene_graph']['relations'][0]['targets'],['spout'])
