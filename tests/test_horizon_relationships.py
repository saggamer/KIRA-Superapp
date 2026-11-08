import copy
import unittest
from orchestrator_v15.horizon_relationships import SceneGraph,validate_conditioning_budget


def observatory():
    return {'entities':[{'id':'teapot','label':'ceramic teapot'},
                        {'id':'spout','label':'spout','kind':'part','owner':'teapot'},
                        {'id':'telescope','label':'brass telescope'}],
            'relations':[{'id':'optical_path','subject':'telescope','predicate':'through','targets':['spout']}]}


def greenhouse():
    return {'entities':[{'id':'house','label':'enclosed glass greenhouse'},
                        {'id':'left','label':'left stone island'}, {'id':'right','label':'right stone island'}],
            'relations':[{'id':'bridge','subject':'house','predicate':'between','targets':['left','right']}]}


class RelationshipGraphTests(unittest.TestCase):
    def test_compilation_keeps_exact_owner_and_target(self):
        graph=SceneGraph.from_dict(observatory())
        self.assertIn("brass telescope through ceramic teapot's spout",graph.render())
        other=observatory()
        other['relations'][0]['targets']=['teapot']
        self.assertNotEqual(graph.fingerprint,SceneGraph.from_dict(other).fingerprint)

    def test_unknown_evidence_cannot_complete_a_scene(self):
        result=SceneGraph.from_dict(observatory()).assess(instances={},observations={},reviewer='agent_visual')
        self.assertFalse(result['complete'])
        self.assertIn('unknown',result['checks'].values())

    def test_telescope_in_body_does_not_satisfy_spout(self):
        graph=SceneGraph.from_dict(observatory())
        instances={'teapot':['pot1'],'spout':['spout1'],'telescope':['scope1']}
        edge={'status':'pass','subject':'telescope','predicate':'through','targets':['teapot_body']}
        result=graph.assess(instances=instances,owners={'spout1':'pot1'},
                            observations={'optical_path':edge},reviewer='human')
        self.assertEqual(result['checks']['relation:optical_path'],'fail')
        edge['targets']=['spout']
        self.assertTrue(graph.assess(instances=instances,owners={'spout1':'pot1'},
                                     observations={'optical_path':edge},reviewer='human')['complete'])

    def test_part_of_wrong_owner_does_not_pass(self):
        graph=SceneGraph.from_dict(observatory())
        result=graph.assess(instances={'teapot':['pot1'],'spout':['spout1'],'telescope':['scope1']},
                            owners={'spout1':'another_pot'},reviewer='human',
                            observations={'optical_path':{'status':'pass','subject':'telescope','predicate':'through','targets':['spout']}})
        self.assertEqual(result['checks']['owner:spout'],'fail')
        self.assertFalse(result['complete'])

    def test_two_islands_cannot_bind_to_one_observed_instance(self):
        graph=SceneGraph.from_dict(greenhouse())
        observation={'bridge':{'status':'pass','subject':'house','predicate':'between','targets':['left','right']}}
        result=graph.assess(instances={'house':['h'],'left':['island1'],'right':['island1']},
                            observations=observation,reviewer='agent_visual')
        self.assertFalse(result['complete'])
        self.assertEqual(result['checks']['relation:bridge'],'fail')

    def test_between_anchor_order_is_symmetric(self):
        one=greenhouse()
        two=copy.deepcopy(one)
        two['relations'][0]['targets'].reverse()
        a,b=map(SceneGraph.from_dict,(one,two))
        self.assertEqual(a.fingerprint,b.fingerprint)
        self.assertEqual(a.signatures,b.signatures)

    def test_scoped_material_is_not_inferred_from_global_style(self):
        value={'entities':[{'id':'pear','label':'pear','material':'clay'}],'relations':[]}
        graph=SceneGraph.from_dict(value)
        self.assertIn('material of this object: clay',graph.render())
        result=graph.assess(instances={'pear':['p']},observations={},materials={'style':'clay'},reviewer='human')
        self.assertEqual(result['checks']['material:pear'],'unknown')

    def test_contradictions_bad_boxes_counts_and_ownership_are_rejected(self):
        invalid=[]
        value=greenhouse()
        value['relations']=[{'id':'a','subject':'left','predicate':'left_of','targets':['right']},
                            {'id':'b','subject':'left','predicate':'right_of','targets':['right']}]
        invalid.append(value)
        value=observatory(); value['entities'][0]['count']=True; invalid.append(value)
        value=observatory(); value['entities'][0]['box']=[0,0,float('nan'),1]; invalid.append(value)
        value=observatory(); value['entities'][1]['owner']='missing'; invalid.append(value)
        value=greenhouse(); value['relations'][0]['targets']=['left','left']; invalid.append(value)
        value=greenhouse(); value['entities'].append(copy.deepcopy(value['entities'][0])); invalid.append(value)
        for record in invalid:
            with self.assertRaises(ValueError):
                SceneGraph.from_dict(record)

    def test_pixel_only_process_cannot_certify_semantic_edges(self):
        with self.assertRaises(ValueError):
            SceneGraph.from_dict(observatory()).assess(instances={},observations={},reviewer='programmatic')

    def test_template_overhead_cannot_silently_truncate_a_scene(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                return '<user> '+messages[0]['content']+' <assistant>'
            def __call__(self,text,**kwargs):
                return {'input_ids':text.split()}
        self.assertEqual(validate_conditioning_budget(Tokenizer(),'scope through spout',5),5)
        with self.assertRaises(ValueError):
            validate_conditioning_budget(Tokenizer(),'scope through spout',4)
