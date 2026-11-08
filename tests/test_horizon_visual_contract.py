import copy
import unittest
import numpy as np
from PIL import Image
from orchestrator_v15.horizon_visual_contract import VisualContract, preserve_outside_mask, condition_edit_prompt
from tests import test_horizon_experience as fixtures


def teapot():
    return {'brief':'A ceramic camera teapot with its lens as the spout.',
            'scene_graph':{'entities':[{'id':'pot','label':'camera teapot','material':'ceramic'},
                                       {'id':'lens','label':'camera lens','kind':'part','owner':'pot'}],
                           'relations':[]},'part_roles':{'lens':'pouring_spout'}}


class ContractTests(unittest.TestCase):
    def test_function_binds_to_owned_part_before_original_style(self):
        contract=VisualContract.from_dict(teapot())
        prompt=contract.compile()
        self.assertIn("camera teapot's camera lens serves as the camera teapot's pouring spout",prompt)
        self.assertIn('hollow outlet',prompt)
        self.assertLess(prompt.index('pouring spout'),prompt.index('Requested image:'))
        self.assertEqual(contract.fingerprint,VisualContract.from_dict(contract.to_dict()).fingerprint)

    def test_glass_cue_is_local_and_does_not_change_other_materials(self):
        value=teapot();value['scene_graph']['entities'][0]['material']='transparent glass'
        value['scene_graph']['entities'][1]['material']='copper'
        text=VisualContract.from_dict(value).compile()
        self.assertIn('background visible through its walls',text)
        self.assertIn("Exactly 1 camera teapot's camera lens; this object is copper",text)

    def test_invalid_functional_binding_is_rejected(self):
        for roles in ({'pot':'pouring_spout'},{'absent':'handle'},{'lens':'unsupported'}):
            value=teapot();value['part_roles']=roles
            with self.assertRaises(ValueError):VisualContract.from_dict(value)

    def test_missing_role_evidence_does_not_certify_success(self):
        contract=VisualContract.from_dict(teapot())
        observed=dict(instances={'pot':['p'],'lens':['l']},owners={'l':'p'},materials={'pot':'ceramic'},
                      observations={},reviewer='agent_visual')
        result=contract.assess(**observed)
        self.assertFalse(result['complete']);self.assertEqual(result['checks']['function:lens'],'unknown')
        self.assertTrue(contract.assess(**observed,role_observations={'lens':'pass'})['complete'])
        with self.assertRaises(ValueError):contract.assess(**observed,role_observations={'lens':True})


class MaskProtectionTests(unittest.TestCase):
    def test_mask_protection_does_not_rewrite_all_objects_by_default(self):
        prompt='Replace all the food with mashed potatoes.'
        mask=Image.new('L',(24,24),255)
        self.assertEqual(condition_edit_prompt(prompt,mask),prompt)
        self.assertIn('selected',condition_edit_prompt(prompt,mask,regional=True))
        with self.assertRaises(ValueError):condition_edit_prompt(prompt,regional=True)

    def test_feather_never_changes_protected_pixels_including_holes(self):
        rng=np.random.default_rng(17)
        original=rng.integers(0,256,(24,24,3),dtype=np.uint8)
        candidate=255-original
        mask=np.zeros((24,24),dtype=np.uint8);mask[4:20,4:20]=255;mask[9:12,9:12]=0
        out,evidence=preserve_outside_mask(Image.fromarray(original),Image.fromarray(candidate),Image.fromarray(mask),feather=3)
        result=np.asarray(out)
        self.assertTrue(np.array_equal(original[mask==0],result[mask==0]))
        self.assertTrue((original[mask>0]!=result[mask>0]).any())
        self.assertTrue(evidence['protected_pixels_exact']);self.assertIsNone(evidence['instruction_success'])

    def test_invalid_dimensions_empty_and_color_masks_fail(self):
        source=Image.new('RGB',(10,10));mask=Image.new('L',(10,10),255)
        for candidate,bad in [(Image.new('RGB',(11,10)),mask),(source,Image.new('L',(10,10))),
                              (source,Image.new('RGB',(10,10),'white'))]:
            with self.assertRaises(ValueError):preserve_outside_mask(source,candidate,bad)
        with self.assertRaises(ValueError):preserve_outside_mask(source,source,mask,feather=float('nan'))


class ContractExperienceTests(unittest.TestCase):
    setUp=fixtures.HorizonExperienceTests.setUp
    record=fixtures.HorizonExperienceTests.record

    def test_held_out_lessons_are_not_retrieved_but_exact_retry_is_explicit(self):
        item=self.record(split='test')
        self.memory.review(item['id'],reviewer='human',instruction=False,preservation=False,
                           critique='Wrong edit',policies=['preserve_scene'],correction='Retain the existing trunk.')
        self.assertEqual(self.memory.plan('editing','Edit another elephant')['lesson_episodes'],[])
        self.assertIn('Retain the existing trunk',self.memory.plan('editing',item['prompt'],retry_of=item['id'])['effective_prompt'])
        child=self.record(parent=item['id'])
        self.assertEqual(child['split'],'test')

    def test_functional_success_needs_role_review_and_retry_cannot_rebind(self):
        contract=teapot();item=self.record(mode='creativity',source=None,prompt=contract['brief'],visual_contract=contract)
        evidence={'instances':{'pot':['p'],'lens':['l']},'owners':{'l':'p'},'materials':{'pot':'ceramic'},'observations':{}}
        fields=dict(reviewer='human',instruction=True,preservation=True,creativity=True,critique='Observed physical role',relationship_observation=evidence)
        with self.assertRaises(ValueError):self.memory.review(item['id'],**fields)
        self.memory.review(item['id'],**fields,role_observations={'lens':'pass'})
        changed=copy.deepcopy(contract);changed['part_roles']['lens']='handle'
        with self.assertRaises(ValueError):self.record(mode='creativity',source=None,prompt=contract['brief'],parent=item['id'],visual_contract=changed)


if __name__=='__main__':unittest.main()
