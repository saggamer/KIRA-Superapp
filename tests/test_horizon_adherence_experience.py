import hashlib,copy,tempfile,unittest
from pathlib import Path
from orchestrator_v15.horizon_adherence import AdherenceContract,AdherenceRequirement
from orchestrator_v15.horizon_experience import HorizonExperience

class AdherenceExperienceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.source=self.root/'source';self.source.write_bytes(b'source');self.output=self.root/'initial';self.output.write_bytes(b'bad output');self.memory=HorizonExperience(self.root/'memory')
  self.contract=AdherenceContract('Remove only the man.','editing',(AdherenceRequirement('remove','action','The man is absent.'),AdherenceRequirement('retain','preservation','The woman and background are unchanged.')),hashlib.sha256(self.source.read_bytes()).hexdigest())
 def record(self,**kwargs):
  values={'mode':'editing','prompt':self.contract.prompt,'source':self.source,'output':self.output,'model':'fixture','adherence_contract':self.contract.to_dict()};values.update(kwargs);return self.memory.record(**values)
 def observations(self,item,status='pass'):
  c=AdherenceContract.from_dict(item['adherence_contract'])
  return {'contract_sha256':c.fingerprint,'output_sha256':item['assets']['output']['sha256'],'reviewer':'agent_visual','checks':{r.id:{'status':status,'evidence':'Observed requirement result.' if status!='unknown' else ''} for r in c.requirements}}
 def review(self,item,**kwargs):
  fields={'reviewer':'agent_visual','instruction':False,'preservation':False,'critique':'Requirement review.'};fields.update(kwargs);return self.memory.review(item['id'],**fields)
 def test_unknown_checks_cannot_be_success(self):
  item=self.record()
  with self.assertRaises(ValueError):self.review(item,instruction=True,preservation=True)
  with self.assertRaises(ValueError):self.review(item,instruction=True,preservation=True,adherence_observations=self.observations(item,'unknown'))
 def test_retry_inherits_and_cannot_relax_requirements(self):
  parent=self.record();other=self.root/'repair';other.write_bytes(b'repair')
  child=self.record(output=other,parent=parent['id'],adherence_contract=None)
  self.assertEqual(child['adherence_contract'],parent['adherence_contract'])
  changed=self.contract.to_dict();changed['requirements']=changed['requirements'][:1]
  with self.assertRaises(ValueError):self.record(output=other,parent=parent['id'],adherence_contract=changed)
 def test_replay_needs_explicit_failure_then_all_requirement_success(self):
  parent=self.record();self.review(parent,adherence_observations=self.observations(parent,'fail'))
  other=self.root/'repair';other.write_bytes(b'correct output');child=self.record(output=other,parent=parent['id'])
  self.review(child,instruction=True,preservation=True,adherence_observations=self.observations(child))
  pairs=self.memory.replay_pairs();self.assertEqual(len(pairs),1);self.assertEqual(pairs[0]['adherence_contract'],self.contract.to_dict())
  # Assessment booleans cannot certify stale/changed requirement observations.
  item=self.memory.get(child['id']);item['review']['adherence_observations']['checks']['remove']['status']='unknown';self.memory._write(item)
  self.assertEqual(self.memory.replay_pairs(),[])
 def test_unknown_parent_is_not_a_verified_negative(self):
  parent=self.record();self.review(parent,adherence_observations=self.observations(parent,'unknown'))
  other=self.root/'repair';other.write_bytes(b'correct output');child=self.record(output=other,parent=parent['id'])
  self.review(child,instruction=True,preservation=True,adherence_observations=self.observations(child))
  self.assertEqual(self.memory.replay_pairs(),[])
 def test_reviewer_provenance_and_source_binding(self):
  item=self.record();obs=self.observations(item);obs['reviewer']='human'
  with self.assertRaises(ValueError):self.review(item,instruction=True,preservation=True,adherence_observations=obs)
  self.source.write_bytes(b'different source')
  with self.assertRaises(ValueError):self.record()
 def test_heldout_repairs_never_enter_replay(self):
  parent=self.record(split='test');self.review(parent,adherence_observations=self.observations(parent,'fail'))
  other=self.root/'repair';other.write_bytes(b'correct output');child=self.record(output=other,parent=parent['id'])
  self.review(child,instruction=True,preservation=True,adherence_observations=self.observations(child))
  self.assertEqual(child['split'],'test');self.assertEqual(self.memory.replay_pairs(),[])
 def test_reencoded_heldout_pixels_cannot_teach_or_replay(self):
  from PIL import Image,PngImagePlugin
  source=self.root/'original.png';alias=self.root/'alias.png';image=Image.new('RGB',(8,8),'green');image.save(source)
  metadata=PngImagePlugin.PngInfo();metadata.add_text('comment','different file encoding');image.save(alias,pnginfo=metadata)
  self.assertNotEqual(source.read_bytes(),alias.read_bytes())
  self.memory.record(mode='editing',prompt='Paint the object red',source=source,output=self.output,model='fixture',split='test',group='heldout')
  parent=self.memory.record(mode='editing',prompt='Paint the object red',source=alias,output=self.output,model='fixture',group='apparently_other_group')
  self.memory.review(parent['id'],reviewer='agent_visual',instruction=False,preservation=False,critique='Failed edit',policies=['preserve_scene'])
  self.assertEqual(self.memory.plan('editing','Paint the object red')['lesson_episodes'],[])
  repaired=self.root/'reencoded-repair';repaired.write_bytes(b'repair')
  child=self.memory.record(mode='editing',prompt=parent['prompt'],source=alias,output=repaired,model='fixture',parent=parent['id'])
  self.memory.review(child['id'],reviewer='agent_visual',instruction=True,preservation=True,critique='Repaired')
  self.assertEqual(self.memory.replay_pairs(),[])
