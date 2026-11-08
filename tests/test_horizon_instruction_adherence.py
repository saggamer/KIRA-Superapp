import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from orchestrator_v15.horizon_instruction import encode_instruction,InstructionBudgetError
from orchestrator_v15.horizon_adherence import AdherenceContract,AdherenceRequirement

class EncodingTests(unittest.TestCase):
 def test_complete_bytes_and_last_clause_survive(self):
  prompt='  Paint the bag red; do not change the shoes.  '
  e=encode_instruction(prompt,max_prompt_bytes=160,max_context_tokens=1024,visual_tokens=708)
  self.assertEqual(bytes(e.token_ids[1:-1]).decode(),prompt);self.assertFalse(e.evidence()['truncated'])
 def test_exact_limit_unicode_and_overflow(self):
  self.assertEqual(encode_instruction('x'*160,max_prompt_bytes=160).prompt_bytes,160)
  self.assertEqual(encode_instruction('🐈🐈',max_prompt_bytes=8).prompt_bytes,8)
  for prompt in ['x'*161,'🐈🐈🐈']:
   with self.assertRaises(InstructionBudgetError):encode_instruction(prompt,max_prompt_bytes=8 if prompt.startswith('🐈') else 160)
 def test_actual_visual_context_budget(self):
  with self.assertRaises(InstructionBudgetError):encode_instruction('hello',max_prompt_bytes=160,max_context_tokens=712,visual_tokens=708)
  for prompt in ['','   ',None]:
   with self.assertRaises(ValueError):encode_instruction(prompt,max_prompt_bytes=160)
 def test_cli_rejects_question_instead_of_ignoring_it(self):
  r=subprocess.run([sys.executable,'scripts/run_horizon_unified_experts.py','--checkpoint','/tmp/absent','--mode','understanding','--source','/tmp/source','--prompt','How many people?'],capture_output=True,text=True)
  self.assertEqual(r.returncode,2);self.assertIn('question conditioning',r.stderr)
 def test_cli_overflow_fails_before_loading_weights(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'model').mkdir();(p/'runtime.json').write_text(json.dumps({'train_latent_scale':1,'tokenizer':{'max_prompt_bytes':160}}));(p/'model/config.json').write_text(json.dumps({'config':{'max_tokens':1024}}))
   r=subprocess.run([sys.executable,'scripts/run_horizon_unified_experts.py','--checkpoint',str(p),'--mode','creativity','--prompt','x'*161,'--output',str(p/'result.png')],capture_output=True,text=True)
   self.assertEqual(r.returncode,2);self.assertIn('No instruction was truncated',r.stderr);self.assertNotIn('Traceback',r.stderr)

class AdherenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'output';self.path.write_bytes(b'fixed image evidence')
  self.contract=AdherenceContract('Exactly two red cubes. No text.','creativity',(
   AdherenceRequirement('two_cubes','count','Exactly two cubes are visible.'),AdherenceRequirement('no_text','forbidden','No text is visible.')))
 def tearDown(self):self.tmp.cleanup()
 def observation(self):
  return {'contract_sha256':self.contract.fingerprint,'output_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),'reviewer':'assistant_visual','checks':{r.id:{'status':'pass','evidence':'Explicit visual observation.'} for r in self.contract.requirements}}
 def test_all_checks_required_aesthetics_cannot_compensate(self):
  obs=self.observation();self.assertTrue(self.contract.assess(obs,self.path)['instruction_success'])
  obs['checks']['no_text']={'status':'fail','evidence':'Letters are printed on one cube.'};obs['aesthetic_score']=1
  result=self.contract.assess(obs,self.path);self.assertFalse(result['instruction_success']);self.assertEqual(result['failures'],['no_text'])
 def test_missing_check_stays_unknown(self):
  obs=self.observation();del obs['checks']['no_text'];result=self.contract.assess(obs,self.path)
  self.assertEqual(result['status'],'unknown');self.assertFalse(result['instruction_success'])
 def test_tampered_output_and_changed_contract_rejected(self):
  obs=self.observation();self.path.write_bytes(b'changed')
  with self.assertRaises(ValueError):self.contract.assess(obs,self.path)
  obs=self.observation();obs['contract_sha256']='0'*64
  with self.assertRaises(ValueError):self.contract.assess(obs,self.path)
 def test_programmatic_similarity_cannot_certify_semantics(self):
  obs=self.observation();obs['reviewer']='programmatic'
  with self.assertRaises(ValueError):self.contract.assess(obs,self.path)
 def test_unknown_ids_and_missing_evidence_rejected(self):
  obs=self.observation();obs['checks']['wrong']={'status':'pass','evidence':'test'}
  with self.assertRaises(ValueError):self.contract.assess(obs,self.path)
  obs=self.observation();obs['checks']['no_text']['evidence']=''
  with self.assertRaises(ValueError):self.contract.assess(obs,self.path)
 def test_exact_request_and_source_binding(self):
  digest=hashlib.sha256(self.path.read_bytes()).hexdigest();contract=AdherenceContract('Remove only the man.','editing',self.contract.requirements,digest)
  contract.validate_request('Remove only the man.','editing',self.path)
  with self.assertRaises(ValueError):contract.validate_request('Remove both people.','editing',self.path)
  self.assertEqual(self.contract.fingerprint,AdherenceContract.from_dict(self.contract.to_dict()).fingerprint)
