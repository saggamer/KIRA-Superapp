import ast,contextlib,types,unittest
from pathlib import Path
from scripts.horizon_big_train_patches import complete_instruction_patch

class Tokenizer:
 def apply_chat_template(self,messages,**kwargs):return 'template:'+messages[0]['content']
 def __call__(self,prompt,**kwargs):
  assert kwargs.get('truncation') is False
  return {'input_ids':list(prompt)}
class Pipeline:
 def __init__(self):self.tokenizer=Tokenizer();self.called=False
 def encode_prompt(self,**kwargs):self.called=True;return kwargs['prompt'],'ids'

class NativeInstructionGuardTests(unittest.TestCase):
 def entry(self,mode,limit):
  path=Path('training_runs/horizon1_big_20261020/results')/f'horizon_train_{mode}.py'
  patched=complete_instruction_patch(path.read_text());tree=ast.parse(patched)
  function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compute_text_embeddings')
  namespace={'args':types.SimpleNamespace(max_sequence_length=limit,text_encoder_out_layers=(9,18,27)),'torch':types.SimpleNamespace(no_grad=contextlib.nullcontext)}
  exec(compile(ast.Module(body=[function],type_ignores=[]),'guarded_native_function','exec'),namespace)
  return namespace['compute_text_embeddings']
 def test_both_real_trainer_functions_keep_entire_prompt(self):
  for mode in ('editing','creativity'):
   pipeline=Pipeline();fn=self.entry(mode,100)
   text=['Paint red. Do not alter the shoes.']
   result=fn(text,pipeline);self.assertEqual(result[0],text);self.assertTrue(pipeline.called)
 def test_overflow_rejected_before_encoding_or_caching(self):
  for mode in ('editing','creativity'):
   pipeline=Pipeline();fn=self.entry(mode,12)
   with self.assertRaisesRegex(ValueError,'no instruction was truncated'):fn(['Paint red. Do not alter shoes.'],pipeline)
   self.assertFalse(pipeline.called)
 def test_changed_upstream_and_duplicate_patch_fail_closed(self):
  with self.assertRaises(ValueError):complete_instruction_patch('def another_function(): pass')
  original=Path('training_runs/horizon1_big_20261020/results/horizon_train_editing.py').read_text()
  with self.assertRaises(ValueError):complete_instruction_patch(complete_instruction_patch(original))
