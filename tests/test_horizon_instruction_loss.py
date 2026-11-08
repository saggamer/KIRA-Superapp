import copy,unittest
import torch
from orchestrator_v15.horizon_instruction_torch import instruction_flow_loss
from orchestrator_v15.horizon_unified_torch import HorizonUnifiedTransformer,UnifiedConfig
from tests.test_horizon_unified_torch import fixture

class InstructionLossTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(12)
 def test_instruction_blind_prediction_has_no_conditioning_gap(self):
  prediction=torch.ones(2,3,4,requires_grad=True)
  r=instruction_flow_loss(prediction,prediction,torch.zeros_like(prediction),torch.ones(2,dtype=torch.bool))
  self.assertTrue(torch.equal(r['conditioning_gap'],torch.zeros(2)));self.assertAlmostEqual(float(r['ranking_loss'].detach()),.1,places=5)
 def test_unknown_or_paraphrase_negative_has_no_ranking_signal(self):
  prediction=torch.ones(2,3,4,requires_grad=True);wrong=torch.full_like(prediction,float('nan'))
  r=instruction_flow_loss(prediction,wrong,torch.zeros_like(prediction),torch.zeros(2,dtype=torch.bool))
  self.assertEqual(float(r['ranking_loss'].detach()),0);self.assertTrue(torch.isfinite(r['loss']));r['loss'].backward()
 def test_counterfactual_software_optimization(self):
  model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters(),lr=.05);x=torch.eye(2);target=torch.tensor([[1.],[-1.]])
  for _ in range(100):
   optimizer.zero_grad();positive=model(x);wrong=model(x.flip(0));r=instruction_flow_loss(positive,wrong,target,torch.ones(2,dtype=torch.bool));r['loss'].backward();optimizer.step()
  self.assertLess(float(r['positive_loss'].detach()),.01);self.assertTrue((r['conditioning_gap']>.1).all())
 def test_core_rejects_noise_and_timestep_changes(self):
  model=HorizonUnifiedTransformer(UnifiedConfig(16,width=16,heads=4,layers=2,latent_width=4,vision_width=8,max_tokens=64,experts=3,attention_experts=1));correct=fixture();correct[1].supervise_text=False;wrong=copy.deepcopy(correct);wrong[1].values=torch.tensor([[5,6,7,8]])
  result=model.instruction_loss(correct,wrong,{2:torch.zeros(1,4,4)},torch.ones(1,dtype=torch.bool),visual_mode='editing');result['loss'].backward()
  self.assertGreater(float(model.blocks[0].visual_attention['editing']['qkv'].weight.grad.norm()),0)
  wrong[-1].timestep+=.1
  with self.assertRaises(ValueError):model.instruction_loss(correct,wrong,{2:torch.zeros(1,4,4)},torch.ones(1,dtype=torch.bool),visual_mode='editing')
  wrong=copy.deepcopy(correct);wrong[1].values=torch.tensor([[5,6,7,8]]);wrong[0].values+=1
  with self.assertRaises(ValueError):model.instruction_loss(correct,wrong,{2:torch.zeros(1,4,4)},torch.ones(1,dtype=torch.bool),visual_mode='editing')
 def test_verified_nonfinite_negative_rejected(self):
  with self.assertRaises(ValueError):instruction_flow_loss(torch.zeros(1,2),torch.full((1,2),float('nan')),torch.zeros(1,2),torch.ones(1,dtype=torch.bool))
 def test_mixed_batch_does_not_treat_identical_prompt_as_negative(self):
  model=HorizonUnifiedTransformer(UnifiedConfig(16,width=16,heads=4,layers=2,latent_width=4,vision_width=8,max_tokens=64))
  positive=fixture();positive[1].supervise_text=False
  for segment in positive:
   segment.values=segment.values.repeat(2,*([1]*(segment.values.ndim-1)))
   if segment.xy is not None:segment.xy=segment.xy.repeat(2,1,1)
   if segment.timestep is not None:segment.timestep=segment.timestep.repeat(2)
  wrong=copy.deepcopy(positive);wrong[1].values[1]=torch.tensor([5,6,7,8])
  with self.assertRaisesRegex(ValueError,'identical instruction'):
   model.instruction_loss(positive,wrong,{2:torch.zeros(2,4,4)},torch.ones(2,dtype=torch.bool),visual_mode='editing')
