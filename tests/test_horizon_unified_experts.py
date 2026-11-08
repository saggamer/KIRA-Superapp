import unittest,tempfile
from pathlib import Path
import torch
from orchestrator_v15.horizon_unified_torch import HorizonUnifiedTransformer,UnifiedConfig
from tests.test_horizon_unified_torch import fixture
class ExpertTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(4);torch.set_num_threads(1)
  self.model=HorizonUnifiedTransformer(UnifiedConfig(16,width=16,heads=4,layers=2,latent_width=4,vision_width=8,max_tokens=64,experts=3))
 def test_explicit_routing_and_isolated_updates(self):
  batch=fixture()
  with self.assertRaises(ValueError):self.model(batch)
  for mode,other in [('editing','creativity'),('creativity','editing')]:
   self.model.zero_grad(set_to_none=True)
   loss=self.model.joint_loss(batch,{2:torch.randn(1,4,4)},visual_mode=mode)['loss'];loss.backward()
   for block in self.model.blocks:
    self.assertGreater(float(block.qkv.weight.grad.norm()),0)
    self.assertGreater(float(block.ffn[0].weight.grad.norm()),0)
    self.assertGreater(float(block.visual_ffn[mode][0].weight.grad.norm()),0)
    self.assertIsNone(block.visual_ffn[other][0].weight.grad)
 def test_expert_roundtrip(self):
  batch=fixture();before=self.model(batch,visual_mode='editing')['flow'].detach()
  with tempfile.TemporaryDirectory() as tmp:
   self.model.save_checkpoint(Path(tmp)/'model');after=HorizonUnifiedTransformer.load_checkpoint(Path(tmp)/'model')
   torch.testing.assert_close(before,after(batch,visual_mode='editing')['flow'],rtol=0,atol=0)
 def test_autocast_keeps_routed_write_dtype_and_gradients(self):
  batch=fixture()
  with torch.autocast('cpu',dtype=torch.bfloat16):
   loss=self.model.joint_loss(batch,{2:torch.randn(1,4,4)},visual_mode='editing')['loss']
  self.assertTrue(torch.isfinite(loss));loss.backward()
  self.assertGreater(float(self.model.blocks[0].visual_ffn['editing'][0].weight.grad.norm()),0)
