import tempfile
import unittest
from pathlib import Path
import torch
from dataclasses import replace
from orchestrator_v15.horizon_context import ContextSpan,generalized_attention_mask
from orchestrator_v15.horizon_unified_torch import HorizonUnifiedTransformer,UnifiedConfig,unified_attention_mask
from tests.test_horizon_unified_torch import fixture
from tests.test_horizon_context import layout

class BagelArchitectureTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(9);torch.set_num_threads(1)
  self.config=UnifiedConfig(16,width=16,heads=4,layers=2,latent_width=4,vision_width=8,max_tokens=64)
 def test_vectorized_mask_matches_reference(self):
  cases=[layout(),[ContextSpan('text',3,0),ContextSpan('semantic_vision',2,1,image_id='target'),ContextSpan('noisy_latent',2,2,image_id='target')]]
  for spans in cases:self.assertEqual(unified_attention_mask(spans).tolist(),generalized_attention_mask(spans))
 def test_upcycle_dense_preserves_both_modes_and_understanding(self):
  dense=HorizonUnifiedTransformer(self.config).eval();mot=HorizonUnifiedTransformer.upcycle(dense)
  segments=fixture()
  for mode in ('editing','creativity'):
   before=dense(segments,visual_mode=mode);after=mot(segments,visual_mode=mode)
   torch.testing.assert_close(before['hidden'],after['hidden'],rtol=1e-5,atol=1e-6)
  torch.testing.assert_close(dense(segments[:2])['text_logits'],mot(segments[:2])['text_logits'],rtol=1e-5,atol=1e-6)
  self.assertIs(mot.text.weight,mot.language_head.weight)
 def test_upcycle_retains_existing_ffn_experts(self):
  donor=HorizonUnifiedTransformer(replace(self.config,experts=3))
  with torch.no_grad():donor.blocks[0].visual_ffn['editing'][0].weight.add_(0.3)
  mot=HorizonUnifiedTransformer.upcycle(donor);segments=fixture()
  for mode in ('editing','creativity'):
   torch.testing.assert_close(donor(segments,visual_mode=mode)['flow'],mot(segments,visual_mode=mode)['flow'],rtol=1e-5,atol=1e-6)
 def test_visual_attention_updates_only_active_expert(self):
  model=HorizonUnifiedTransformer(replace(self.config,experts=3,attention_experts=1,qk_norm=1))
  segments=fixture()
  for mode,other in [('editing','creativity'),('creativity','editing')]:
   model.zero_grad(set_to_none=True)
   with torch.autocast('cpu',dtype=torch.bfloat16):loss=model.joint_loss(segments,{2:torch.randn(1,4,4)},visual_mode=mode)['loss']
   loss.backward()
   for block in model.blocks:
    self.assertGreater(float(block.visual_attention[mode]['qkv'].weight.grad.norm()),0)
    self.assertGreater(float(block.visual_attention[mode]['q_norm'].weight.grad.norm()),0)
    self.assertGreater(float(block.qkv.weight.grad.norm()),0)
    self.assertIsNone(block.visual_attention[other]['qkv'].weight.grad)
 def test_full_experts_communicate_across_modalities(self):
  model=HorizonUnifiedTransformer(replace(self.config,experts=3,attention_experts=1,qk_norm=1));segments=fixture()
  model.joint_loss(segments,{2:torch.zeros(1,4,4)},visual_mode='editing')['flow_loss'].backward()
  self.assertGreater(float(model.text.weight.grad.norm()),0)
  model.zero_grad(set_to_none=True);segments[0].values.requires_grad_()
  model.joint_loss(segments,{2:torch.zeros(1,4,4)},visual_mode='editing')['text_loss'].backward()
  self.assertGreater(float(segments[0].values.grad.norm()),0)
 def test_exact_full_expert_roundtrip(self):
  model=HorizonUnifiedTransformer(replace(self.config,experts=3,attention_experts=1,qk_norm=1)).eval();segments=fixture()
  with tempfile.TemporaryDirectory() as tmp:
   model.save_checkpoint(Path(tmp)/'core');loaded=HorizonUnifiedTransformer.load_checkpoint(Path(tmp)/'core')
   torch.testing.assert_close(model(segments,visual_mode='editing')['flow'],loaded(segments,visual_mode='editing')['flow'],rtol=0,atol=0)
 def test_invalid_expert_configuration(self):
  with self.assertRaises(ValueError):replace(self.config,attention_experts=1)
  with self.assertRaises(ValueError):replace(self.config,qk_norm=2)
