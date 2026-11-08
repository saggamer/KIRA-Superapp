import unittest
from pathlib import Path
import torch
from orchestrator_v15.horizon_edit_training import edit_category, select_balanced, verify_split_identities, CATEGORIES
from orchestrator_v15.horizon_edit_loss import balanced_edit_flow_loss
from scripts.horizon_big_train_patches import cpu_cache_patch, complete_instruction_patch
from scripts.horizon_broad_train_patches import broad_edit_patch

class BroadEditTests(unittest.TestCase):
 def test_categories_and_complete_prompt(self):
  for text,expected in [('Remove the chair','removal'),('Add a cloud','addition'),('Replace the chair with a bicycle','replacement'),('Make the car red','color'),('The sign should say HELLO','text'),('Move the chair left','spatial'),('Make the weather sunny','appearance')]:
   self.assertEqual(edit_category(text),expected)
 def test_balancing_deduplicates_and_requires_coverage(self):
  rows=[dict(record_id=c+str(i),category=c) for c in CATEGORIES for i in range(4)]
  picked=select_balanced(rows+rows,21,3)
  self.assertEqual(len({r['record_id'] for r in picked}),21)
  self.assertTrue(all(sum(r['category']==c for r in picked)==3 for c in CATEGORIES))
  with self.assertRaises(ValueError):select_balanced(rows[:-4],21,1)
 def test_split_group_and_canonical_pixels(self):
  a={'group_id':'one','pixel_hashes':['pixels1']};b={'group_id':'two','pixel_hashes':['pixels2']}
  self.assertTrue(verify_split_identities({'train':[a],'test':[b]}))
  for conflicting in ({**b,'group_id':'one'},{**b,'pixel_hashes':['pixels1']}):
   with self.assertRaises(ValueError):verify_split_identities({'train':[a],'test':[conflicting]})
 def test_scarce_development_category_is_retained(self):
  rows=[dict(record_id=c+str(i),category=c) for c in CATEGORIES for i in range(1 if c=='spatial' else 6)]
  chosen=select_balanced(rows,14,1)
  self.assertEqual(len(chosen),14)
  self.assertEqual(sum(r['category']=='spatial' for r in chosen),1)
  self.assertEqual({r['category'] for r in chosen},set(CATEGORIES))
  with self.assertRaises(ValueError):select_balanced(rows,14,2)
 def test_loss_identity_global_and_gradient(self):
  for source in (torch.zeros(1,2,2,2),torch.ones(1,2,2,2)):
   prediction=torch.full_like(source,2,requires_grad=True);target=torch.zeros_like(source)
   loss=balanced_edit_flow_loss(prediction,target,source,target,torch.ones(1,1,1,1))
   self.assertAlmostEqual(loss.item(),4);loss.backward();self.assertGreater(prediction.grad.norm().item(),0)
 def test_preservation_weight_and_no_target_grad(self):
  pred=torch.tensor([[[[1.,3.]]]],requires_grad=True)
  target=torch.zeros_like(pred,requires_grad=True);source=torch.tensor([[[[0.,1.]]]],requires_grad=True)
  loss=balanced_edit_flow_loss(pred,target,source,target,torch.ones(1,1,1,1))
  self.assertAlmostEqual(loss.item(),.6+3.6,places=5);loss.backward()
  self.assertIsNone(target.grad);self.assertIsNone(source.grad)
 def test_rejects_nonfinite_and_misalignment(self):
  value=torch.zeros(1,1,2,2)
  with self.assertRaises(ValueError):balanced_edit_flow_loss(value,value,value[:,:,:1],value,torch.ones(1))
  with self.assertRaises(ValueError):balanced_edit_flow_loss(value+float('nan'),value,value,value,torch.ones(1))
 def test_actual_pinned_training_patch_preserves_guards(self):
  original=Path('training_runs/horizon_fusion_20261017/visual_results/train_dreambooth_lora_flux2_klein_img2img.py').read_text()
  patched=broad_edit_patch(complete_instruction_patch(cpu_cache_patch(original,'editing')))
  self.assertIn('BROAD_PARENT_LOADED',patched)
  self.assertIn('balanced_edit_flow_loss(model_pred, target, cond_model_input',patched)
  self.assertIn('no instruction was truncated',patched)
  self.assertIn('HORIZON_UPDATE',patched)
  self.assertIn('latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist.mode().detach().cpu())',patched)
  self.assertEqual(patched.count('global_step += 1'),1)
  with self.assertRaises(ValueError):broad_edit_patch(patched)
  with self.assertRaises(ValueError):broad_edit_patch(original.replace('    transformer.add_adapter(transformer_lora_config)','    different()'))
