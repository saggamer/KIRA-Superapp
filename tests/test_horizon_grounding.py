import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from orchestrator_v15.horizon_grounding import annotation_mask,spatial_scene,mask_metrics
from orchestrator_v15.horizon_grounding_torch import install_grounding,source_grounding,region_loss,annotated_flow_loss,save_grounding,load_grounding
from scripts.horizon_grounding_train_patches import grounding_training_patch
from scripts.horizon_big_train_patches import cpu_cache_patch,complete_instruction_patch

class NativeToy(nn.Module):
 def __init__(self):
  super().__init__();self.x_embedder=nn.Linear(2,8);self.proj_out=nn.Linear(8,2)
 def forward(self,hidden_states,encoder_hidden_states,**kwargs):
  features=self.x_embedder(hidden_states)+encoder_hidden_states.mean(1,keepdim=True)
  return (self.proj_out(features),)

class GroundingTests(unittest.TestCase):
 def test_transparency_not_photograph_brightness(self):
  image=Image.new('RGBA',(20,20),(255,255,255,255));image.putpixel((2,3),(40,60,80,0))
  mask,evidence=annotation_mask(image,(20,20));value=np.asarray(mask)
  self.assertEqual(evidence['encoding'],'inverse_alpha');self.assertEqual(value[3,2],255);self.assertEqual(int((value>0).sum()),1)
  with self.assertRaises(ValueError):annotation_mask(Image.new('RGB',(20,20),(50,60,70)))
 def test_real_mask_previews_decode_as_inverse_alpha(self):
  for i in range(4):
   mask,evidence=annotation_mask(Image.open(f'training_runs/horizon1_grounding_20261101/preview-{i}-mask_img.png'))
   self.assertEqual(evidence['encoding'],'inverse_alpha');self.assertEqual(mask.size,(512,512));self.assertGreater(evidence['fraction'],0)
 def test_same_image_different_instructions_have_different_region_labels(self):
  rows=spatial_scene(18,'scene');self.assertEqual(len(rows),4)
  self.assertTrue(all(r['source'].tobytes()==rows[0]['source'].tobytes() for r in rows))
  self.assertEqual(len({r['mask'].tobytes() for r in rows}),4)
  self.assertEqual(len({r['prompt'] for r in rows}),4)
  for r in rows:
   changed=np.any(np.asarray(r['source'])!=np.asarray(r['target']),axis=-1)
   self.assertTrue(np.array_equal(changed,np.asarray(r['mask'])>0))
 def test_source_only_and_no_mask_leakage(self):
  model=NativeToy();head=install_grounding(model);head.output.weight.data.fill_(.1)
  source=torch.randn(1,2,2,2);text=torch.randn(1,3,8);ids=torch.zeros(1,4,4)
  pack=lambda x:x.flatten(2).transpose(1,2)
  logits=source_grounding(model,source,text,ids,ids,pack);self.assertEqual(logits.shape,(1,4,1))
  labels=torch.zeros_like(logits);labels[:,0]=1;loss=region_loss(logits,labels);loss.backward()
  self.assertGreater(head.output.weight.grad.abs().sum().item(),0)
  with self.assertRaises(ValueError):model(hidden_states=pack(source),encoder_hidden_states=text,horizon_ground_only=True,horizon_edit_region=labels)
  self.assertEqual(len(model.proj_out._forward_hooks),0)
 def test_zero_hint_embedding_preserves_function_and_explicit_hint_changes_output(self):
  model=NativeToy();x=torch.randn(1,8,2);text=torch.randn(1,3,8);expected=model(hidden_states=x,encoder_hidden_states=text)[0]
  head=install_grounding(model);hint=torch.ones(1,4,1)
  actual=model(hidden_states=x,encoder_hidden_states=text,horizon_edit_region=hint)[0]
  self.assertTrue(torch.equal(expected,actual));head.region_embedding.data.fill_(1)
  self.assertFalse(torch.equal(expected,model(hidden_states=x,encoder_hidden_states=text,horizon_edit_region=hint)[0]))
  self.assertEqual(len(model.x_embedder._forward_hooks),0)
 def test_mask_loss_handles_empty_identity_and_nonfinite(self):
  logits=torch.zeros(2,16,1,requires_grad=True);labels=torch.zeros_like(logits)
  value=region_loss(logits,labels);value.backward();self.assertTrue(torch.isfinite(logits.grad).all())
  with self.assertRaises(ValueError):region_loss(logits+float('nan'),labels)
 def test_annotation_flow_and_save_load(self):
  pred=torch.tensor([[[[1.,3.]]]],requires_grad=True);target=torch.zeros_like(pred);region=torch.tensor([[[[0.,1.]]]])
  self.assertAlmostEqual(annotated_flow_loss(pred,target,torch.ones(1),region).item(),4.2,places=5)
  with tempfile.TemporaryDirectory() as tmp:
   model=NativeToy();head=install_grounding(model);save_grounding(model,tmp)
   other=NativeToy();loaded=load_grounding(other,tmp)
   self.assertTrue(all(torch.equal(v,loaded.state_dict()[k]) for k,v in head.state_dict().items()))
 def test_location_metric_and_pinned_trainer(self):
  mask=np.zeros((8,8));mask[2:4,3:5]=1
  self.assertEqual(mask_metrics(mask,mask)['iou'],1)
  source=Path('training_runs/horizon_fusion_20261017/visual_results/train_dreambooth_lora_flux2_klein_img2img.py').read_text()
  patched=grounding_training_patch(complete_instruction_patch(cpu_cache_patch(source,'editing')))
  self.assertIn('source_grounding(transformer, cond_model_input.detach()',patched)
  self.assertIn('horizon_edit_region=hint_tokens',patched)
  self.assertIn('self.horizon_masks',patched)
  self.assertIn('"lr":0.001',patched)
  self.assertIn('no instruction was truncated',patched)
  with self.assertRaises(ValueError):grounding_training_patch(patched)
