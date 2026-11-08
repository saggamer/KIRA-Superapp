import unittest
import torch
from orchestrator_v15.horizon_instruction_grounder import InstructionGrounder,paired_instruction_loss
class InstructionGrounderTests(unittest.TestCase):
 def test_explicit_instruction_and_image_gradients(self):
  torch.manual_seed(8);head=InstructionGrounder(12,16,16)
  image=torch.randn(2,16,12);rgb=torch.rand(2,16,3);text=torch.randn(2,5,16);mask=torch.ones(2,5,dtype=torch.bool)
  labels=torch.zeros(2,16,1);labels[0,:4]=1;labels[1,-4:]=1
  logits=head(image,rgb,text,mask,4,4)
  loss,_=paired_instruction_loss(logits,labels,torch.tensor([1,0]),torch.ones(2,dtype=torch.bool));loss.backward()
  self.assertGreater(head.text_projection.weight.grad.abs().sum().item(),0)
  self.assertGreater(head.image_projection.weight.grad.abs().sum().item(),0)
  other=head(image,rgb,text.flip(0),mask,4,4)
  self.assertFalse(torch.equal(logits,other))
 def test_padding_is_excluded_and_empty_rejected(self):
  head=InstructionGrounder(12,16,16).eval();image=torch.randn(1,16,12);rgb=torch.rand(1,16,3);text=torch.randn(1,5,16);mask=torch.tensor([[1,1,1,0,0]],dtype=torch.bool)
  expected=head(image,rgb,text,mask,4,4);text[:,3:]=1000
  self.assertTrue(torch.allclose(expected,head(image,rgb,text,mask,4,4),atol=1e-6))
  with self.assertRaises(ValueError):head(image,rgb,text,torch.zeros_like(mask),4,4)
 def test_ignored_instruction_violates_margin_and_pair_must_be_verified(self):
  logits=torch.zeros(2,16,1,requires_grad=True);labels=torch.zeros_like(logits);labels[0,:4]=1;labels[1,-4:]=1
  loss,details=paired_instruction_loss(logits,labels,torch.tensor([1,0]),torch.ones(2,dtype=torch.bool))
  self.assertAlmostEqual(details['ranking_loss'],.25,places=6)
  with self.assertRaises(ValueError):paired_instruction_loss(logits,labels,torch.tensor([1,0]))
  with self.assertRaises(ValueError):paired_instruction_loss(logits,labels,torch.tensor([0,1]),torch.ones(2,dtype=torch.bool))
