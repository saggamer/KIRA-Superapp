import tempfile
import unittest
from pathlib import Path
import torch
from orchestrator_v15.horizon_context import ContextSpan
from orchestrator_v15.horizon_unified_torch import HorizonUnifiedTransformer,UnifiedConfig,UnifiedSegment


def fixture(sample=0):
    xy=torch.tensor([[[0.25,0.25],[0.75,0.25],[0.25,0.75],[0.75,0.75]]])
    return [UnifiedSegment(ContextSpan('clean_latent',4,0,sample,'source'),torch.randn(1,4,4),xy=xy),
            UnifiedSegment(ContextSpan('text',4,1,sample),torch.tensor([[1,2,3,4]]),supervise_text=True),
            UnifiedSegment(ContextSpan('noisy_latent',4,2,sample,'target'),torch.randn(1,4,4),xy=xy,timestep=torch.tensor([0.5]))]


class UnifiedCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model=HorizonUnifiedTransformer(UnifiedConfig(16,width=16,heads=4,layers=2,latent_width=4,vision_width=8,max_tokens=64))
        self.segments=fixture()

    def test_language_conditions_image_through_shared_layers(self):
        before=self.model(self.segments)['flow'].detach()
        self.model.joint_loss(self.segments,{2:torch.zeros(1,4,4)})['flow_loss'].backward()
        self.assertGreater(float(self.model.text.weight.grad.norm()),0)
        for block in self.model.blocks:self.assertGreater(float(block.qkv.weight.grad.norm()),0)
        self.segments[1].values=torch.tensor([[5,6,7,8]])
        self.assertFalse(torch.equal(before,self.model(self.segments)['flow']))

    def test_source_image_conditions_language_through_same_layers(self):
        self.segments[0].values.requires_grad_()
        self.model.joint_loss(self.segments,{2:torch.zeros(1,4,4)})['text_loss'].backward()
        self.assertGreater(float(self.segments[0].values.grad.norm()),0)
        self.assertGreater(float(self.model.latent_in.weight.grad.norm()),0)

    def test_future_text_and_noisy_target_cannot_change_earlier_text(self):
        before=self.model(self.segments)['text_logits'].detach()
        self.segments[1].values[0,-1]=9
        self.segments[2].values+=100
        after=self.model(self.segments)['text_logits']
        torch.testing.assert_close(before[:,:-1],after[:,:-1],rtol=0,atol=0)

    def test_packed_samples_do_not_communicate(self):
        segments=self.segments+fixture(sample=1)
        before=self.model(segments)['hidden'][:,12:].detach()
        segments[0].values+=200
        segments[1].values[:]=7
        segments[2].values-=100
        after=self.model(segments)['hidden'][:,12:]
        torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_rejects_indirect_clean_target_leakage(self):
        # Otherwise clean target -> intervening text -> noisy target leaks at layer 2.
        self.segments[0].span=ContextSpan('clean_latent',4,0,0,'target')
        with self.assertRaisesRegex(ValueError,'No clean/semantic target'):
            self.model(self.segments)

    def test_validates_coordinates_timestep_and_loss_targets(self):
        self.segments[2].timestep=torch.tensor([1.1])
        with self.assertRaisesRegex(ValueError,'timestep'):self.model(self.segments)
        self.segments[2].timestep=torch.tensor([0.5]);self.segments[0].xy=None
        with self.assertRaisesRegex(ValueError,'coordinates'):self.model(self.segments)
        self.segments=fixture()
        with self.assertRaisesRegex(ValueError,'exactly one'):self.model.joint_loss(self.segments,{})

    def test_roundtrip_retains_shared_head_and_predictions(self):
        before=self.model(self.segments)['flow'].detach()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'core';self.model.save_checkpoint(path)
            restored=HorizonUnifiedTransformer.load_checkpoint(path)
            self.assertIs(restored.text.weight,restored.language_head.weight)
            torch.testing.assert_close(before,restored(self.segments)['flow'],rtol=0,atol=0)

    def test_joint_optimization_software_fixture(self):
        # Synthetic tensors prove optimizability, not image quality or grounding.
        target=torch.randn(1,4,4)
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=0.01)
        first=float(self.model.joint_loss(self.segments,{2:target})['loss'].detach())
        for _ in range(40):
            optimizer.zero_grad();loss=self.model.joint_loss(self.segments,{2:target})['loss'];loss.backward();optimizer.step()
        last=float(self.model.joint_loss(self.segments,{2:target})['loss'].detach())
        self.assertLess(last,first*0.2)


if __name__=='__main__':unittest.main()
