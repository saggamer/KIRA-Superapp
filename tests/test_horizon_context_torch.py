import io
import unittest
import torch
from orchestrator_v15.horizon_context_torch import JointContextResidual,RelationBindingHead,relationship_ranking_loss
from tests.test_horizon_context import layout


class JointContextKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.block=JointContextResidual(16,heads=4,rank=4)
        self.hidden=torch.randn(1,15,16)

    def test_identity_zero_initialization_and_parameter_accounting(self):
        self.assertIs(self.block(self.hidden,layout(),mode=None),self.hidden)
        for mode in ('editing','creativity'):
            self.assertTrue(torch.equal(self.block(self.hidden,layout(),mode=mode),self.hidden))
        self.assertEqual(self.block.trainable_parameters(),36*16*4+6)
        self.assertFalse(self.block.base_qkv.weight.requires_grad)

    def test_training_one_visual_mode_preserves_other_mode_and_native_path(self):
        for mode in ('editing','creativity'):
            for projection in self.block.up[mode].values():
                torch.nn.init.normal_(projection.weight,std=0.1)
        before=self.block(self.hidden,layout(),mode='creativity').detach().clone()
        optimizer=torch.optim.AdamW(self.block.parameters(),lr=0.02)
        for _ in range(4):
            optimizer.zero_grad()
            output=self.block(self.hidden,layout(),mode='editing')
            (output-(self.hidden+0.2)).square().mean().backward()
            optimizer.step()
        self.assertTrue(torch.equal(before,self.block(self.hidden,layout(),mode='creativity')))
        self.assertIs(self.block(self.hidden,layout(),mode=None),self.hidden)
        self.assertIsNone(self.block.base_qkv.weight.grad)
        self.assertGreater(float(self.block.up['editing']['relations'].weight.grad.norm()),0)

    def test_no_noisy_target_or_other_sample_leak_into_later_text(self):
        for projection in self.block.up['creativity'].values():
            torch.nn.init.normal_(projection.weight,std=0.5)
        before=self.block(self.hidden,layout(),mode='creativity')
        changed=self.hidden.clone()
        changed[:,8:10]+=100
        changed[:,14:15]-=200
        after=self.block(changed,layout(),mode='creativity')
        self.assertTrue(torch.equal(before[:,12:14],after[:,12:14]))

    def test_exact_state_reload_and_explicit_layout_validation(self):
        data=io.BytesIO()
        torch.save(self.block.state_dict(),data)
        data.seek(0)
        restored=JointContextResidual(16,heads=4,rank=4)
        restored.load_state_dict(torch.load(data,weights_only=True))
        self.assertTrue(torch.equal(self.block(self.hidden,layout(),mode='editing'),restored(self.hidden,layout(),mode='editing')))
        with self.assertRaises(ValueError):
            self.block(self.hidden[:,:3],layout(),mode='editing')


class RelationHeadTests(unittest.TestCase):
    def test_swapped_roles_can_be_optimized_as_counterfactuals(self):
        torch.manual_seed(33)
        head=RelationBindingHead(8,4,rank=4)
        subject,target=torch.randn(4,8),torch.randn(4,8)
        predicate=torch.tensor([0,1,2,3])
        known=torch.ones(4,dtype=torch.bool)
        optimizer=torch.optim.Adam(head.parameters(),lr=0.03)
        first=None
        for _ in range(24):
            optimizer.zero_grad()
            positive=head(subject,target,predicate)
            negative=head(target,subject,predicate)
            loss=relationship_ranking_loss(positive,negative,known)
            first=float(loss.detach()) if first is None else first
            loss.backward(); optimizer.step()
        self.assertLess(float(loss.detach()),first*0.5)
        self.assertTrue((head(subject,target,predicate)>head(target,subject,predicate)).all())

    def test_unknown_labels_do_not_train_a_false_negative(self):
        positive=torch.tensor([1.0],requires_grad=True)
        negative=torch.tensor([2.0],requires_grad=True)
        loss=relationship_ranking_loss(positive,negative,torch.tensor([False]))
        loss.backward()
        self.assertEqual(float(loss.detach()),0)
        self.assertEqual(float(positive.grad),0)
        self.assertEqual(float(relationship_ranking_loss(torch.tensor([float('nan')]),negative,
                                                       torch.tensor([False])).detach()),0)
        with self.assertRaises(ValueError):
            relationship_ranking_loss(torch.tensor([float('nan')]),negative,torch.tensor([True]))
