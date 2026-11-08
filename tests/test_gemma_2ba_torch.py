"""Sparse routing and full-matrix expert training on a tiny CPU stand-in."""
import unittest

import torch
from torch import nn

from orchestrator_v15.gemma_2ba_torch import ChannelGatedFullFFN, attach_gemma_2ba


class TinyFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.gate_proj(x)) * self.up_proj(x))


class Gemma2BATorchTests(unittest.TestCase):
    def test_attach_uses_transformers_gemma4_decoder_path(self):
        class TinyBudget:
            routed_layers = (29, 41)
            hidden_size = 4
            experts_per_layer = 3
            top_k = 2
            channels = ('browser', 'computer', 'code', 'knowledge', 'reasoning', 'general')

            def validate(self):
                pass

        model = nn.Module()
        model.model = nn.Module()
        model.model.language_model = nn.Module()
        model.model.language_model.layers = nn.ModuleList()
        for _ in range(42):
            layer = nn.Module()
            layer.mlp = TinyFFN()
            model.model.language_model.layers.append(layer)
        modules = attach_gemma_2ba(model, TinyBudget())
        self.assertEqual(set(modules), {29, 41})
        self.assertIs(model.model.language_model.layers[29].mlp, modules[29])
        self.assertFalse(next(modules[29].shared.parameters()).requires_grad)

    def test_exact_donor_parity_before_expert_training(self):
        torch.manual_seed(4)
        shared = TinyFFN()
        x = torch.randn(2, 3, 4)
        expected = shared(x).detach().clone()
        routed = ChannelGatedFullFFN(shared, hidden_size=4)
        routed.eval()
        self.assertTrue(torch.allclose(routed(x), expected, atol=1e-6))

    def test_full_expert_matrix_updates_without_changing_shared_donor(self):
        torch.manual_seed(5)
        routed = ChannelGatedFullFFN(TinyFFN(), hidden_size=4)
        donor = routed.shared.gate_proj.weight.detach().clone()
        expert_before = routed.experts[2].gate_proj.weight.detach().clone()
        routed.train_expert(2)
        self.assertTrue(routed.experts[2].gate_proj.weight.requires_grad)
        self.assertFalse(routed.experts[0].gate_proj.weight.requires_grad)
        x = torch.randn(2, 3, 4)
        loss = routed(x).square().mean()
        loss.backward()
        optimizer = torch.optim.SGD((p for p in routed.parameters() if p.requires_grad), lr=0.5)
        optimizer.step()
        self.assertTrue(torch.equal(routed.shared.gate_proj.weight, donor))
        self.assertFalse(torch.equal(routed.experts[2].gate_proj.weight, expert_before))

    def test_channel_gate_trains_and_dispatches_two_experts(self):
        torch.manual_seed(6)
        routed = ChannelGatedFullFFN(TinyFFN(), hidden_size=4)
        routed.train_router()
        routed.set_channel('browser')
        x = torch.randn(2, 3, 4)
        y = routed(x)
        self.assertEqual(y.shape, x.shape)
        loss = y.square().mean() + routed.routing_loss()
        loss.backward()
        self.assertIsNotNone(routed.router.weight.grad)
        self.assertIsNotNone(routed.channel_bias.grad)
        self.assertEqual(sum(p.requires_grad for p in routed.experts[0].parameters()), 0)
        with self.assertRaises(ValueError):
            routed.set_channel('unknown')


if __name__ == '__main__':
    unittest.main()
