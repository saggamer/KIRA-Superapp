"""Explicit GPU test; run separately from the ordinary CPU/unit test suite."""
import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from orchestrator_v15.moe import SparseUpcycledFFN


class ToyFFN(nn.Module):
    def __init__(self):
        super().__init__()
        for name,inp,out in [('gate_proj',64,128),('up_proj',64,128),('down_proj',128,64)]:
            self[name]=nn.QuantizedLinear.from_linear(nn.Linear(inp,out,bias=False))

    def __call__(self,x):
        return self.down_proj(nn.gelu_approx(self.gate_proj(x))*self.up_proj(x))


class SparseGPUChecks(unittest.TestCase):
    def test_bf16_parity_and_frozen_quantized_weights(self):
        mx.random.seed(15)
        base=ToyFFN();moe=SparseUpcycledFFN(base,8,4)
        x=mx.random.normal((1,5,64)).astype(mx.bfloat16)
        self.assertEqual(float(mx.max(mx.abs(moe(x)-base(x))).item()),0)
        names=[k for k,_ in tree_flatten(moe.trainable_parameters())]
        self.assertTrue(all(k.endswith(('lora_a','lora_b','router.weight')) for k in names))

    def test_only_selected_experts_receive_adapter_gradients(self):
        base=ToyFFN();moe=SparseUpcycledFFN(base,8,4)
        weights=mx.zeros((8,64));weights[6]=mx.ones((64,));weights[7]=mx.ones((64,))*2
        moe.router.weight=weights
        x=mx.ones((1,3,64))
        loss,grads=nn.value_and_grad(moe,lambda m:mx.mean(m(x)**2))(moe)
        mx.eval(loss,grads)
        b=grads['down_proj']['lora_b']
        self.assertEqual(float(mx.sum(mx.abs(b[:6])).item()),0)
        self.assertGreater(float(mx.sum(mx.abs(b[6:])).item()),0)


if __name__=='__main__':unittest.main()
