from __future__ import annotations

"""Experimental KIRA Thinker with an auxiliary action-permission head.

This is deliberately separate from the deployed Thinker. The head trains from
the last *prompt* state, so it cannot read the reference answer, and its
gradient reaches the KIRA PLE/MoE adapters through the frozen Qwen donor.
"""

import mlx.core as mx
import mlx.nn as nn

from .mlx_thinker import KiraMLXThinker


class KiraPrecisionThinker(KiraMLXThinker):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        width = max(64, self.hidden_size // 8)
        self.permission_down = nn.Linear(self.hidden_size, width, bias=False)
        self.permission_out = nn.Linear(width, 2, bias=True)
        # The auxiliary head starts uninformative instead of injecting a
        # random preference for either permission class.
        self.permission_out.weight = mx.zeros_like(self.permission_out.weight)
        self.permission_out.bias = mx.zeros_like(self.permission_out.bias)

    def permission_logits(self, hidden: mx.array, prompt_end: int) -> mx.array:
        if not 0 < prompt_end <= hidden.shape[1]:
            raise ValueError("Permission classification must use a prompt token.")
        state = hidden[:, prompt_end - 1, :].astype(mx.float32)
        return self.permission_out(nn.silu(self.permission_down(state)))
