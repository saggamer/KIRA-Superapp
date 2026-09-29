import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from kira_live.expert_profiles import expert_layout, selected_parameter_budget, AGENTIC_INDICES, CONVERSATION_INDICES
from kira_live.expert_surgery import split_expert_weights, expert_checkpoint_config


class SplitExpertTests(unittest.TestCase):
    def test_capacity_and_disjoint_groups(self):
        budget = selected_parameter_budget()
        self.assertEqual(budget['agentic_matrices'], 50331648)
        self.assertEqual(budget['conversation_emotion_matrices'], 50331648)
        self.assertEqual(budget['total'], 100958208)
        self.assertFalse(set(AGENTIC_INDICES) & set(CONVERSATION_INDICES))
        self.assertEqual(set(AGENTIC_INDICES) | set(CONVERSATION_INDICES), set(range(10)))
        self.assertEqual(expert_layout(128)[0], (128,) * 10)

    def test_surgery_preserves_function_and_nonexperts(self):
        rng = np.random.default_rng(1)
        weights = {'ngram_layers.0.weight': np.ones((2, 2), dtype=np.float32)}
        for layer in range(16):
            for expert in range(10):
                weights[f'expert_layers.{layer}.down.{expert}.weight'] = rng.normal(size=(2, 4)).astype(np.float32)
                weights[f'expert_layers.{layer}.up.{expert}.weight'] = rng.normal(size=(4, 2)).astype(np.float32)
        new = split_expert_weights(weights)
        self.assertIs(new['ngram_layers.0.weight'], weights['ngram_layers.0.weight'])
        for expert in range(10):
            key = f'expert_layers.0'
            down, up = new[f'{key}.down.{expert}.weight'], new[f'{key}.up.{expert}.weight']
            self.assertEqual(down.shape[0], 768 if expert in AGENTIC_INDICES else 384)
            np.testing.assert_array_equal(down[:2], weights[f'{key}.down.{expert}.weight'])
            self.assertFalse(up[:, 2:].any())
            old_gain = 0.40 if expert in AGENTIC_INDICES else 0.25 if expert in (1, 8) else 0.35
            new_gain = 0.50 if expert in AGENTIC_INDICES else 0.25
            np.testing.assert_allclose(new_gain*up[:, :2], old_gain*weights[f'{key}.up.{expert}.weight'], rtol=1e-6, atol=1e-7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'base.safetensors'
            save_file(new, str(path))
            self.assertEqual(expert_checkpoint_config(path), {'expert_rank':384, 'expert_profile':'split_50_50'})

    def test_gradient_mask_preserves_existing_channels(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten
        from scripts.train_kira_live_thinker_v2 import mask_preserved_channels
        gradients = tree_unflatten([('expert_layers.12.down.2.weight', mx.ones((8, 4))),
                                    ('expert_layers.12.up.2.weight', mx.ones((4, 8)))])
        masked = dict(tree_flatten(mask_preserved_channels(gradients, 2)))
        np.testing.assert_array_equal(np.array(masked['expert_layers.12.down.2.weight'])[:2], 0)
        np.testing.assert_array_equal(np.array(masked['expert_layers.12.down.2.weight'])[2:], 1)
        np.testing.assert_array_equal(np.array(masked['expert_layers.12.up.2.weight'])[:, :2], 0)
        np.testing.assert_array_equal(np.array(masked['expert_layers.12.up.2.weight'])[:, 2:], 1)


if __name__ == '__main__':
    unittest.main()
