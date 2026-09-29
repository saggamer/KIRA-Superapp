import unittest
import tempfile
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from kira_live.expert_surgery import widen_expert_weights, expert_checkpoint_rank


class ExpertSurgeryTests(unittest.TestCase):
    def fixture(self):
        weights = {'router': np.ones((4, 4), dtype=np.float32)}
        for layer in range(16):
            for expert in range(10):
                weights[f'expert_layers.{layer}.down.{expert}.weight'] = np.ones((2, 4), dtype=np.float32)
                weights[f'expert_layers.{layer}.up.{expert}.weight'] = np.ones((4, 2), dtype=np.float32)
        return weights

    def test_preserves_old_channels_and_zero_initializes_new_outputs(self):
        old = self.fixture()
        new = widen_expert_weights(old, 8)
        self.assertIs(new['router'], old['router'])
        for layer in range(16):
            for expert in range(10):
                key = f'expert_layers.{layer}'
                np.testing.assert_array_equal(new[f'{key}.down.{expert}.weight'][:2], old[f'{key}.down.{expert}.weight'])
                np.testing.assert_array_equal(new[f'{key}.up.{expert}.weight'][:, :2], old[f'{key}.up.{expert}.weight'])
                self.assertFalse(new[f'{key}.up.{expert}.weight'][:, 2:].any())
        self.assertEqual(old['expert_layers.0.down.0.weight'].shape, (2, 4))

    def test_rejects_shrink_and_missing_expert(self):
        with self.assertRaises(ValueError):
            widen_expert_weights(self.fixture(), 2)
        weights = self.fixture()
        del weights['expert_layers.0.up.0.weight']
        with self.assertRaises(ValueError):
            widen_expert_weights(weights, 8)

    def test_budget_is_about_100m_selected_parameters(self):
        active = 16 * (6 * 2 * 1024 * 512 + 18 * 1024)
        self.assertEqual(active, 100958208)

    def test_header_rank_checks_all_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experts.safetensors'
            weights = self.fixture()
            save_file(weights, str(path))
            self.assertEqual(expert_checkpoint_rank(path), 2)
            weights['expert_layers.0.up.0.weight'] = np.ones((4, 3), dtype=np.float32)
            save_file(weights, str(path))
            with self.assertRaises(ValueError):
                expert_checkpoint_rank(path)
