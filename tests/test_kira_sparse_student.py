import unittest
import json
from pathlib import Path

from kira_live.sparse_student_architecture import SparseStudentSpec


class SparseStudentArchitectureTests(unittest.TestCase):
    def test_experimental_variant_is_not_marked_usable_without_a_full_checkpoint(self):
        manifest = json.loads(
            (Path(__file__).resolve().parents[1] / "kira_live/experimental_live_1_manifest.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "Experimental Live 1")
        self.assertFalse(manifest["release_gates"]["full_checkpoint_exists"])
        self.assertNotEqual(manifest["status"], "ready")

    def test_total_and_active_budgets(self):
        spec = SparseStudentSpec()
        budget = spec.validate_budget()
        self.assertEqual(spec.name, "Experimental Live 1")
        self.assertEqual(budget["total_thinker"], 2_522_091_574)
        self.assertEqual(budget["active_thinker"], 478_351_414)
        self.assertEqual(budget["emotion_router"], 3_840)
        self.assertEqual(spec.dense_prefix_layers, 4)
        self.assertEqual(sum(spec.expert_families.values()), 24)
        self.assertEqual(spec.active_experts_per_token, 2)

    def test_cuda_training_graph_exact_parameter_count_without_allocating_weights(self):
        import torch
        from kira_live.experimental_live_1_torch import ExperimentalLive1Torch

        with torch.device("meta"):
            model = ExperimentalLive1Torch.from_spec()
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            SparseStudentSpec().validate_budget()["total_thinker"],
        )

    def test_dense_donor_is_not_mislabelled_sparse(self):
        # Actual Qwen3.5-2B checkpoint metadata: embeddings and attention alone
        # exceed the proposed active budget, before any dense FFN is executed.
        donor_embeddings = 508_559_360
        donor_attention = 467_195_712
        self.assertGreater(donor_embeddings + donor_attention, 500_000_000)

    def test_budget_rejects_more_active_experts(self):
        with self.assertRaisesRegex(ValueError, "active parameters"):
            SparseStudentSpec(active_experts_per_token=3).validate_budget()


if __name__ == "__main__":
    unittest.main()
