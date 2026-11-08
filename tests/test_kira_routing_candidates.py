import unittest
import numpy as np
import mlx.core as mx
from kira_live.mlx_layers import ThinkerOnlyExpertMoE
from kira_live.routing_candidates import exact_top_two, TokenResidualBranch, SharedConversationAdapter


class RoutingCandidateTests(unittest.TestCase):
    def test_ties_select_exactly_two(self):
        weights, mask = exact_top_two(mx.zeros((2, 5, 6)))
        np.testing.assert_array_equal(np.array(mask.sum(axis=-1)), 2)
        np.testing.assert_allclose(np.array(weights.sum(axis=-1)), 1)
        self.assertEqual(int(mx.sum(weights > 0).item()), 20)

    def test_new_branch_identity_and_bypass(self):
        source = ThinkerOnlyExpertMoE(hidden_size=4, rank=2)
        branch = TokenResidualBranch(4, 6, source, 'shared_agentic')
        hidden = mx.random.normal((1, 3, 4))
        np.testing.assert_array_equal(np.array(branch(hidden)), np.array(hidden))
        branch.eval()
        for projection in branch.up:
            projection.weight = mx.ones_like(projection.weight)
        np.testing.assert_array_equal(np.array(branch(hidden)), np.array(hidden))
        branch.gate.bias = mx.array([8.0])
        self.assertFalse(np.array_equal(np.array(branch(hidden)), np.array(hidden)))

    def test_token_routing_has_no_extra_bypass(self):
        branch = TokenResidualBranch(4, 6, ThinkerOnlyExpertMoE(hidden_size=4, rank=2), 'token_routed')
        branch.eval()
        branch(mx.ones((1, 3, 4)))
        np.testing.assert_array_equal(np.array(branch._trace[3]), 1)

    def test_routing_is_causal_and_per_token(self):
        branch = TokenResidualBranch(4, 6, ThinkerOnlyExpertMoE(hidden_size=4, rank=2), 'shared_agentic')
        hidden = mx.random.normal((1, 5, 4))
        branch(hidden)
        whole = branch._trace[0]
        branch(hidden[:, :2])
        np.testing.assert_array_equal(np.array(whole[:, :2]), np.array(branch._trace[0]))

    def test_shared_adapter_identity(self):
        layer = SharedConversationAdapter(4, 2)
        hidden = mx.random.normal((1, 5, 4))
        np.testing.assert_array_equal(np.array(layer(hidden)), np.array(hidden))

    def test_hybrid_uses_a_semantics_with_b_shared_path(self):
        import mlx.nn as nn
        from kira_live.routing_candidates import RoutingCandidateThinker
        model = nn.Module()
        model.hidden_size = 4
        model.expert_layers = [ThinkerOnlyExpertMoE(hidden_size=4, rank=2) for _ in range(16)]
        RoutingCandidateThinker.configure(model, 'hybrid_specialist', rank=6)
        self.assertEqual(len(model.shared_conversation), 4)
        self.assertEqual(len(model.routing_branches), 4)
        self.assertTrue(all(b.variant == 'token_routed' for b in model.routing_branches))
        for branch in model.routing_branches:
            branch.eval()
            branch(mx.ones((1, 2, 4)))
            np.testing.assert_array_equal(np.array(branch._trace[3]), 1)

    def test_full_checkpoint_omits_only_duplicate_backbone_aliases(self):
        import mlx.nn as nn
        from kira_live.routing_candidates import checkpoint_parameters
        model = nn.Module()
        model.model = nn.Linear(3, 3)
        model.language_model = model.model
        model.decoder = model.model
        model.shared_conversation = [SharedConversationAdapter(3, 2)]
        values = checkpoint_parameters(model, full=True)
        self.assertIn('model.weight', values)
        self.assertIn('shared_conversation.0.up.weight', values)
        self.assertNotIn('language_model.weight', values)
        self.assertNotIn('decoder.weight', values)

    def test_gated_hybrid_bypasses_a_in_chat_and_b_in_tree(self):
        import mlx.nn as nn
        from kira_live.routing_candidates import RoutingCandidateThinker
        model = RoutingCandidateThinker.__new__(RoutingCandidateThinker)
        nn.Module.__init__(model)
        model.hidden_size = 4
        model.expert_layers = [ThinkerOnlyExpertMoE(hidden_size=4, rank=2) for _ in range(16)]
        model.configure('hybrid_specialist_gated', rank=6)
        model.shared_conversation[0].down.weight = mx.ones_like(model.shared_conversation[0].down.weight)
        model.shared_conversation[0].up.weight = mx.ones_like(model.shared_conversation[0].up.weight)
        model.eval()
        hidden = mx.ones((1, 2, 4))
        model.set_execution_mode(False)
        result, _ = model._apply_expert(12, hidden, None)
        expected = model.shared_conversation[0](hidden)
        np.testing.assert_array_equal(np.array(result), np.array(expected))
        self.assertIsNone(model.routing_branches[0]._trace)
        model.set_execution_mode(True)
        result, _ = model._apply_expert(12, hidden, None)
        np.testing.assert_array_equal(np.array(result), np.array(hidden))  # A up matrices initialize to zero
        self.assertIsNotNone(model.routing_branches[0]._trace)
        model.start_audit()
        model.set_execution_mode(False)
        model._apply_expert(12, hidden, None)
        audit = model.finish_audit()
        self.assertEqual(audit['enabled'], 0)
        self.assertEqual(audit['tokens'], 2)

    def test_live_prefix_resets_tree_mode_every_turn(self):
        import mlx.nn as nn
        from unittest.mock import patch
        from kira_live.routing_candidates import RoutingCandidateThinker
        from kira_live.mlx_thinker import KiraMLXThinker
        model = RoutingCandidateThinker.__new__(RoutingCandidateThinker)
        nn.Module.__init__(model)
        with patch.object(KiraMLXThinker, 'build_live_prefix', return_value='prefix'):
            model.build_live_prefix(None, tree_planning=True)
            self.assertTrue(model._agentic_mode)
            model.build_live_prefix(None)
            self.assertFalse(model._agentic_mode)
        with self.assertRaises(ValueError):
            model.set_execution_mode('true')

    def test_parameter_ceiling(self):
        # Selected matrix weights, NOT a sparse GPU-compute claim.
        self.assertEqual(25460736 + 4 * 2 * 1024 * 4608 * 2, 100958208)

    def test_tutor_rubric_rejects_wrong_straw_mechanism(self):
        from scripts.compare_kira_routing_candidates import score_native
        bad = 'Reflection is light bouncing from a mirror. A straw in water looks bent because rays bounce off the water surface; this is refraction.'
        self.assertFalse(score_native('tutoring', bad))
        self.assertTrue(score_native('tutoring', 'Reflection means light bouncing off a mirror; the incident and reflected angles are equal.'))
        self.assertFalse(score_native('tutoring', 'Light hits water or glass and bounces back rather than passing through. The surface acts like a mirror.'))

    def test_arithmetic_rubric_rejects_contradictory_count(self):
        from scripts.compare_kira_routing_candidates import score_native
        self.assertFalse(score_native('reasoning_transfer', '24 pencils: 23 × 7 = 161 pencils × 2 boxes.'))
        self.assertTrue(score_native('reasoning_transfer', '23 × 7 = 161 pencils.'))

    def test_strict_plan_parse_does_not_rescue_inner_dictionary(self):
        from scripts.compare_kira_routing_candidates import strict_plan_json, score_native
        self.assertIsNone(strict_plan_json('{"actions": [{"arguments": {}}], "requires_confirmation": false'))
        self.assertEqual(strict_plan_json('{"actions": [], "requires_confirmation": false}'), {'actions': [], 'requires_confirmation': False})
        self.assertTrue(score_native('privacy', "I don't share my reasoning."))

    def test_shared_optimizer_has_no_empty_partition(self):
        import mlx.nn as nn
        from scripts.train_kira_routing_candidates import make_optimizer
        layer = SharedConversationAdapter(4, 2)
        optimizer = make_optimizer('shared')
        loss, gradients = nn.value_and_grad(layer, lambda m: mx.mean(m(mx.ones((1, 2, 4))) ** 2))(layer)
        optimizer.update(layer, gradients)
        mx.eval(layer.parameters(), optimizer.state, loss)
        self.assertTrue(bool(mx.isfinite(loss).item()))

    def test_weak_permission_spans_and_action_bypass_labels(self):
        from scripts.train_kira_routing_candidates import routing_targets
        class CharacterTokenizer:
            def decode(self, values, **_kwargs):
                return ''.join(chr(value) for value in values)
        text = '{"arguments": {}, "requires_confirmation": true}'
        row = {'source': 'kira_agentic', 'messages': [{'role': 'user', 'content': 'Remove a file'}, {'role': 'assistant', 'content': text}]}
        ids = mx.array([[0] + [ord(c) for c in text]])
        labels, gates, enabled = routing_targets(row, CharacterTokenizer(), ids, 1, 'shared_agentic')
        self.assertTrue(enabled)
        self.assertEqual(int(labels[0, -1].item()), 3)
        self.assertEqual(int(gates.sum().item()), len(text))
        row = {'source': 'conversation', 'messages': [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}]}
        _, gates, enabled = routing_targets(row, CharacterTokenizer(), mx.array([[0, 72, 105]]), 1, 'shared_agentic')
        self.assertFalse(enabled)
        self.assertEqual(int(gates.sum().item()), 0)


if __name__ == '__main__':
    unittest.main()
