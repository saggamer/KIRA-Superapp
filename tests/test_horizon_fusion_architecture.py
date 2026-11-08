import unittest
from orchestrator_v15.horizon_fusion import HorizonFusionArchitecture


class HorizonFusionArchitectureTests(unittest.TestCase):
    def test_nonvisual_routes_bypass_fusion(self):
        architecture = HorizonFusionArchitecture()
        for mode in ('conversation', 'tools', 'ppt', 'pdf'):
            self.assertEqual(architecture.route(mode)['island'], 'existing_controller')
            self.assertIsNone(architecture.route(mode)['bridge'])
        for mode in ('editing', 'creativity'):
            self.assertEqual(architecture.route(mode)['bridge'], mode)
        with self.assertRaises(ValueError):
            architecture.route('unknown')

    def test_future_compact_budget_is_explicitly_a_design_estimate(self):
        record = HorizonFusionArchitecture().to_dict()
        self.assertEqual(record['name'], 'Horizon 1')
        self.assertFalse(record['fully_unified_bagel_model'])
        self.assertFalse(record['controller_changed'])
        self.assertEqual(record['architecture_revision'],2)
        context=record['relationship_context']
        self.assertIn('untrained',context['neural_status'])
        self.assertLess(context['eight_block_design_allowance'],
                        record['compact_target']['components']['residual_experts_and_bridges_allowance'])
        budget = record['compact_target']
        self.assertLessEqual(budget['total_upper_bound'], budget['ceiling'])
        self.assertIn('not trained', budget['status'])


if __name__ == '__main__':
    unittest.main()
