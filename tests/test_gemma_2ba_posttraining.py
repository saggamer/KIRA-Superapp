"""Parameter and data filters for the Gemma 2B-active research candidate."""
import unittest

from orchestrator_v15.gemma_2ba_budget import Gemma2BABudget
from scripts.prepare_gemma_2ba_posttraining import EXTERNAL_EFFECT, SENSITIVE, temporal_rows


class Gemma2BATests(unittest.TestCase):
    def test_two_billion_routed_parameters_active_under_twelve_billion_total(self):
        budget = Gemma2BABudget()
        budget.validate()
        self.assertEqual(budget.parameters_per_expert, 78_643_200)
        self.assertEqual(budget.active_expert_parameters_per_token, 2_044_723_200)
        self.assertEqual(budget.total_logical_bound, 11_067_184_874)
        with self.assertRaises(ValueError):
            Gemma2BABudget(routed_layers=tuple(range(30, 42))).validate()
        with self.assertRaises(ValueError):
            Gemma2BABudget(experts_per_layer=4).validate()

    def test_desktop_source_filter_catches_credentials_and_external_effects(self):
        self.assertIsNotNone(SENSITIVE.search('api_key = abc12345'))
        self.assertIsNotNone(SENSITIVE.search('person@example.com'))
        self.assertIsNotNone(EXTERNAL_EFFECT.search('Submit the payment form'))
        self.assertIsNone(EXTERNAL_EFFECT.search('Open the browser settings page'))

    def test_temporal_examples_have_date_and_source(self):
        rows = list(temporal_rows())
        self.assertEqual({split for split, _ in rows}, {'train', 'temporal_valid'})
        for _, row in rows:
            self.assertGreater(row['provenance']['published_on'], '2026-01-31')
            self.assertTrue(row['provenance']['source_url'].startswith('https://'))


if __name__ == '__main__':
    unittest.main()
