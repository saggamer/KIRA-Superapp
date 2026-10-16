"""Spark corpus and 9B budget invariants that prevent silent regressions."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import json

from orchestrator_v15.contracts import reward
from orchestrator_v15.spark_budget import SparkBudget
from orchestrator_v15.spark_prompt import SPARK_BROWSER_SYSTEM
from orchestrator_v15.spark_tools import native_browser_call_to_block
from scripts.prepare_orchestrator_v15_spark9b import (
    executed_browser_cases, has_sensitive_pattern, normalize_messages, tool_cases,
    native_browser_row,
)
from scripts.train_spark_full_rl import select_candidates, structured_reward


class Spark9BTests(unittest.TestCase):
    def test_budget_counts_every_full_expert(self):
        budget = SparkBudget()
        budget.validate()
        self.assertEqual(budget.added_expert_parameters, 4_718_592_000)
        self.assertEqual(budget.total_parameters, 8_830_835_264)
        with self.assertRaises(ValueError):
            SparkBudget(added_experts_per_layer=16).validate()

    def test_native_tool_call_arguments_are_normalized_without_mutation(self):
        original = [{'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'search', 'arguments': '{"query":"Kira"}'}}]}]
        result = normalize_messages(original)
        self.assertEqual(result[0]['tool_calls'][0]['function']['arguments'],
                         {'query': 'Kira'})
        self.assertIsInstance(original[0]['tool_calls'][0]['function']['arguments'], str)

    def test_sensitive_training_pattern_quarantine(self):
        self.assertTrue(has_sensitive_pattern([{'content': 'Contact person@company.com'}]))
        self.assertTrue(has_sensitive_pattern([{'content': 'api_key: abcdefgh'}]))
        self.assertFalse(has_sensitive_pattern([{'content': 'Search the Kira documentation'}]))

    def test_authored_cases_use_typed_kira_bridge(self):
        cases = list(tool_cases(2))
        self.assertEqual(len(cases), 42)
        for case in cases:
            self.assertEqual(reward(case['messages'][-1]['content'], case['verifier']), 1.0)
        fill = next(case for case in cases if case['family'] == 'browser_fill')
        self.assertIn('OPERATION: fill', fill['messages'][-1]['content'])
        self.assertNotIn('fill_text', fill['messages'][-1]['content'])

    def test_executed_source_keeps_task_group_and_no_ref_index(self):
        steps = {
            'amazon_browse_category_trajectory.json': (4, 5),
            'amazon_category_exploration_trajectory.json': (1, 2),
            'gpt54_lms_calculate_weighted_grade_trajectory.json': (1, 2, 3, 4),
            'gpt54_lms_check_course_grade_trajectory.json': (1, 2, 3, 4),
            'gpt54_lms_exam_conflict_trajectory.json': (1, 2, 3),
            'gpt54_lms_find_project_assignments_trajectory.json': (1, 2, 3, 5),
            'gpt54_lms_read_urgent_announcement_trajectory.json': (1,),
            'gpt54_lms_view_late_policy_trajectory.json': (1, 2, 3, 4),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for filename, numbers in steps.items():
                episode = {'task_id': filename.replace('_trajectory.json', ''),
                           'evaluation': {'success': True},
                           'agent': {'trajectory': [
                               {'step': number, 'status': 'success',
                                'targets': {'role': 'button', 'name': f'Control {number}'},
                                'action': ({'action': 'select', 'value': 'Newest'}
                                           if filename.startswith('amazon_browse') and number == 4
                                           else {'action': 'click'})}
                               for number in numbers]}}
                (root / filename).write_text(json.dumps(episode))
            with patch('scripts.prepare_orchestrator_v15_spark9b.BREAKINGWEB', root):
                cases = list(executed_browser_cases())
                expanded = list(executed_browser_cases(expanded=True))
        self.assertEqual(len(cases), 4)
        self.assertEqual(len({case['group_id'] for case in cases}), 2)
        for case in cases:
            self.assertEqual(reward(case['messages'][-1]['content'], case['verifier']), 1.0)
            self.assertNotIn('INDEX:', case['messages'][-1]['content'])
            self.assertEqual(case['upstream']['repo_revision'],
                             '51734a1be1f006c35208748b5cf1640f95530333')
        self.assertEqual(len(expanded), 24)
        self.assertEqual(len({case['group_id'] for case in expanded}), 8)

    def test_rl_reward_never_scores_invalid_prose_as_valid_tool(self):
        row = next(case for case in tool_cases(1) if case['family'] == 'browser_inspect')
        self.assertEqual(structured_reward(row['messages'][-1]['content'], row['verifier']), 1.0)
        self.assertEqual(structured_reward('I will inspect the tab now.', row['verifier']), -1.0)
        partial = structured_reward('BROWSER_TOOL ACTION: inspect_page', row['verifier'],
                                    lexical_shaping=True)
        self.assertGreater(partial, -1.0)
        self.assertLess(partial, 0.0)

    def test_rl_selection_includes_browser_rows_before_easy_answers(self):
        rows = [
            {'id': 'truth-0', 'family': 'truth', 'answer_start': 200,
             'verifier': {'kind': 'exact'}},
            {'id': 'inspect-0', 'family': 'browser_inspect', 'answer_start': 293,
             'verifier': {'kind': 'block'}},
            {'id': 'fill-0', 'family': 'browser_fill', 'answer_start': 305,
             'verifier': {'kind': 'block'}},
            {'id': 'too-long', 'family': 'browser_fill', 'answer_start': 385,
             'verifier': {'kind': 'block'}},
        ]
        self.assertEqual([r['id'] for r in select_candidates(rows)],
                         ['inspect-0', 'fill-0', 'truth-0'])

    def test_browser_prompt_shows_executable_bridge_syntax(self):
        self.assertIn('[BROWSER_TOOL]\nACTION: navigate', SPARK_BROWSER_SYSTEM)
        self.assertIn('OBSERVATION_ID', SPARK_BROWSER_SYSTEM)
        self.assertNotIn('fill_text', SPARK_BROWSER_SYSTEM)

    def test_native_browser_call_requires_fresh_index_observation(self):
        head = '<tool_call>BROWSER_TOOL'
        action = '<arg_key>action</arg_key><arg_value>act</arg_value>'
        click = '<arg_key>operation</arg_key><arg_value>click</arg_value>'
        index = '<arg_key>index</arg_key><arg_value>3</arg_value>'
        end = '</tool_call>'
        self.assertIsNone(native_browser_call_to_block(head + action + click + index + end))
        fresh = '<arg_key>observation_id</arg_key><arg_value>fresh-1</arg_value>'
        block = native_browser_call_to_block(head + action + click + index + fresh + end)
        self.assertIn('INDEX: 3\nOBSERVATION_ID: fresh-1', block)
        raw = native_browser_call_to_block('BROWSER_TOOL' + action + click + index + fresh)
        self.assertEqual(raw, block)
        self.assertIsNone(native_browser_call_to_block('I will click index 3.'))

    def test_browser_case_converts_to_spark_native_tool_target(self):
        original = next(case for case in tool_cases(1) if case['family'] == 'browser_index')
        converted = native_browser_row(original)
        self.assertTrue(converted['native_browser_tool'])
        self.assertEqual(converted['messages'][-1]['tool_calls'][0]['function']['name'],
                         'BROWSER_TOOL')
        self.assertEqual(converted['messages'][-1]['tool_calls'][0]['function']['arguments']['index'], 3)
        self.assertEqual(converted['bridge_target'], original['messages'][-1]['content'])
        self.assertIn('[BROWSER_TOOL]', original['messages'][-1]['content'])


if __name__ == '__main__':
    unittest.main()
