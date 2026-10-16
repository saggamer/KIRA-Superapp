import unittest

from orchestrator_v15.agentic_action_repair import direct_result, repair_action
from orchestrator_v15.contracts import parse_block


class AgenticActionRepairTests(unittest.TestCase):
    def test_read_only_tool_families_are_grounded_in_request(self):
        cases = (
            ('[BROWSER_TOOL]\nACTION: APP_LIST\n[/BROWSER_TOOL]',
             'List the available apps before planning.', 'APP_LIST'),
            ('[BROWSER_TOOL]\nACTION: observe\nURL: notes\n[/BROWSER_TOOL]',
             'Inspect the read-only Notes app integration context.', 'APP_INTEGRATION'),
            ('[MCP_DISCOVER]\n[/MCP_DISCOVER]',
             'Check which specialist subagents are registered.', 'SUBAGENT_REGISTRY'),
        )
        for proposed, request, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(parse_block(repair_action(proposed, request))[0], expected)

    def test_explicit_run_flag_is_repaired_but_queue_stays_queued(self):
        proposed = ('[SUBAGENT_TASK]\nAGENT: audit_helper\nTASK: Review the budget.\n'
                    '[/SUBAGENT_TASK]')
        run = repair_action(proposed, 'Run audit_helper now on this task: Review the budget.')
        queue = repair_action(proposed, 'Queue audit_helper without running it: Review the budget.')
        self.assertEqual(parse_block(run)[1]['RUN'], 'true')
        self.assertEqual(parse_block(queue)[1]['RUN'], 'false')

    def test_unrelated_or_unobserved_work_is_not_rewritten(self):
        proposed = '[MCP_DISCOVER]\n[/MCP_DISCOVER]'
        self.assertEqual(repair_action(proposed, 'Discover MCP configuration targets.'), proposed)
        self.assertIsNone(direct_result('SUBAGENT_TASK', 'Run audit_helper now.', 'No evidence'))

    def test_simple_results_stop_from_tool_evidence(self):
        self.assertIn('Notes', direct_result('APP_INTEGRATION',
            'Inspect Notes app integration context.', 'APP_INTEGRATION: Notes context'))
        self.assertIn('audit_helper', direct_result('SUBAGENT_REGISTRY',
            'Check registered subagents.', 'SUBAGENT_REGISTRY: audit_helper'))
        self.assertIn('queued', direct_result('SUBAGENT_TASK',
            'Queue audit_helper.', 'SUBAGENT_TASK: queued packet'))


if __name__ == '__main__':
    unittest.main()
