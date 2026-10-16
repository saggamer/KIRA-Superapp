import unittest

from orchestrator_v15.coding_agent_controller import CodingAgentController, is_coding_agent_request
from orchestrator_v15.contracts import block


REGISTRY = ('SUBAGENT_REGISTRY:\n```text\n'
            'code_worker              mlx      coding implementation proposals\n'
            'review_worker            mlx      independent code review and test planning\n```')
CODER = block('SUBAGENT_TASK', agent='code_worker',
              task='Propose a patch for the parser bug with a bounded explanation.', run='true')
REVIEW = block('SUBAGENT_TASK', agent='review_worker',
               task='Review the parser.py proposed change and check the edge case.', run='true')


def response(agent, output):
    import json
    return ('SUBAGENT_TASK:\nAgent: `' + agent + '`\n```json\n' +
            json.dumps({'ok': True, 'agent': agent, 'output': output}) + '\n```')


class CodingAgentControllerTests(unittest.TestCase):
    def test_trigger_is_scoped_to_coding_and_agents(self):
        self.assertTrue(is_coding_agent_request('Have coding agents fix and review this bug.'))
        self.assertFalse(is_coding_agent_request('Use the browser to research a bug.'))
        self.assertFalse(is_coding_agent_request('List registered subagents.'))
        self.assertFalse(is_coding_agent_request('Ask one coding agent to inspect this bug.'))
        self.assertFalse(is_coding_agent_request(
            'Queue a code review for the agent without running it.'))

    def test_registry_coder_reviewer_and_honest_final(self):
        ledger = CodingAgentController('Have coding agents fix and review the parser bug.')
        self.assertIsNotNone(ledger.preflight(CODER))
        registry = block('SUBAGENT_REGISTRY')
        self.assertIsNone(ledger.preflight(registry))
        ledger.record(registry, REGISTRY)
        self.assertIsNone(ledger.preflight(CODER))
        ledger.record(CODER, response('code_worker', 'Change parser.py to reject empty input.'))
        self.assertIsNotNone(ledger.preflight(REVIEW))
        grounded_review = block('SUBAGENT_TASK', agent='review_worker',
                                task='Review coder proposal: Change parser.py to reject empty input. '
                                     'Check missing tests and edge cases.', run='true')
        self.assertIsNone(ledger.preflight(grounded_review))
        ledger.record(grounded_review, response('review_worker',
                                                'Add an empty-input test; patch looks plausible.'))
        final = ledger.final_answer()
        self.assertIn('Change parser.py', final)
        self.assertIn('Add an empty-input test', final)
        self.assertIn('did not apply a patch or run tests', final)
        self.assertIsNotNone(ledger.preflight(CODER))

    def test_queued_or_unregistered_worker_cannot_count(self):
        ledger = CodingAgentController('Coordinate coding agents on this bug.')
        ledger.record(block('SUBAGENT_REGISTRY'), REGISTRY)
        self.assertIsNotNone(ledger.preflight(block('SUBAGENT_TASK', agent='unknown_agent',
                                                    task='Propose a code patch for the parser bug.', run='true')))
        self.assertIsNotNone(ledger.preflight(block('SUBAGENT_TASK', agent='code_worker',
                                                    task='Propose a code patch for the parser bug.', run='false')))

    def test_empty_task_repair_uses_observed_roles_and_coder_evidence(self):
        ledger = CodingAgentController('Coordinate coding agents to fix the parser bug.')
        empty = block('SUBAGENT_TASK')
        self.assertEqual(ledger.repair_empty_task(empty), empty)
        ledger.record(block('SUBAGENT_REGISTRY'), REGISTRY)
        coder_action = ledger.repair_empty_task(empty)
        self.assertIn('AGENT: code_worker', coder_action)
        self.assertIsNone(ledger.preflight(coder_action))
        ledger.record(coder_action, response('code_worker', 'Change parser.py to reject empty input.'))
        review_action = ledger.repair_empty_task(empty)
        self.assertIn('AGENT: review_worker', review_action)
        self.assertIn('parser.py', review_action)
        self.assertIsNone(ledger.preflight(review_action))

    def test_failure_and_missing_review_worker_stop_without_success(self):
        ledger = CodingAgentController('Coordinate coding agents on this bug.')
        ledger.record(block('SUBAGENT_REGISTRY'), REGISTRY)
        ledger.record(CODER, 'SUBAGENT_TASK ERROR: model failed')
        self.assertIn('stopped', ledger.final_answer())
        self.assertIn('No code changes', ledger.final_answer())
        empty = CodingAgentController('Coordinate coding agents on this bug.')
        empty.record(block('SUBAGENT_REGISTRY'),
                     'SUBAGENT_REGISTRY:\n```text\ncode_worker mlx coding implementation\n```')
        self.assertIn('No distinct registered', empty.final_answer())

    def test_result_identity_must_match_assigned_worker(self):
        ledger = CodingAgentController('Coordinate coding agents on this bug.')
        ledger.record(block('SUBAGENT_REGISTRY'), REGISTRY)
        ledger.record(CODER, response('review_worker', 'Change parser.py.'))
        self.assertIn('identity did not match', ledger.final_answer())


if __name__ == '__main__':
    unittest.main()
