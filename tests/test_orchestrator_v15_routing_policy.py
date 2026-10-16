import unittest

from orchestrator_v15.routing_policy import subagent_dispatch


class RoutingPolicyTests(unittest.TestCase):
    def test_registered_dispatch(self):
        self.assertTrue(subagent_dispatch('Registry confirms review_bot. Queue this task without running it.'))
        self.assertTrue(subagent_dispatch('The registry confirms metrics_bot. Run it now.'))

    def test_read_only_and_unrelated_requests_keep_original_route(self):
        self.assertFalse(subagent_dispatch('Check which subagents are registered before assigning work.'))
        self.assertFalse(subagent_dispatch('Discover MCP configurations.'))
        self.assertFalse(subagent_dispatch('Run a browser search for headphones.'))


if __name__ == '__main__': unittest.main()
