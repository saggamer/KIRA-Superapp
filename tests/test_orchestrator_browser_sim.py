import unittest

from orchestrator_v15.browser_sim import BrowserSim, Task, TASK_KINDS
from orchestrator_v15.contracts import block


class BrowserSimTests(unittest.TestCase):
    def test_oracle_trajectories_finish_with_verified_state(self):
        for kind in TASK_KINDS:
            with self.subTest(kind=kind):
                sim=BrowserSim(Task(kind,101))
                rewards=[]
                for _ in range(5):
                    response=sim.oracle()
                    reward,_=sim.step(response)
                    rewards.append(reward)
                    if sim.done:break
                self.assertTrue(sim.success())
                self.assertGreater(sum(rewards),0)

    def test_unauthorized_and_duplicate_purchases_are_blocked(self):
        sim=BrowserSim(Task('permission',102))
        reward,_=sim.step(block('BROWSER_TOOL',action='act',operation='click',
                                target_role='button',target_name='Place order',tabid=sim.task.tab))
        self.assertEqual(reward,-2.0)
        self.assertFalse(sim.success())
        self.assertEqual(sim.purchase_count,0)

    def test_no_unverified_success_credit(self):
        sim=BrowserSim(Task('lookup',103))
        reward,_=sim.step(f'The page reports {sim.task.count} verified items.')
        self.assertLess(reward,0)
        self.assertFalse(sim.success())


if __name__=='__main__':unittest.main()
