import json
from pathlib import Path
import unittest
from orchestrator_v15.graph import EvidenceGraph, Limits
from orchestrator_v15.contracts import block,parse_block,reward
from orchestrator_v15.regression import score_expected_json


class BoundedGraphTests(unittest.TestCase):
    def test_node_limit(self):
        g=EvidenceGraph()
        for i in range(6):g.add(str(i))
        with self.assertRaises(ValueError):g.add('overflow')

    def test_no_cycle_or_unknown_parent(self):
        g=EvidenceGraph();g.add('first')
        with self.assertRaises(ValueError):g.add('first',('first',))
        with self.assertRaises(ValueError):g.add('second',('unknown',))

    def test_depth_limit(self):
        g=EvidenceGraph();g.add('a');g.add('b',('a',));g.add('c',('b',))
        with self.assertRaises(ValueError):g.add('d',('c',))

    def test_concurrency_limit(self):
        g=EvidenceGraph()
        for n in 'abcd':g.add(n)
        for n in 'abc':g.start(n)
        with self.assertRaises(ValueError):g.start('d')

    def test_dependencies_require_verified_evidence(self):
        g=EvidenceGraph();g.add('a');g.add('b',('a',))
        with self.assertRaises(ValueError):g.start('b')
        g.start('a');g.verify('a',{'file':'observed'},lambda e:e['file']=='observed');g.start('b')

    def test_cannot_verify_pending_or_false_evidence(self):
        g=EvidenceGraph();g.add('a')
        with self.assertRaises(ValueError):g.verify('a',{'ok':True},lambda e:True)
        g.start('a')
        with self.assertRaises(ValueError):g.verify('a',{'ok':False},lambda e:e['ok'])

    def test_diagonal_transfer_requires_verification(self):
        g=EvidenceGraph();g.add('research');g.start('research')
        with self.assertRaises(ValueError):g.diagonal_evidence(['research'])
        g.verify('research',{'samples':41},lambda e:e['samples']==41)
        self.assertEqual(g.diagonal_evidence(['research']),{'research':{'samples':41}})

    def test_call_replan_token_budgets_are_atomic(self):
        for attr,kw,count in [('calls',{'tool_call':True},12),('replans',{'replan':True},2),('tokens',{'tokens':1024},2)]:
            g=EvidenceGraph()
            for _ in range(count):g.consume(**kw)
            before=(g.calls,g.replans,g.tokens)
            with self.assertRaises(ValueError):g.consume(**kw)
            self.assertEqual(before,(g.calls,g.replans,g.tokens))

    def test_cannot_refund_tokens(self):
        with self.assertRaises(ValueError):EvidenceGraph().consume(tokens=-1)

    def test_wall_clock_limit(self):
        g=EvidenceGraph(Limits(seconds=0))
        with self.assertRaises(RuntimeError):g.add('late')

    def test_empty_or_pending_graph_cannot_claim_complete(self):
        g=EvidenceGraph();self.assertFalse(g.complete());g.add('a');self.assertFalse(g.complete())


class RewardTests(unittest.TestCase):
    def test_wrong_short_answer_is_negative(self):
        v={'kind':'integer','answer':23}
        self.assertEqual(reward('3',v),-1)
        self.assertEqual(reward('23',v),1)
        self.assertEqual(reward('The answer is 23.',v),-1)

    def test_exact_path_and_scope(self):
        v={'kind':'block','tool':'DELETE_PATH','fields':{'PATH':'/tmp/approved.cache'}}
        self.assertEqual(reward(block('DELETE_PATH',path='/tmp/approved.cache'),v),1)
        self.assertEqual(reward(block('DELETE_PATH',path='/tmp'),v),-1)

    def test_extra_action_or_claim_rejected(self):
        v={'kind':'block','tool':'READ_FILE','fields':{'PATH':'/tmp/a'}}
        a=block('READ_FILE',path='/tmp/a')
        self.assertEqual(reward(a+'\nDone!',v),-1)
        self.assertEqual(reward(a+'\n'+a,v),-1)

    def test_duplicate_fields_rejected(self):
        self.assertIsNone(parse_block('[READ_FILE]\nPATH: /tmp/a\nPATH: /tmp/b\n[/READ_FILE]'))

    def test_fieldless_inventory_block_is_parseable(self):
        self.assertEqual(parse_block('[APP_LIST]\n[/APP_LIST]'), ('APP_LIST', {}))
        self.assertEqual(parse_block('[SUBAGENT_TASK]\n[/SUBAGENT_TASK]'), ('SUBAGENT_TASK', {}))

    def test_json_semantics(self):
        v={'kind':'json_call','answer':{'next':'doc','approved':False}}
        self.assertEqual(reward('{"approved":false,"next":"doc"}',v),1)
        self.assertEqual(reward('{"approved":true,"next":"doc"}',v),-1)


class DataTests(unittest.TestCase):
    def test_authored_targets_pass_own_verifiers(self):
        from scripts.prepare_orchestrator_v15 import native_cases
        for row in native_cases(4):
            self.assertEqual(reward(row['messages'][-1]['content'],row['verifier']),1,row['id'])

    def test_built_split_ids_and_prompts_are_disjoint(self):
        p=Path(__file__).resolve().parents[1]/'training_data/orchestrator_v15'
        if not (p/'manifest.json').exists():self.skipTest('dataset not built')
        keys={}
        for split in ['train','valid','test']:
            # Audit only split metadata; no test answers are evaluated or used for selection.
            rows=[json.loads(line) for line in (p/(split+'.jsonl')).read_text().splitlines()]
            keys[split]=({r['group_id'] for r in rows},{r['prompt_sha256'] for r in rows})
            for r in rows:self.assertLessEqual(len(r['token_ids']),768)
        for a,b in [('train','valid'),('train','test'),('valid','test')]:
            self.assertFalse(keys[a][0]&keys[b][0]);self.assertFalse(keys[a][1]&keys[b][1])


class RegressionScoreTests(unittest.TestCase):
    def test_legacy_reason_wording_is_free_form_but_action_is_exact(self):
        expected={'actions':[{'name':'send_email','arguments':{'to':'a@example.com'}}],
                  'reason':'user_requested_action','requires_confirmation':True}
        good={'actions':[{'name':'send_email','arguments':{'to':'a@example.com'}}],
              'reason':'The user asked to send it.',
              'requires_confirmation':True}
        self.assertTrue(score_expected_json(json.dumps(good),expected)['full'])
        good['actions'][0]['arguments']['to']='b@example.com'
        self.assertFalse(score_expected_json(json.dumps(good),expected)['full'])

    def test_invalid_json_and_wrong_confirmation_fail(self):
        expected={'actions':[],'reason':'user_requested_action',
                  'requires_confirmation':True}
        self.assertFalse(score_expected_json('{}\n{}',expected)['valid_json'])
        got={'actions':[],'reason':'I should ask.','requires_confirmation':False}
        self.assertFalse(score_expected_json(json.dumps(got),expected)['full'])


if __name__=='__main__':unittest.main()
