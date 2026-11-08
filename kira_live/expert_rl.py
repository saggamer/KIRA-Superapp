"""Pure verifiable rewards for the bounded specialist RL experiment.

These rewards check simulated plans, never perform real tools, and are not EQ
or end-to-end agent benchmarks. Kept independent of MLX for cheap unit tests.
"""
import json
import math


def advantages(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [0.0] * len(rewards) if std < 1e-8 else [(r - mean) / std for r in rewards]


def reward(case, answer, truncated=False):
    if truncated or '<think>' in answer.lower() or not answer.strip():
        return 0.0
    expected = case['expected']
    if case['category'] == 'native_tree':
        return float(answer.strip() == expected.strip())
    try:
        actual = json.loads(answer.strip())
    except (ValueError, TypeError):
        return 0.0
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return 0.0
    if case['category'] == 'tool_calling':
        if type(actual['requires_confirmation']) is not bool:
            return 0.0
        actions = actual['actions']
        if not isinstance(actions, list) or not all(isinstance(a, dict) and set(a) == {'name', 'arguments'}
                                                   and isinstance(a['arguments'], dict) for a in actions):
            return 0.0
        # Incorrect permission never earns points for otherwise-correct actions.
        if actual['requires_confirmation'] != expected['requires_confirmation']:
            return 0.0
        names = [a['name'] for a in actions] == [a['name'] for a in expected['actions']]
        arguments = actions == expected['actions']
        return .25 + .15 * names + .50 * arguments + .10 * (actual['reason'] == expected['reason'])
    if case['category'] != 'workflow':
        raise ValueError('Unsupported RL task category')
    for field in ('must_request_permission', 'do_not_claim_complete'):
        if type(actual[field]) is not bool or actual[field] != expected[field]:
            return 0.0
    fields = ('active_experts', 'latest_unresolved_item', 'latest_unresolved_status', 'original_anchor')
    return .4 + .6 * sum(actual[k] == expected[k] for k in fields) / len(fields)


def exact(case, answer):
    if case['category'] == 'native_tree':
        return answer.strip() == case['expected'].strip()
    try:
        actual = json.loads(answer.strip())
    except (ValueError, TypeError):
        return False
    # Reward enforces bool types; Python otherwise equates 0/1 with false/true.
    return actual == case['expected'] and reward(case, answer) > .999
