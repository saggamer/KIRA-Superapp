"""Opt-in batched Diagonal Orchestral reasoning candidate.

Three independent, concise decision branches share the same *observed* state.
MLX batches their decoding; a separate model pass evaluates them. The Kira
controller remains the execution authority and rejects invalid actions.
"""
import json
from types import SimpleNamespace

from orchestrator_v15.contracts import parse_block
from orchestrator_v15.diagonal_level import score_proposal
from orchestrator_v15.browser_action_repair import repair_observed_action


LENSES = (
    "Find the smallest useful next step toward the user's goal.",
    "Look for missing source evidence and a useful alternate route.",
    "Audit target grounding, permission, stale state, and verification.",
)
BRANCH_SCHEMA = (
    'Return only JSON with string keys "evidence", "plan", "risk", "action". '
    'Keep evidence, plan and risk each under 180 characters. Evidence may cite '
    'only tool results already supplied. Action must be exactly one closed '
    '[BROWSER_TOOL] block or a grounded final answer. Do not execute tools, '
    'claim an unobserved result, or include another action block.'
    ' For an observed control, use ACTION: act, OPERATION: click/fill/select_option, '
    'TARGET_ROLE and TARGET_NAME from the latest elements, and the actual TABID. '
    'Do not use a TARGET CSS expression.'
    ' For lower-page content, propose ACTION: scroll with PIXELS: 700 and use the returned visibleText.'
)

ACTION_FIELDS = frozenset({
    'ACTION', 'QUERY', 'URL', 'TABID', 'SCREENSHOT', 'OPERATION', 'TARGET_NAME',
    'TARGET_ROLE', 'SELECTOR', 'INDEX', 'VALUE', 'TEXT', 'OBSERVATION_ID', 'PIXELS',
    'HISTORY_ID', 'KIND', 'SOURCE', 'LIMIT', 'OFFSET', 'TEXT_OFFSET', 'RESULT_OFFSET', 'RESULT_INDEX',
    'ENABLED', 'COSMETIC', 'AUTO_UPDATE', 'SITE', 'SITE_ENABLED', 'RULE', 'PAGE_URL', 'RESOURCE_TYPE',
})
ALLOWED_BROWSER_ACTIONS = frozenset({
    'search', 'navigate', 'new_tab', 'observe', 'act', 'snapshot',
    'inspect_page', 'back', 'select_tab', 'list_tabs', 'scroll', 'reload',
    'search_history', 'read_history', 'revisit_history',
    'adblock_status', 'adblock_set', 'adblock_rule', 'adblock_check', 'adblock_update',
})


def _valid_browser_block(value):
    parsed = parse_block(value)
    return bool(parsed and parsed[0] == 'BROWSER_TOOL' and
                parsed[1].get('ACTION', '').lower() in ALLOWED_BROWSER_ACTIONS)


def _action_text(value):
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return None
    fields = {str(key).upper(): item for key, item in value.items()}
    tool = fields.pop('TOOL', 'BROWSER_TOOL')
    text = fields.pop('TEXT', None)
    if text is not None and text not in ('[BROWSER_TOOL]', 'BROWSER_TOOL'):
        if not fields and isinstance(text, str) and len(text) <= 900 and '[' not in text and ']' not in text:
            return text.strip()
        return None
    arguments = fields.pop('ARGUMENTS', None)
    if arguments is not None:
        if not isinstance(arguments, dict) or fields.keys() - {'NAME', 'ACTION'}:
            return None
        name = fields.pop('NAME', None) or fields.pop('ACTION', None)
        fields = {str(key).upper(): item for key, item in arguments.items()}
        if name is None:
            name = fields.pop('OPERATION', None)
        if name is None:
            return None
        fields = {'ACTION': name, **fields}
    if tool != 'BROWSER_TOOL' or not fields or not set(fields) <= ACTION_FIELDS or 'ACTION' not in fields:
        return None
    if any(not isinstance(item, (str, int, bool)) or '\n' in str(item) or '[' in str(item)
           for item in fields.values()):
        return None
    lines = [f'{key}: {str(value).lower() if isinstance(value, bool) else value}'
             for key, value in fields.items()]
    return '[BROWSER_TOOL]\n' + '\n'.join(lines) + '\n[/BROWSER_TOOL]'


def batch_closed_json(model, tokenizer, prompts, *, max_tokens, sampler,
                      generator_cls=None, verbose=False):
    """Decode independent MLX branches together; release each completed JSON early."""
    if generator_cls is None:
        from mlx_lm.generate import BatchGenerator as generator_cls
    eos_ids = getattr(tokenizer, 'eos_token_ids', None) or [tokenizer.eos_token_id]
    generator = generator_cls(model, stop_tokens=[[token] for token in eos_ids],
                              sampler=sampler)
    ids = generator.insert(prompts, [max_tokens] * len(prompts))
    tokens = {uid: [] for uid in ids}
    active = set(ids)
    try:
        while active:
            responses = generator.next_generated()
            if not responses:
                break
            complete = []
            for response in responses:
                uid = response.uid
                if uid not in active:
                    continue
                if response.finish_reason != 'stop':
                    tokens[uid].append(response.token)
                if response.finish_reason is not None:
                    active.remove(uid)
                    continue
                # Decode only when a token may have closed the top-level JSON.
                if '}' in tokenizer.decode([response.token]):
                    try:
                        value = json.loads(tokenizer.decode(tokens[uid]).strip(), strict=False)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, dict):
                        active.remove(uid)
                        complete.append(uid)
            if complete:
                generator.remove(complete)
    finally:
        generator.close()
    return SimpleNamespace(texts=[tokenizer.decode(tokens[uid]) for uid in ids])


def parse_branch(raw):
    """Accept bounded, auditable branch summaries, never arbitrary tool text."""
    plain = str(raw).strip()
    try:
        value = json.loads(plain, strict=False)
    except (TypeError, ValueError):
        parsed = parse_block(plain)
        if _valid_browser_block(plain):
            return {'evidence': '', 'plan': '', 'risk': '', 'action': plain}
        if (plain and len(plain) <= 900 and not plain.startswith(('{', '```'))
                and '[' not in plain and ']' not in plain):
            return {'evidence': '', 'plan': '', 'risk': '', 'action': plain}
        return None
    if not isinstance(value, dict) or set(value) != {'evidence', 'plan', 'risk', 'action'}:
        return None
    if any(not isinstance(value[key], str) or len(value[key]) > 1000
           for key in ('evidence', 'plan', 'risk')):
        return None
    action = _action_text(value['action'])
    if action is None:
        return None
    if not action or len(action) > 1300:
        return None
    if '[BROWSER_TOOL]' in action and not _valid_browser_block(action):
        return None
    if action.startswith('FINAL_ANSWER:'):
        action = action.removeprefix('FINAL_ANSWER:').strip()
    value = {key: item[:180] if key in ('evidence', 'plan', 'risk') else item
             for key, item in value.items()}
    value['action'] = action
    return value


def generate_branches(model, tokenizer, messages, *, max_tokens=256,
                      batch_generate_fn=None, evaluator_generate_fn=None,
                      sampler=None):
    """Batch three independent plans, then run an independent evaluation pass."""
    if batch_generate_fn is None:
        batch_generate_fn = batch_closed_json
    if evaluator_generate_fn is None:
        from mlx_lm import generate as evaluator_generate_fn
    prompts = []
    for lens in LENSES:
        branch_messages = [dict(item) for item in messages]
        branch_messages[-1]['content'] += '\nIndependent Diagonal branch: ' + lens + '\n' + BRANCH_SCHEMA
        prompt = tokenizer.apply_chat_template(branch_messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        prompts.append(tokenizer.encode(prompt, add_special_tokens=False))
    result = batch_generate_fn(model, tokenizer, prompts, max_tokens=min(384, max_tokens),
                               sampler=sampler, verbose=False)
    raw_branches = list(result.texts)
    branches = [parse_branch(raw) for raw in raw_branches]
    valid = [(index, branch) for index, branch in enumerate(branches) if branch]
    judge_scores = [0] * len(branches)
    distinct_actions = {branch['action'] for _, branch in valid}
    if len(distinct_actions) > 1:
        descriptions = [dict(index=index, evidence=branch['evidence'], plan=branch['plan'],
                             risk=branch['risk'], action=branch['action']) for index, branch in valid]
        judge_messages = [
            {'role': 'system', 'content': (
                'Independently evaluate the three candidate next steps against the supplied '
                'user request and tool evidence. Do not execute any action. Return only a JSON '
                'array of three integers from 0 to 5 in branch order. Penalize unsupported '
                'claims, redundant steps, invalid tool actions and missing verification.')},
            {'role': 'user', 'content': 'User request and latest evidence:\n' +
             str(messages[-1].get('content', ''))[:2500] +
             '\nCandidate branches:\n' + json.dumps(descriptions, ensure_ascii=False)},
        ]
        judge_prompt = tokenizer.apply_chat_template(judge_messages, tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
        judge_raw = evaluator_generate_fn(model, tokenizer, prompt=judge_prompt,
                                          max_tokens=48, sampler=sampler, verbose=False)
        try:
            parsed = json.loads(str(judge_raw).strip())
            if (isinstance(parsed, list) and len(parsed) == len(branches)
                    and all(type(x) is int and 0 <= x <= 5 for x in parsed)):
                judge_scores = parsed
        except (TypeError, ValueError):
            pass
    return {'branches': branches, 'raw_branches': raw_branches,
            'judge_scores': judge_scores, 'generation_mode': 'mlx_batch_early_stop',
            'model_generations': len(LENSES) + int(len(distinct_actions) > 1)}


def select_branch(result, controller):
    """Independent model review breaks ties; deterministic validity wins."""
    ranked = []
    for index, branch in enumerate(result['branches']):
        if branch is None:
            continue
        action = repair_observed_action(branch['action'], controller)
        score = score_proposal(action, controller)
        if score is None:
            continue
        ranked.append((score, result['judge_scores'][index], -index, index, action))
    if not ranked:
        return None, {'valid': 0, 'selected_index': None}
    selected = max(ranked)
    return selected[-1], {'valid': len(ranked), 'selected_index': selected[-2],
                          'deterministic_score': selected[0], 'judge_score': selected[1]}
