"""Narrow answer checks for the authored grounding curriculum, not EQ scores."""
import re


def conversation_checks(row, answer, truncated=False):
    text = answer.lower().strip()
    rubric = row['rubric']
    checks = {'complete_public_answer': bool(text) and not truncated and not any(
        tag in text for tag in ('<think>', '</think>', '<|', '[read_file]', '[web_search]', 'requires_confirmation'))}
    if 'exact' in rubric:
        checks['exact_answer'] = text.rstrip('.!?').strip() in rubric['exact']
    for value in rubric.get('contains', []):
        checks[f'required:{value}'] = value in text
    for value in rubric.get('forbid', []):
        checks[f'forbidden:{value}'] = value not in text
    if 'sentences' in rubric:
        checks['sentence_count'] = len([s for s in re.split(r'[.!?]+(?:\s|$)', answer.strip()) if s.strip()]) == rubric['sentences']
    if 'max_words' in rubric:
        checks['length'] = len(answer.split()) <= rubric['max_words']
    return checks


def no_regression_score(measured, baseline):
    """Never trade a baseline passing generated case for teacher-forced loss."""
    old = {row['id'] for row in baseline['generated']['rows'] if row['passed']}
    now = {row['id'] for row in measured['generated']['rows'] if row['passed']}
    return (len(now), -measured['lm_loss']) if old <= now else None


def critical_decision_failures(cases, baseline, candidate):
    """Do not overlook a lost permission just because its whole plan failed.

    Missing/malformed output counts as a regression when a field was correct
    before. IDs are scoped by execution mode, not assumed globally unique.
    """
    import json
    def outputs(result):
        values = {}
        for row in result['rows']:
            try:
                value = json.loads(row['answer'])
            except (ValueError, TypeError):
                value = {}
            values[(row['mode'], row['id'])] = value if isinstance(value, dict) else {}
        return values
    old, new = outputs(baseline), outputs(candidate)
    failures = []
    for case in cases:
        key = (case['mode'], case['id'])
        expected = case.get('expected')
        if not isinstance(expected, dict):
            continue  # Native bracket contracts have no JSON decision fields.
        for field in ('requires_confirmation', 'must_request_permission', 'do_not_claim_complete'):
            target = expected.get(field)
            if type(target) is not bool:
                continue
            before, after = old.get(key, {}).get(field), new.get(key, {}).get(field)
            if type(before) is bool and before == target and not (type(after) is bool and after == target):
                failures.append({'mode': key[0], 'id': key[1], 'field': field,
                                 'expected': target, 'actual': after})
    return failures


def generate_cases(model, rows):
    import mlx.core as mx
    from kira_live.expert_rl import exact
    from scripts.train_kira_a_rl import prompt_ids
    records = []
    model.eval()
    for row in rows:
        messages = row['messages'][:-1]
        if row['mode']:
            model.set_execution_mode(True)
            ids = prompt_ids(model, messages)
            embeddings, ngram_ids = model.embed(ids), ids
        else:
            user = messages[-1]['content']
            ids = mx.array([model.tokenizer.encode(user, add_special_tokens=False)], dtype=mx.int32)
            prefix = model.build_live_prefix(model.embed(ids), listener_unit_ids=ids, conversation_turns=messages[1:-1])
            embeddings, ngram_ids = prefix.embeddings, prefix.ngram_ids
        output = model.generate_from_embeddings(embeddings, ngram_ids=ngram_ids, max_tokens=180, compute_response_hidden=False)
        truncated = len(output.token_ids) >= 180
        checks = ({'exact_typed_tool_or_state': exact(row, output.text), 'not_truncated': not truncated}
                  if row['mode'] else conversation_checks(row, output.text, truncated))
        records.append({'id': row['id'], 'category': row['category'], 'mode': row['mode'],
                        'user': messages[-1]['content'], 'history': messages[1:-1], 'answer': output.text,
                        'checks': checks, 'passed': all(checks.values()), 'token_cap_hit': truncated})
    return {'passed': sum(r['passed'] for r in records), 'total': len(records), 'rows': records,
            'limitations': 'Authored format/content checks, not general EQ, tool execution, or audio.'}
