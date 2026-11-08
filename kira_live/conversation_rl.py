"""Narrow, inspectable conversation rewards, not an EQ or factuality judge."""
import re


def sentences(text):
    return len([part for part in re.split(r'[.!?]+(?:\s|$)', text.strip()) if part.strip()])


def score(case, answer, truncated=False):
    text = answer.strip()
    low = text.lower()
    rubric = case['rubric']
    clean = bool(text) and not truncated and not any(
        marker in low for marker in ('<think>', '</think>', '<|', '[read_file]', '[web_search]', 'requires_confirmation'))
    identity_relevant = rubric.get('identity', False)
    if not identity_relevant and re.search(r'\bkira\b', low):
        clean = False
    forbidden = [pattern for pattern in rubric.get('forbidden', []) if re.search(pattern, low)]
    missing_hard = [pattern for pattern in rubric.get('hard_required', []) if not re.search(pattern, low)]
    if not clean or forbidden or missing_hard:
        return {'reward': 0.0, 'passed': False, 'checks': {'clean': clean, 'no_contradiction': not forbidden},
                'forbidden_matches': forbidden, 'missing_hard_requirements': missing_hard}
    checks = {'clean': True, 'no_contradiction': True}
    for index, pattern in enumerate(rubric.get('required', [])):
        checks[f'required_{index}'] = bool(re.search(pattern, low))
    if 'sentences' in rubric:
        checks['sentence_count'] = sentences(text) == rubric['sentences']
    if 'max_words' in rubric:
        checks['brief'] = len(text.split()) <= rubric['max_words']
    if 'exact' in rubric:
        normalized = text.strip().rstrip('.!?').strip().lower()
        checks['exact_answer'] = normalized in [v.lower() for v in rubric['exact']]
    if rubric.get('no_advice'):
        checks['no_advice'] = not re.search(r'\b(you should|try\b|step \d|breath\w*|you can start|make a plan)\b', low)
    if rubric.get('direct_draft'):
        checks['no_preface'] = not re.search(r'\b(here(?: is|\x27s)|you can (?:say|send)|a message you|as an ai)\b', low)
    # No reward for empty generic replies: task-specific checks must exist.
    task_checks = [value for key, value in checks.items() if key not in ('clean', 'no_contradiction')]
    if not task_checks:
        raise ValueError('Missing conversation reward criteria')
    return {'reward': sum(task_checks) / len(task_checks), 'passed': all(task_checks), 'checks': checks,
            'forbidden_matches': []}
