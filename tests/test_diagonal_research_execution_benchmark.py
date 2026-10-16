import json
import unittest

from scripts.benchmark_diagonal_research_execution import score


BASE = 'http://127.0.0.1:12345'


def action(block, observation=None):
    result = block
    if observation is not None:
        result += '\n\nBROWSER_TOOL:\n' + json.dumps(observation)
    return {'action': block, 'result': result}


def navigate(path, text):
    url = BASE + path
    return action('[BROWSER_TOOL]\nACTION: navigate\nURL: ' + url + '\n[/BROWSER_TOOL]',
                  {'url': url, 'status': 200, 'text': text})


def act(operation, target, value=None, result=None):
    block = ('[BROWSER_TOOL]\nACTION: act\nOPERATION: ' + operation +
             '\nTARGET_NAME: ' + target +
             ('\nVALUE: ' + value if value is not None else '') +
             '\n[/BROWSER_TOOL]')
    return action(block, result)


class ResearchExecutionScoringTests(unittest.TestCase):
    def test_plan_requires_all_observed_sources_and_correct_submission(self):
        actions = [navigate('/source/a', 'Plan A $30 88% 3 days'),
                   navigate('/source/b', 'Plan B $39 94% 2 days'),
                   navigate('/source/c', 'Plan C $42 98% 1 day'),
                   act('select_option', 'Plan', 'B'),
                   act('fill', 'Evidence', '$39, 94%, 2 days'),
                   act('click', 'Save recommendation', result={
                       'url': BASE + '/save', 'status': 200, 'text': 'Saved result'})]
        submissions = [{'path': '/save', 'fields': {
            'plan': 'B', 'evidence': '$39, 94%, 2 days'}}]
        self.assertTrue(all(score('plan_research', actions, submissions,
                                  'Plan B was saved.', BASE).values()))
        missing = score('plan_research', actions[:2] + actions[3:], submissions,
                        'Plan B was saved.', BASE)
        self.assertFalse(missing['all_sources_observed'])
        wrong = score('plan_research', actions, [{'path': '/save', 'fields': {
            'plan': 'A', 'evidence': '$39, 94%, 2 days'}}],
            'Plan B was saved.', BASE)
        self.assertFalse(wrong['correct_plan'])
        duplicate = score('plan_research', actions, submissions * 2,
                          'Plan B was saved.', BASE)
        self.assertFalse(duplicate['single_submission'])

    def test_public_research_requires_saved_values_from_loaded_page(self):
        title = 'Python Documentation'
        url = 'https://docs.python.org/3/'
        actions = [action('[BROWSER_TOOL]\nACTION: search\nQUERY: official Python docs\n[/BROWSER_TOOL]',
                          {'results': [{'url': url, 'title': title}]}),
                   action('[BROWSER_TOOL]\nACTION: navigate\nURL: ' + url + '\n[/BROWSER_TOOL]',
                          {'url': url, 'title': title, 'status': 200,
                           'text': 'Official Python documentation'}),
                   act('fill', 'Page title', title),
                   act('fill', 'Page URL', url),
                   act('click', 'Save evidence', result={
                       'url': BASE + '/log', 'status': 200, 'text': 'Saved result'})]
        submissions = [{'path': '/log', 'fields': {'title': title, 'url': url}}]
        self.assertTrue(all(score('public_research_log', actions, submissions,
                                  title + '\n' + url, BASE).values()))
        fabricated = score('public_research_log', actions, [{'path': '/log',
            'fields': {'title': 'Invented', 'url': url}}],
            'Invented\n' + url, BASE)
        self.assertFalse(fabricated['matching_evidence_saved'])


if __name__ == '__main__':
    unittest.main()
