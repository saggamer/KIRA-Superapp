"""Task-scoped browser evidence and recovery state, never model-authored memory."""
from collections import OrderedDict
import json
import re
from urllib.parse import urlsplit


def web_url(value):
    value = str(value or '')
    try:
        parts = urlsplit(value)
        return value if len(value) <= 2048 and parts.scheme in {'http', 'https'} and parts.netloc else ''
    except ValueError:
        return ''


def page_state(observation):
    """Describe observed obstacles; this never authorizes or performs recovery."""
    title = str(observation.get('title', '')).lower()
    text = str(observation.get('visibleText') or observation.get('text') or '')[:3000].lower()
    error = str(observation.get('error') or '')
    if error:
        return 'stale' if 'stale' in error.lower() else 'tool_error'
    if (re.search(r'just a moment|attention required|security verification', title) and
            re.search(r'cloudflare|verify|checking|security', title + ' ' + text)) or re.search(
                r'verify (?:that )?you are human|complete the captcha|performing security verification', text):
        return 'challenge'
    status = observation.get('status')
    if status == 401:
        return 'sign_in_required'
    if isinstance(status, int) and status >= 400:
        return 'access_denied' if status == 403 else 'http_error'
    if observation.get('url') and not (text.strip() or observation.get('elements')):
        return 'empty'
    return 'ready' if observation.get('url') else 'unknown'


def normalize_observation(observation):
    """Make structured inspection usable as evidence, without inventing controls."""
    if not isinstance(observation, dict):
        return {'error': 'Browser returned a non-object observation.'}
    page = observation.get('page')
    if not isinstance(page, dict):
        return observation
    result = {key: value for key, value in observation.items() if key != 'page'}
    result.update({key: page[key] for key in ('url', 'title', 'text', 'headings', 'forms') if key in page})
    result['results'] = [{'url': item.get('url'), 'title': item.get('text', '')}
                         for item in page.get('links', []) if isinstance(item, dict)]
    result['elements'] = []  # Structural links are not fresh visible click targets.
    result['inspection'] = True
    return result


class BrowserWorkingState:
    def __init__(self, request=''):
        self.sources = OrderedDict()
        self.links = OrderedDict()
        self.failures = []
        self.interactions = []
        self.current = {}
        self.delta = {}
        self._fingerprint = None
        self.visited = OrderedDict()
        self.terms = set(re.findall(r'[a-z0-9]{3,}', str(request).lower())) - {
            'the', 'and', 'with', 'from', 'this', 'that', 'your', 'for', 'then', 'please',
            'browser', 'page', 'source', 'sources', 'open', 'find', 'read', 'use'}

    def _relevance(self, item):
        text = ' '.join([item.get('title', ''), item.get('snippet', ''), *item.get('excerpts', [])])
        return len(self.terms.intersection(re.findall(r'[a-z0-9]{3,}', text.lower())))

    def _link(self, url, title, snippet=''):
        url = web_url(url)
        if not url:
            return
        self.links[url] = {'url': url, 'title': str(title or '')[:120], 'snippet': str(snippet or '')[:180]}
        self.links.move_to_end(url)
        while len(self.links) > 32:
            least = min(self.links, key=lambda key: self._relevance(self.links[key]))
            self.links.pop(least)

    def record(self, fields, observation):
        state = page_state(observation)
        previous = self.current
        url = web_url(observation.get('url'))
        if url and not observation.get('cached'):
            fingerprint = json.dumps({key: observation.get(key) for key in
                ('url', 'text', 'visibleText', 'elements', 'scroll')}, sort_keys=True, default=str)
            self.current = {key: str(observation[key])[:2048 if key == 'url' else 120] for key in ('tabId', 'url', 'title', 'observationId')
                            if key in observation}
            self.current['state'] = state
            self.current['fresh'] = bool(observation.get('observationId'))
            self.delta = {'page_changed': previous.get('url') != url,
                          'content_changed': self._fingerprint != fingerprint,
                          'observation_changed': previous.get('observationId') != observation.get('observationId'),
                          'last_action': fields.get('OPERATION') or fields.get('ACTION')}
            self._fingerprint = fingerprint
            source = self.sources.get(url, {'url': url, 'excerpts': []})
            source.update({'title': str(observation.get('title') or '')[:120], 'state': state})
            content = str(observation.get('visibleText') or observation.get('text') or '').strip()
            if content and state == 'ready':
                excerpt = content[-400:] if observation.get('scroll', {}).get('y', 0) else content[:400]
                if excerpt not in source['excerpts']:
                    source['excerpts'] = (source['excerpts'][:1] + [excerpt])[-2:]
            self.sources[url] = source
            self.sources.move_to_end(url)
            self.visited[url] = state
            self.visited.move_to_end(url)
            while len(self.visited) > 64:
                self.visited.popitem(last=False)
            while len(self.sources) > 8:
                least = min(self.sources, key=lambda key: self._relevance(self.sources[key]))
                self.sources.pop(least)
        elif state in {'tool_error', 'stale'}:
            self.current = {**previous, 'state': state, 'fresh': False}
            self.delta = {'last_action': fields.get('OPERATION') or fields.get('ACTION'), 'failed': True}
        for item in observation.get('results', []):
            if isinstance(item, dict):
                self._link(item.get('url') or item.get('href'), item.get('title'),
                           item.get('body') or item.get('snippet') or item.get('description'))
        for item in observation.get('elements', []):
            if isinstance(item, dict) and item.get('role') == 'link':
                self._link(item.get('href'), item.get('name'))
        if fields.get('ACTION', '').lower() == 'act' and state not in {'tool_error', 'stale'}:
            self.interactions = (self.interactions + [{
                'operation': fields.get('OPERATION', ''),
                'target': str(fields.get('TARGET_NAME') or fields.get('INDEX') or '')[:120],
                'url': url, 'page_changed': self.delta.get('page_changed', False),
                'content_changed': self.delta.get('content_changed', False)}])[-6:]
        if state in {'challenge', 'access_denied', 'http_error', 'tool_error', 'stale'}:
            failure = {'url': url or previous.get('url', ''), 'state': state,
                       'action': fields.get('OPERATION') or fields.get('ACTION'),
                       'detail': str(observation.get('error') or observation.get('title') or '')[:180]}
            self.failures = (self.failures + [failure])[-4:]

    def snapshot(self):
        # Keep unopened alternatives ahead of already-read links, retaining their
        # exact observed URLs. Sources contain excerpts, never inferred facts.
        links = sorted(self.links.values(), key=lambda item: (item['url'] in self.visited, -self._relevance(item)))
        candidates = [{**item, 'visited': item['url'] in self.visited,
                       'state': self.visited.get(item['url'], 'unvisited')}
                      for item in links[:8]]
        return {'current': self.current, 'last_change': self.delta,
                'sources': sorted(self.sources.values(), key=self._relevance, reverse=True)[:6], 'candidate_links': candidates,
                'recent_failures': self.failures, 'recent_interactions': self.interactions}

    def recovery_hint(self):
        state = self.current.get('state')
        if state in {'challenge', 'access_denied'}:
            return ('The site is blocking access. Choose an unvisited relevant observed source or search '
                    'for an official alternative. Do not reload the same challenge or solve/bypass it. '
                    'If this exact site is essential, ask the user to complete verification there.')
        if state == 'sign_in_required':
            return 'The site requires sign-in. Use public alternatives if sufficient, or ask the user to sign in; do not invent access.'
        if state == 'stale':
            return 'The action used stale page state. Observe the intended tab again, then choose its current control.'
        if state in {'tool_error', 'http_error', 'empty'}:
            return ('The last step returned an error or no readable page. Inspect the current page once, '
                    'select another observed source, or refine the search. Do not repeat the same failed action unchanged.')
        return ''
