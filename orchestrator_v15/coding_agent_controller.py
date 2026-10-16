"""Bounded, evidence-based coordination of Kira's text-only coding agents.

The local SUBAGENT_TASK runtime returns advice, not a patch, command result, or
test run. This ledger deliberately keeps those claims distinct.
"""
import json
import re

from .contracts import block, parse_block


CODING = re.compile(r"\b(?:code|coding|bug|fix|patch|refactor|implement|tests?|repository|repo)\b", re.I)
AGENTS = re.compile(r"\b(?:agents|subagents|specialist agents|coding agents)\b", re.I)


def is_coding_agent_request(request):
    text = str(request)
    queue_only = re.search(r'\b(?:queue|schedule)\b.{0,90}\b(?:without running|do not run|later)\b|'
                           r'\b(?:without running|do not run)\b', text, re.I)
    paired_roles = re.search(r'\bcoder\b', text, re.I) and re.search(r'\breviewer\b', text, re.I)
    return bool(CODING.search(text) and (AGENTS.search(text) or paired_roles) and not queue_only)


class CodingAgentController:
    """One coder and one independent reviewer, with a fixed execution budget."""

    def __init__(self, request):
        self.request = str(request)
        self.registry_seen = False
        self.agents = {}
        self.coder = None
        self.reviewer = None
        self.coder_result = None
        self.review_result = None
        self.error = None
        self.actions = 0

    def hint(self):
        if not self.registry_seen:
            return 'Inspect SUBAGENT_REGISTRY before assigning coding work.'
        if self.error:
            return 'A worker failed. Report the failure and do not claim code was changed.'
        if not self.coder or not self.reviewer:
            return 'No distinct coding and review agents were observed. Report that limitation.'
        if self.coder_result is None:
            return (f'Run registered coder {self.coder} with RUN: true on the user coding task. '
                    'Request a bounded proposed change and evidence; the worker cannot edit files.')
        if self.review_result is None:
            return (f'Run distinct registered reviewer {self.reviewer} with RUN: true. '
                    f'Give it the coder output as review input: {self.coder_result[:900]}')
        return 'Report the proposal and review as unverified advice; no patch or test execution occurred.'

    def repair_empty_task(self, answer):
        """Fill an empty worker block only from the user goal and observed ledger."""
        parsed = parse_block(answer)
        if not parsed or parsed[0] != 'SUBAGENT_TASK' or parsed[1]:
            return answer
        if not self.registry_seen or self.complete_or_blocked():
            return answer
        goal = re.sub(r'\s+', ' ', self.request).strip()[:400]
        if self.coder_result is None:
            task = (f'Propose a bounded code change for this user task: {goal}. '
                    'Return a specific proposal; do not claim files changed or tests ran.')
            return block('SUBAGENT_TASK', agent=self.coder, task=task, run='true')
        evidence = re.sub(r'\s+', ' ', self.coder_result).strip()[:650]
        task = (f'Review the proposed code change for this user task: {goal}. '
                f'Coder output to review: {evidence}. Return concrete issues and test ideas.')
        return block('SUBAGENT_TASK', agent=self.reviewer, task=task, run='true')

    def preflight(self, answer):
        parsed = parse_block(answer)
        if parsed is None:
            if self.complete_or_blocked():
                return None
            return self.hint()
        tag, fields = parsed
        if not self.registry_seen:
            return None if tag == 'SUBAGENT_REGISTRY' else self.hint()
        if self.complete_or_blocked():
            return 'The coding-agent workflow is finished or blocked; do not run another worker.'
        expected = self.coder if self.coder_result is None else self.reviewer
        if tag != 'SUBAGENT_TASK':
            return self.hint()
        if fields.get('AGENT', '').casefold() != expected.casefold():
            return f'Use the observed registered {"coder" if self.coder_result is None else "reviewer"} {expected}.'
        if fields.get('RUN', '').lower() != 'true':
            return 'Run this bounded worker now with RUN: true; a queued packet is not a result.'
        task = fields.get('TASK', '')
        if len(task) < 24 or not CODING.search(task):
            return 'The worker TASK must describe the concrete coding work or code review.'
        if self.coder_result is not None and not any(
                fragment.casefold() in task.casefold()
                for fragment in self._review_evidence_fragments()):
            return 'Give the reviewer a concrete excerpt from the observed coder output.'
        if self.actions >= 3:
            return 'The coding-agent action budget is exhausted.'
        return None

    def _review_evidence_fragments(self):
        paths = re.findall(r'\b[\w./-]+\.(?:py|js|ts|tsx|go|rs|java|swift)\b',
                           self.coder_result or '', re.I)
        if paths:
            return paths[:8]
        words = re.findall(r'[A-Za-z][A-Za-z0-9_.-]{5,}', self.coder_result or '')
        generic = {'propose', 'change', 'review', 'tests', 'coding', 'patch', 'worker'}
        return [word for word in words if word.casefold() not in generic][:20] or [self.coder_result[:24]]

    def record(self, answer, result):
        parsed = parse_block(answer)
        if parsed is None:
            return
        tag, _fields = parsed
        self.actions += 1
        result = str(result)
        if tag == 'SUBAGENT_REGISTRY':
            self.registry_seen = result.startswith('SUBAGENT_REGISTRY:')
            if not self.registry_seen:
                self.error = 'Registry inspection failed.'
                return
            for line in result.splitlines():
                match = re.match(r'^\s*([a-z][a-z0-9_-]{1,40})\s+(mlx|ollama)\s+(.+)$', line, re.I)
                if match:
                    self.agents[match.group(1)] = match.group(3)
            self.coder = next((name for name, purpose in self.agents.items()
                               if re.search(r'\b(?:coder|coding|implementation|patch)\b', name+' '+purpose, re.I)), None)
            self.reviewer = next((name for name, purpose in self.agents.items()
                                  if name != self.coder and re.search(r'\b(?:review|test|qa|verify)\b', name+' '+purpose, re.I)), None)
        elif tag == 'SUBAGENT_TASK':
            if result.startswith('SUBAGENT_TASK ERROR:') or 'No registered subagent' in result:
                self.error = result[:500]
                return
            match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.S)
            if not match:
                self.error = 'Worker returned no structured execution result.'
                return
            try:
                payload = json.loads(match.group(1))
            except ValueError:
                self.error = 'Worker returned malformed JSON.'
                return
            if payload.get('ok') is not True or not str(payload.get('output') or '').strip():
                self.error = 'Worker did not return a successful, nonempty result.'
                return
            expected = self.coder if self.coder_result is None else self.reviewer
            if payload.get('agent') and str(payload['agent']).casefold() != expected.casefold():
                self.error = 'Worker result identity did not match the assigned agent.'
                return
            output = str(payload['output']).strip()[:1800]
            if self.coder_result is None:
                self.coder_result = output
            else:
                self.review_result = output

    def complete_or_blocked(self):
        return bool(self.error or (self.registry_seen and (not self.coder or not self.reviewer)) or
                    (self.coder_result is not None and self.review_result is not None))

    def final_answer(self):
        if not self.complete_or_blocked():
            return None
        if self.error:
            return f'Coding-agent workflow stopped: {self.error} No code changes or tests were verified.'
        if not self.coder or not self.reviewer:
            return ('No distinct registered coder and reviewer were available. '
                    'No coding task was run, and no code changes or tests were verified.')
        return (f'Unverified coder report ({self.coder}):\n{self.coder_result}\n\n'
                f'Unverified independent review ({self.reviewer}):\n{self.review_result}\n\n'
                'These are agent reports. Kira did not apply a patch or run tests in this workflow.')
