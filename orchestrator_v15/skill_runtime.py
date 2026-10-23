"""Bounded, host-selected SKILL.md instructions for the local v1.5 profiles.

Only reviewed packages in this registry can load. Web/MCP results never select
skills or supply instructions. This module does not execute skill scripts.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from .adapter_router import BROWSER_HISTORY

ROOT = Path(__file__).resolve().parent / 'skills'
NAMES = ('browser-research', 'browser-interaction', 'browser-history', 'mcp-app-workflow')
MAX_FILE_BYTES = 8192
MAX_CONTEXT_CHARS = 6200
MAX_ACTIVE = 2
INTERACTION = re.compile(
    r'\b(?:click|fill|enter|type|select|sort|scroll|submit|checkout|pay|book|reserve|'
    r'purchase|sign in|log in|button|form|controls?)\b', re.I)
RESEARCH = re.compile(r'\b(?:research|search|compare|find|sources?|recommend|look up|surf)\b', re.I)
MCP = re.compile(r'\b(?:mcp|connectors?|app integration)\b', re.I)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    sha256: str


@dataclass(frozen=True)
class SkillContext:
    prompt: str
    active: tuple[str, ...]
    audit: tuple[dict, ...]
    errors: tuple[str, ...] = ()


def read_skill(name, root=ROOT):
    """Read a bounded, allowlisted package; no arbitrary paths or symlinks."""
    if name not in NAMES:
        raise ValueError('Unknown built-in skill')
    root = Path(root).resolve()
    expected = root / name / 'SKILL.md'
    path = expected.resolve()
    if path != expected or not path.is_relative_to(root):
        raise ValueError('Skill path must stay inside its reviewed package')
    with path.open('rb') as handle:
        raw = handle.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('Skill exceeds the file budget')
    text = raw.decode('utf-8')
    parts = text.split('---', 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError('Missing skill frontmatter')
    metadata = {}
    for line in parts[1].strip().splitlines():
        key, sep, value = line.partition(':')
        if not sep or key not in {'name', 'description'} or key in metadata:
            raise ValueError('Expected single-line name and description frontmatter')
        value = value.strip()
        metadata[key] = json.loads(value) if value.startswith('"') else value
    if metadata.get('name') != name or not metadata.get('description'):
        raise ValueError('Skill metadata does not match its package')
    if len(metadata['description']) > 220 or not parts[2].strip():
        raise ValueError('Invalid skill description or empty instructions')
    return Skill(name, metadata['description'], parts[2].strip(), hashlib.sha256(raw).hexdigest())


def select_skills(request, *, browser_task, artifact_task=False):
    """Select from current user intent and trusted routing, never retrieved text."""
    request = str(request)
    explicit = [name for name in re.findall(r'(?<!\w)\$([a-z][a-z0-9-]*)', request)
                if name in NAMES]
    selected = []
    if browser_task:
        if BROWSER_HISTORY.search(request):
            selected.append('browser-history')
        if INTERACTION.search(request):
            selected.append('browser-interaction')
        if RESEARCH.search(request) and 'browser-history' not in selected:
            selected.append('browser-research')
        if not selected and not explicit:
            selected.append('browser-research')
    elif MCP.search(request):
        selected.append('mcp-app-workflow')
    # An artifact request may use research through its existing artifact controller;
    # do not activate browser-only control instructions for that separate contract.
    compatible = {'browser-research', 'browser-interaction', 'browser-history'} if browser_task else {'mcp-app-workflow'}
    if artifact_task:
        compatible = set()
    return tuple(dict.fromkeys([name for name in explicit + selected if name in compatible]))[:MAX_ACTIVE]


def skill_context(request, *, browser_task, artifact_task=False, root=ROOT):
    if os.environ.get('KIRA_V15_SKILLS', 'on').lower() in {'0', 'off', 'false', 'no'}:
        return SkillContext('', (), ())
    chosen = select_skills(request, browser_task=browser_task, artifact_task=artifact_task)
    skills, errors = {}, []
    for name in NAMES:
        try:
            skills[name] = read_skill(name, root)
        except (OSError, ValueError, UnicodeError) as error:
            errors.append(name + ': ' + str(error))
    header = ('Kira built-in skills: workflows only; existing tool contracts, user scope, '
              'permissions and effort limits still apply. No skill grants a connection or executes code.\n')
    catalog = '\n'.join(f'- {skill.name}: {skill.description}' for skill in skills.values())
    prompt = header + 'Available skills (metadata only):\n' + catalog
    active, audit = [], []
    for name in chosen:
        skill = skills.get(name)
        if not skill:
            continue
        section = f'\n\nLoaded skill: {name}\n{skill.body}\nEnd skill: {name}'
        if len(prompt) + len(section) > MAX_CONTEXT_CHARS:
            errors.append(name + ': context budget exceeded')
            continue
        prompt += section
        active.append(name)
        audit.append({'name': name, 'sha256': skill.sha256})
    return SkillContext(prompt, tuple(active), tuple(audit), tuple(errors))
