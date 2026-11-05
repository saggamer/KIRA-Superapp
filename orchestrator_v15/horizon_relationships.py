"""Explicit object/part graphs and fail-closed visual relationship reviews.

This module validates intended scenes and supplied observations. It does not
detect objects or infer relationships from pixels.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

PREDICATES = {'inside','contains','on','above','below','left_of','right_of','behind',
              'in_front_of','between','through','around','attached_to','riding',
              'supports','integrated_with'}
KINDS = {'object','part','scene'}


def validate_conditioning_budget(tokenizer, prompt, max_sequence_length):
    """Match the pinned Klein Qwen template and reject truncation before inference."""
    text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                                       add_generation_prompt=True,enable_thinking=False)
    count=len(tokenizer(text,truncation=False)['input_ids'])
    if count > max_sequence_length:
        raise ValueError(f'The complete scene plan needs {count} tokens, exceeding {max_sequence_length}; shorten it explicitly')
    return count


@dataclass(frozen=True)
class SceneEntity:
    id: str
    label: str
    kind: str = 'object'
    count: int = 1
    owner: str | None = None
    material: str | None = None
    box: tuple | None = None


@dataclass(frozen=True)
class SceneRelation:
    id: str
    subject: str
    predicate: str
    targets: tuple


@dataclass(frozen=True)
class SceneGraph:
    entities: tuple
    relations: tuple
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1 or not 1 <= len(self.entities) <= 32 or len(self.relations) > 64:
            raise ValueError('Unsupported graph version or graph size')
        ids = [node.id for node in self.entities]
        edges = [edge.id for edge in self.relations]
        for value in ids + edges:
            if not isinstance(value,str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,47}',value):
                raise ValueError('Use short lowercase symbolic IDs')
        if len(set(ids)) != len(ids) or len(set(edges)) != len(edges):
            raise ValueError('Entity and relation IDs must be unique')
        nodes = {node.id:node for node in self.entities}
        for node in self.entities:
            if node.kind not in KINDS or not isinstance(node.label,str) or not 1 <= len(node.label.strip()) <= 160:
                raise ValueError('Invalid entity type or label')
            if type(node.count) is not int or not 1 <= node.count <= 16:
                raise ValueError('Entity count must be an integer from one to sixteen')
            if node.material is not None and (not isinstance(node.material,str) or not 1 <= len(node.material.strip()) <= 80):
                raise ValueError('Invalid scoped material')
            if node.kind == 'part' and (node.owner not in nodes or node.owner == node.id):
                raise ValueError('Every part needs an existing distinct owner')
            if node.kind != 'part' and node.owner is not None:
                raise ValueError('Only parts can name an owner')
            if node.box is not None:
                if len(node.box) != 4 or any(type(x) not in {int,float} or not math.isfinite(x) for x in node.box):
                    raise ValueError('Box must contain four finite normalized coordinates')
                x0,y0,x1,y1 = node.box
                if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
                    raise ValueError('Invalid normalized entity box')
        for node in self.entities:
            seen, current = set(), node
            while current.owner:
                if current.id in seen:
                    raise ValueError('Part ownership cannot form a cycle')
                seen.add(current.id)
                current = nodes[current.owner]
        directed = {}
        for edge in self.relations:
            if edge.subject not in nodes or edge.predicate not in PREDICATES:
                raise ValueError('Unknown relation subject or predicate')
            needed = 2 if edge.predicate == 'between' else 1
            if any(not isinstance(target,str) for target in edge.targets) or len(edge.targets) != needed or len(set(edge.targets)) != needed:
                raise ValueError('Between needs two distinct anchors; other relations need one')
            if any(target not in nodes or target == edge.subject for target in edge.targets):
                raise ValueError('Relations need distinct existing subject and anchors')
            if edge.predicate in {'left_of','right_of','above','below','inside','contains'}:
                family = {'right_of':'left_of','below':'above','contains':'inside'}.get(edge.predicate,edge.predicate)
                pair = (edge.subject,edge.targets[0])
                if edge.predicate in {'right_of','below','contains'}:
                    pair = pair[::-1]
                directed.setdefault(family,set()).add(pair)
        for pairs in directed.values():
            state = {}
            def visit(node):
                if state.get(node) == 1:
                    raise ValueError('Contradictory directed spatial relations')
                if state.get(node) == 2:
                    return
                state[node] = 1
                for left,right in pairs:
                    if left == node:
                        visit(right)
                state[node] = 2
            for node in nodes:
                visit(node)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value,dict) or set(value) - {'version','entities','relations'}:
            raise ValueError('Unknown graph fields')
        entities = []
        for record in value.get('entities',[]):
            if not isinstance(record,dict):
                raise ValueError('Entity records must be objects')
            item = dict(record)
            if item.get('box') is not None:
                item['box'] = tuple(item['box'])
            try:
                entities.append(SceneEntity(**item))
            except TypeError as exc:
                raise ValueError('Invalid entity fields') from exc
        relations = []
        for record in value.get('relations',[]):
            if not isinstance(record,dict) or not isinstance(record.get('targets'),list):
                raise ValueError('Relation records need an explicit target list')
            try:
                relations.append(SceneRelation(**{**record,'targets':tuple(record['targets'])}))
            except TypeError as exc:
                raise ValueError('Invalid relationship fields') from exc
        return cls(tuple(entities),tuple(relations),value.get('version',1))

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @property
    def fingerprint(self):
        value = self.to_dict()
        value['entities'].sort(key=lambda item:item['id'])
        value['relations'].sort(key=lambda item:item['id'])
        for edge in value['relations']:
            if edge['predicate'] == 'between':
                edge['targets'].sort()
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @property
    def signatures(self):
        nodes = {node.id:node for node in self.entities}
        result = set()
        for edge in self.relations:
            anchors = [nodes[target].kind + ':' + nodes[target].label.casefold() for target in edge.targets]
            if edge.predicate == 'between':
                anchors.sort()
            result.add(edge.predicate + ':' + '|'.join(anchors))
        return result

    def render(self):
        nodes = {node.id:node for node in self.entities}
        def name(identifier):
            node = nodes[identifier]
            return (nodes[node.owner].label + "'s " if node.owner else '') + node.label
        lines = ['Required scene objects and relationships:']
        for node in self.entities:
            text = f'{node.count} x {name(node.id)}'
            if node.material:
                text += f'; material of this object: {node.material}'
            if node.box:
                text += '; planned normalized box: ' + ','.join(f'{x:g}' for x in node.box)
            lines.append(text + '.')
        for edge in self.relations:
            lines.append(name(edge.subject) + ' ' + edge.predicate.replace('_',' ') + ' '
                         + ' and '.join(name(target) for target in edge.targets) + '.')
        return '\n'.join(lines)

    def assess(self, *, instances, observations, materials=None, owners=None, reviewer):
        """Assess supplied visual judgments; missing evidence stays unknown."""
        if reviewer not in {'human','agent_visual'}:
            raise ValueError('Semantic graph observations require a human or visual reviewer')
        materials = materials or {}
        owners = owners or {}
        nodes = {node.id:node for node in self.entities}
        if set(instances) - set(nodes) or set(observations) - {edge.id for edge in self.relations}:
            raise ValueError('Observation refers to an unknown graph ID')
        results = {}
        occupied = set()
        for node in self.entities:
            detected = instances.get(node.id)
            if detected is None:
                results['count:'+node.id] = 'unknown'
            else:
                if not isinstance(detected,list) or any(not isinstance(value,str) or not value for value in detected):
                    raise ValueError('Instances need explicit nonempty visual IDs')
                unique = set(detected)
                valid = len(detected) == len(unique) == node.count and not (unique & occupied)
                results['count:'+node.id] = 'pass' if valid else 'fail'
                occupied.update(unique)
            if node.material:
                observed = materials.get(node.id)
                if observed is not None and not isinstance(observed,str):
                    raise ValueError('Observed material must be a label or unknown')
                results['material:'+node.id] = ('unknown' if observed is None else
                    'pass' if observed.strip().casefold() == node.material.strip().casefold() else 'fail')
        for node in self.entities:
            if node.owner:
                parts, parents = instances.get(node.id), instances.get(node.owner)
                missing = not parts or not parents or any(part not in owners for part in parts)
                results['owner:'+node.id] = ('unknown' if missing else 'pass'
                    if results['count:'+node.id] == results['count:'+node.owner] == 'pass'
                    and all(owners[part] in parents for part in parts) else 'fail')
        for edge in self.relations:
            item = observations.get(edge.id)
            key = 'relation:'+edge.id
            if item is None or item.get('status') == 'unknown':
                results[key] = 'unknown'
                continue
            if item.get('status') not in {'pass','fail'}:
                raise ValueError('Relationship status must be pass, fail or unknown')
            actual = tuple(item.get('targets',[]))
            anchors_match = (len(actual) == 2 and len(set(actual)) == 2 and set(actual) == set(edge.targets)) if edge.predicate == 'between' else actual == edge.targets
            matches = item.get('subject') == edge.subject and item.get('predicate') == edge.predicate and anchors_match
            grounded = all(results['count:'+node] == 'pass' for node in (edge.subject,*edge.targets))
            grounded = grounded and all(results.get('owner:'+node,'pass') == 'pass' for node in (edge.subject,*edge.targets))
            results[key] = 'pass' if item['status'] == 'pass' and matches and grounded else 'fail'
        return {'complete':all(value == 'pass' for value in results.values()),
                'checks':results,'reviewer':reviewer,'automatic_pixel_grounding':False,
                'graph_fingerprint':self.fingerprint}
