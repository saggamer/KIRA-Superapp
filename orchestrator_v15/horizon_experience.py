"""Horizon 1 episodic experience: evidence, review, retrieval and replay.

Memory changes task planning, not neural weights. Only explicit reviewed
outcomes can supply lessons; generated outputs start as pending attempts.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid
from .horizon_relationships import SceneGraph


def _scene_graph(value, parent=None):
    existing = parent.get('scene_graph') if parent else None
    graph = value if isinstance(value,SceneGraph) else SceneGraph.from_dict(value) if value is not None else None
    if existing:
        old = SceneGraph.from_dict(existing)
        if graph and graph.fingerprint != old.fingerprint:
            raise ValueError('A retry cannot silently change its required scene relationships')
        graph = graph or old
    return graph


POLICIES = {
    'preserve_scene': 'Change the requested target. Preserve the camera, background, lighting and geometry of unrelated objects.',
    'perform_requested_change': 'The requested change must be clearly visible. Returning an unchanged source is a failed edit. Preservation applies only to unrelated content.',
    'check_anatomy': 'Preserve natural anatomy and the original number of body parts. Enlarge existing eyes in their original positions; do not add extra eyes.',
    'make_pose_explicit': 'Make the requested pose change clearly visible in the specified body part, while preserving identity and all other body parts.',
    'explore_composition': 'Explore a distinctive silhouette, spatial arrangement and material combination that satisfy this brief. Avoid merely recoloring a familiar composition.',
    'anchor_subject': 'Keep the main requested subject immediately recognizable before adding imaginative styling.',
    'show_function': 'Show visible functional details that identify the requested object. Do not replace a usable object with an abstract decorative symbol.',
    'bind_relationships': 'Represent requested spatial relationships literally. Do not merely place the requested motifs next to each other.',
}
REVIEWERS = {'agent_visual', 'human', 'programmatic'}
MODES = {'editing', 'creativity'}


def _tokens(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - {'the','a','an','of','to','in','and','its','this','make','with'}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pixel_hash(path):
    """Catch identical images re-encoded as different PNG/JPEG container bytes."""
    from PIL import Image,UnidentifiedImageError
    try:
        with Image.open(path) as image:
            pixels=image.convert('RGBA')
            header=f'RGBA:{pixels.width}:{pixels.height}:'.encode()
            return hashlib.sha256(header+pixels.tobytes()).hexdigest()
    except (OSError,UnidentifiedImageError,ValueError):
        return None


def _asset_identities(asset):
    identities={asset['sha256']}
    pixels=_pixel_hash(asset['path'])
    if pixels:identities.add('pixels:'+pixels)
    return identities


class HorizonExperience:
    def __init__(self, directory):
        self.directory = Path(directory)

    def record(self, *, mode, prompt, output, model, source=None, split='experience',
               group=None, seed=None, parent=None, effective_prompt=None, lessons=(), scene_graph=None,
               visual_contract=None, adherence_contract=None):
        if mode not in MODES or not prompt.strip():
            raise ValueError('An explicit visual mode and prompt are required')
        if mode == 'editing' and source is None:
            raise ValueError('Editing experience requires source pixels')
        if split not in {'train', 'validation', 'test', 'experience'}:
            raise ValueError('Unknown data split')
        parent_item = self.get(parent) if parent else None
        if parent_item and parent_item['split'] in {'test','validation'}:
            split = parent_item['split']
        if parent_item and (parent_item['mode'] != mode or parent_item['prompt'] != prompt):
            raise ValueError('A repair must retain the original task and mode')
        from .horizon_visual_contract import VisualContract
        inherited = parent_item.get('visual_contract') if parent_item else None
        contract = VisualContract.from_dict(visual_contract or inherited) if visual_contract or inherited else None
        if inherited and contract.fingerprint != VisualContract.from_dict(inherited).fingerprint:
            raise ValueError('A repair cannot change its required functional contract')
        if contract and (contract.mode != mode or contract.brief != prompt):
            raise ValueError('Visual contract must match the exact task and mode')
        from .horizon_adherence import AdherenceContract
        prior_adherence=parent_item.get('adherence_contract') if parent_item else None
        adherence=AdherenceContract.from_dict(adherence_contract or prior_adherence) if adherence_contract or prior_adherence else None
        if prior_adherence and adherence.fingerprint != AdherenceContract.from_dict(prior_adherence).fingerprint:
            raise ValueError('A repair cannot relax or replace its instruction requirements')
        if adherence:adherence.validate_request(prompt,mode,source)
        graph = _scene_graph(scene_graph or (contract.graph if contract else None),parent_item)
        if contract and graph.fingerprint != contract.graph.fingerprint:
            raise ValueError('Scene graph and functional contract disagree')
        assets = {}
        for role, path in (('output', output), ('source', source)):
            if path is not None:
                path = Path(path).resolve(strict=True)
                assets[role] = {'path': str(path), 'sha256': _hash(path),'pixel_sha256':_pixel_hash(path)}
        if parent_item and mode == 'editing' and assets['source']['sha256'] != parent_item['assets']['source']['sha256']:
            raise ValueError('A repair must use the same source pixels')
        item = {'schema':1, 'model_name':'Horizon 1', 'id':uuid.uuid4().hex,
                'created':datetime.now(timezone.utc).isoformat(), 'mode':mode,
                'prompt':prompt, 'effective_prompt':effective_prompt or prompt,
                'model':model, 'assets':assets, 'split':split,
                'group':group or (parent_item['group'] if parent_item else assets.get('source', assets['output'])['sha256']),
                'seed':seed, 'parent':parent, 'retrieved_lessons':list(lessons),
                'scene_graph':graph.to_dict() if graph else None,
                'visual_contract':contract.to_dict() if contract else None,
                'adherence_contract':adherence.to_dict() if adherence else None,
                'review':None}
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write(item)
        return item

    def _write(self, item):
        path = self.directory / f"{item['id']}.json"
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(item, indent=2) + '\n')
        temporary.replace(path)

    def get(self, episode):
        if not re.fullmatch(r'[a-f0-9]{32}', episode):
            raise ValueError('Invalid episode ID')
        return json.loads((self.directory / f'{episode}.json').read_text())

    def review(self, episode, *, reviewer, instruction, preservation, critique,
               policies=(), creativity=None, supersede=False, correction=None, relationship_observation=None,
               role_observations=None, adherence_observations=None):
        item = self.get(episode)
        if reviewer not in REVIEWERS or not critique.strip():
            raise ValueError('Review requires provenance and a critique')
        if type(instruction) is not bool or type(preservation) is not bool:
            raise ValueError('Semantic fulfillment and preservation need separate explicit decisions')
        if any(policy not in POLICIES for policy in policies):
            raise ValueError('Unsupported lesson policy')
        if reviewer == 'programmatic' and (instruction or creativity):
            raise ValueError('Pixel checks cannot establish semantic success or creativity')
        if creativity is not None and type(creativity) is not bool:
            raise ValueError('Creativity judgment must be explicit')
        if correction is not None and (reviewer == 'programmatic' or not correction.strip() or len(correction) > 1500):
            raise ValueError('A visual or human reviewer must explicitly supply a bounded correction')
        if not self.intact(item):
            raise ValueError('Experience evidence changed or is missing')
        adherence_assessment=None
        if item.get('adherence_contract'):
            from .horizon_adherence import AdherenceContract
            adherence=AdherenceContract.from_dict(item['adherence_contract'])
            if adherence_observations is not None:
                if not isinstance(adherence_observations,dict):raise ValueError('Adherence observations must be an object')
                if adherence_observations.get('reviewer','').replace('assistant_visual','agent_visual') != reviewer:
                    raise ValueError('Adherence observations and episode reviewer must have the same provenance')
                adherence_assessment=adherence.assess(adherence_observations,item['assets']['output']['path'])
            if instruction and (not adherence_assessment or not adherence_assessment['instruction_success']):
                raise ValueError('Instruction success requires all required, forbidden and preservation checks to pass')
            preserved=[r.id for r in adherence.requirements if r.category=='preservation']
            if preservation and preserved and (not adherence_assessment or any(
                    adherence_assessment['checks'][key]['status']!='pass' for key in preserved)):
                raise ValueError('Preservation success requires its explicit requirement observations')
        elif adherence_observations is not None:
            raise ValueError('Adherence observations require a recorded instruction contract')
        relationship_assessment = None
        if item.get('scene_graph'):
            if relationship_observation is not None:
                if item.get('visual_contract'):
                    from .horizon_visual_contract import VisualContract
                    relationship_assessment = VisualContract.from_dict(item['visual_contract']).assess(
                        **relationship_observation, role_observations=role_observations, reviewer=reviewer)
                else:
                    relationship_assessment = SceneGraph.from_dict(item['scene_graph']).assess(
                        **relationship_observation,reviewer=reviewer)
            if instruction and (not relationship_assessment or not relationship_assessment['complete']):
                raise ValueError('Graph-conditioned success requires complete explicit object/part/relationship observations')
        if item['review'] is not None and not supersede:
            raise ValueError('Already reviewed; preserve the original judgment')
        if item['review'] is not None:
            item.setdefault('review_history', []).append(item['review'])
        item['review'] = {'reviewer':reviewer, 'instruction':instruction,
                          'preservation':preservation, 'creativity':creativity,
                          'critique':critique, 'policies':list(policies),
                          'correction':correction,
                          'relationship_assessment':relationship_assessment,
                          'relationship_observation':relationship_observation,
                          'role_observations':role_observations,
                          'adherence_assessment':adherence_assessment,
                          'adherence_observations':adherence_observations,
                          'reviewed':datetime.now(timezone.utc).isoformat()}
        self._write(item)
        return item

    @staticmethod
    def intact(item):
        return all(Path(asset['path']).is_file() and _hash(asset['path']) == asset['sha256']
                   for asset in item['assets'].values())

    def episodes(self):
        return [json.loads(path.read_text()) for path in sorted(self.directory.glob('*.json'))]

    def plan(self, mode, prompt, *, group=None, limit=3, retry_of=None, scene_graph=None):
        if mode not in MODES:
            raise ValueError('Unknown visual mode')
        prior = self.get(retry_of) if retry_of else None
        graph = _scene_graph(scene_graph,prior)
        query = _tokens(prompt)
        ranked = []
        episodes = self.episodes()
        protected = {item['group'] for item in episodes if item['split'] in {'test','validation'}}
        protected_pixels = {identity for item in episodes if item['split'] in {'test','validation'}
                            for asset in item['assets'].values() for identity in _asset_identities(asset)}
        for item in episodes:
            # Held-out evidence must not silently become inference-time lessons.
            # An explicit exact-task retry remains allowed, and stays held out.
            if item['id'] != retry_of and (item['split'] in {'test','validation'}
                    or item['group'] in protected or any(_asset_identities(asset) & protected_pixels
                                                       for asset in item['assets'].values())):
                continue
            review = item['review']
            if item['mode'] != mode or not review or not self.intact(item):
                continue
            if group and item['group'] == group:
                continue
            overlap = len(query & _tokens(item['prompt']))
            if graph and item.get('scene_graph'):
                overlap += 4 * len(graph.signatures & SceneGraph.from_dict(item['scene_graph']).signatures)
            # Cross-task editing preservation is useful without subject overlap.
            generic = ({'preserve_scene','perform_requested_change'} if mode == 'editing' else
                       {'anchor_subject','show_function','bind_relationships'})
            policies = [p for p in review['policies'] if p in generic or overlap >= 2]
            if policies:
                ranked.append((overlap, item['created'], item, policies))
        ranked.sort(key=lambda entry:(entry[0], entry[1]), reverse=True)
        selected, policies = [], []
        for _, _, item, lessons in ranked[:max(0, min(limit, 5))]:
            selected.append(item['id'])
            for lesson in lessons:
                if lesson not in policies:
                    policies.append(lesson)
        # Free-form critique is evidence, never executable prompt instructions.
        instructions = [POLICIES[policy] for policy in policies]
        if retry_of:
            if prior['mode'] != mode or prior['prompt'] != prompt or not self.intact(prior) or not prior['review']:
                raise ValueError('Retry requires a reviewed matching task with intact evidence')
            correction = prior['review'].get('correction')
            if correction:
                # Explicit reviewer correction applies only to this exact retry.
                instructions.insert(0, correction)
                if retry_of not in selected:
                    selected.append(retry_of)
        parts = [prompt] + ([graph.render()] if graph else []) + instructions
        effective = '\n'.join(parts)
        return {'model_name':'Horizon 1', 'original_prompt':prompt,
                'effective_prompt':effective, 'lesson_episodes':selected, 'policies':policies,
                'weight_update':False,'scene_graph':graph.to_dict() if graph else None,
                'scene_graph_fingerprint':graph.fingerprint if graph else None}

    def replay_pairs(self):
        """Reviewed successful repairs only, excluding evaluation groups entirely."""
        items = {item['id']:item for item in self.episodes()}
        protected = {item['group'] for item in items.values() if item['split'] in {'test','validation'}}
        protected_pixels = {identity for item in items.values()
                            if item['split'] in {'test','validation'} for asset in item['assets'].values()
                            for identity in _asset_identities(asset)}
        result = []
        for item in items.values():
            parent = items.get(item['parent'])
            review = item['review']
            if not parent or not review or not parent['review']:
                continue
            if item['split'] not in {'train','experience'} or parent['split'] not in {'train','experience'}:
                continue
            if item['group'] in protected or parent['group'] in protected:
                continue
            if any(_asset_identities(asset) & protected_pixels for episode in (item,parent) for asset in episode['assets'].values()):
                continue
            if item['group'] != parent['group'] or item['prompt'] != parent['prompt'] or item['mode'] != parent['mode']:
                continue
            if review['reviewer'] == 'programmatic' or not (review['instruction'] and review['preservation']):
                continue
            if item.get('scene_graph') and not (review.get('relationship_assessment') or {}).get('complete'):
                continue
            if parent.get('scene_graph') and (not item.get('scene_graph') or
                    SceneGraph.from_dict(item['scene_graph']).fingerprint != SceneGraph.from_dict(parent['scene_graph']).fingerprint):
                continue
            if item['mode'] == 'creativity' and review['creativity'] is not True:
                continue
            old = parent['review']
            if old['instruction'] and old['preservation'] and (item['mode'] != 'creativity' or old['creativity']):
                continue
            if not self.intact(item) or not self.intact(parent):
                continue
            if item['assets']['output']['sha256'] == parent['assets']['output']['sha256']:
                continue
            if item['mode'] == 'editing' and item['assets']['source']['sha256'] != parent['assets']['source']['sha256']:
                continue
            if item.get('adherence_contract') or parent.get('adherence_contract'):
                from .horizon_adherence import AdherenceContract
                if not item.get('adherence_contract') or not parent.get('adherence_contract'):continue
                chosen_contract=AdherenceContract.from_dict(item['adherence_contract'])
                rejected_contract=AdherenceContract.from_dict(parent['adherence_contract'])
                if chosen_contract.fingerprint!=rejected_contract.fingerprint:continue
                if any(observations.get('reviewer','').replace('assistant_visual','agent_visual')!=entry['reviewer']
                       for observations,entry in [(review.get('adherence_observations') or {},review),
                                                  (old.get('adherence_observations') or {},old)]):continue
                try:
                    chosen=chosen_contract.assess(review.get('adherence_observations'),item['assets']['output']['path'])
                    rejected=rejected_contract.assess(old.get('adherence_observations'),parent['assets']['output']['path'])
                except ValueError:
                    continue
                if not chosen['instruction_success'] or rejected['status']!='fail':continue
            result.append({'prompt':item['prompt'], 'mode':item['mode'],
                           'source':item['assets'].get('source'), 'chosen':item['assets']['output'],
                           'rejected':parent['assets']['output'], 'chosen_episode':item['id'],
                           'rejected_episode':parent['id'], 'reviewer':review['reviewer'],
                           'scene_graph':item.get('scene_graph'),
                           'visual_contract':item.get('visual_contract'),
                           'adherence_contract':item.get('adherence_contract'),
                           'weight_update':False})
        return result
