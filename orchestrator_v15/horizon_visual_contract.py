"""Explicit visual constraints and source-pixel protection for Horizon inference.

Contracts are supplied scene plans, not automatic object detection. They change
conditioning and compositing; they do not update neural weights.
"""
from dataclasses import dataclass
import hashlib
import json
import math

from .horizon_relationships import SceneGraph


ROLE_CUES = {
    'pouring_spout': 'a projecting hollow outlet with an open tip used for pouring',
    'handle': 'an attached curved grip with open space for fingers',
    'window': 'a clear viewing opening through which the interior is visible',
    'clock_face': 'a visible clock dial with hands',
}
MATERIAL_CUES = {
    'transparent glass': 'clear see-through glass, with the background visible through its walls',
    'clear glass': 'clear see-through glass, with the background visible through its walls',
    'transparent': 'see-through, with the background visible through it',
}


@dataclass(frozen=True)
class VisualContract:
    brief: str
    graph: SceneGraph
    part_roles: tuple = ()
    mode: str = 'creativity'

    def __post_init__(self):
        if self.mode not in {'editing', 'creativity'} or not isinstance(self.brief, str) or not self.brief.strip():
            raise ValueError('A contract needs a brief and explicit visual mode')
        if len(self.brief) > 5000 or not isinstance(self.graph, SceneGraph):
            raise ValueError('Invalid contract brief or scene graph')
        nodes = {node.id: node for node in self.graph.entities}
        seen = set()
        for part, role in self.part_roles:
            if part in seen or part not in nodes or nodes[part].kind != 'part' or role not in ROLE_CUES:
                raise ValueError('Each functional role needs one existing owned part and a supported role')
            seen.add(part)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {'brief', 'scene_graph', 'part_roles', 'mode'}:
            raise ValueError('Unknown visual contract fields')
        roles = value.get('part_roles', {})
        if not isinstance(roles, dict):
            raise ValueError('Part roles must map explicit part IDs to roles')
        return cls(value.get('brief'), SceneGraph.from_dict(value.get('scene_graph')),
                   tuple(sorted(roles.items())), value.get('mode', 'creativity'))

    def to_dict(self):
        return {'brief': self.brief, 'scene_graph': self.graph.to_dict(),
                'part_roles': dict(self.part_roles), 'mode': self.mode}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def compile(self):
        """Put required function/material bindings before decorative wording."""
        nodes = {node.id: node for node in self.graph.entities}
        def name(identifier):
            node = nodes[identifier]
            return (nodes[node.owner].label + "'s " if node.owner else '') + node.label
        lines = ['Required physical scene:']
        for part, role in self.part_roles:
            owner = nodes[nodes[part].owner].label
            lines.append(f"{name(part)} serves as the {owner}'s {role.replace('_', ' ')}: {ROLE_CUES[role]}.")
        for node in self.graph.entities:
            material = MATERIAL_CUES.get((node.material or '').strip().casefold(), node.material)
            line = f'Exactly {node.count} {name(node.id)}'
            if material:
                line += f'; this object is {material}'
            if node.box:
                x0, y0, x1, y1 = node.box
                horizontal = 'left' if x1 <= .5 else 'right' if x0 >= .5 else 'center'
                vertical = 'upper' if y1 <= .5 else 'lower' if y0 >= .5 else 'middle'
                line += f'; in the {vertical} {horizontal} area'
            lines.append(line + '.')
        for edge in self.graph.relations:
            lines.append(name(edge.subject) + ' ' + edge.predicate.replace('_', ' ') + ' '
                         + ' and '.join(name(target) for target in edge.targets) + '.')
        if self.mode == 'editing':
            lines.append('Perform the requested change while preserving unrelated objects, clothing, lighting and camera viewpoint.')
        lines.append('Requested image: ' + self.brief.strip())
        return '\n'.join(lines)

    def assess(self, *, role_observations=None, **graph_observations):
        result = self.graph.assess(**graph_observations)
        supplied = role_observations or {}
        if set(supplied) - {part for part, _ in self.part_roles}:
            raise ValueError('Role observation refers to an unknown functional part')
        for part, _ in self.part_roles:
            status = supplied.get(part, 'unknown')
            if status not in {'pass', 'fail', 'unknown'}:
                raise ValueError('Function observations must be pass, fail or unknown')
            result['checks']['function:' + part] = status
        result['complete'] = all(value == 'pass' for value in result['checks'].values())
        return result


def edit_region_instruction(mask):
    """Describe only supplied mask geometry, without inventing object labels."""
    if mask.mode not in {'L','1'}:
        raise ValueError('Edit-region descriptions require a grayscale mask')
    bounds = mask.convert('L').getbbox()
    if bounds is None:
        raise ValueError('An edit mask needs a nonempty allowed region')
    x0,y0,x1,y1 = bounds
    cx,cy = (x0+x1)/(2*mask.width),(y0+y1)/(2*mask.height)
    horizontal = 'left' if cx < .4 else 'right' if cx > .6 else 'center'
    vertical = 'upper' if cy < .35 else 'lower' if cy > .65 else 'middle'
    return (f'Apply the requested change within the selected {vertical} {horizontal} region. '
            'Preserve the rest of the scene, including unrelated people and objects.')


def condition_edit_prompt(prompt, mask=None, *, regional=False):
    """Protection must not silently narrow broad instructions such as 'all'."""
    if not regional:
        return prompt
    if mask is None:
        raise ValueError('Regional conditioning requires an explicit edit mask')
    return prompt + '\n' + edit_region_instruction(mask)


def preserve_outside_mask(source, generated, mask, *, feather=0):
    """Composite only inside an explicit mask; never expand its allowed support.

    White allows changes; black preserves original bytes. Feathering is inward
    only. The caller supplies a mask from the source, never a target-derived mask.
    """
    import numpy as np
    from PIL import Image, ImageFilter
    if source.size != generated.size or source.size != mask.size:
        raise ValueError('Source, generated output and explicit mask must share dimensions')
    if mask.mode not in {'L', '1'}:
        raise ValueError('Use a grayscale edit mask, with white allowed and black protected')
    if type(feather) not in {int, float} or not math.isfinite(feather) or not 0 <= feather <= 32:
        raise ValueError('Feather radius must be finite and between zero and 32 pixels')
    allowed = np.asarray(mask.convert('L')) > 0
    if not allowed.any():
        raise ValueError('An empty edit region cannot perform a requested edit')
    alpha_image = mask.convert('L')
    if feather:
        alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(feather))
    alpha = np.asarray(alpha_image).astype('float32') / 255
    alpha[~allowed] = 0
    original = np.asarray(source.convert('RGB'))
    candidate = np.asarray(generated.convert('RGB'))
    merged = np.rint(original * (1 - alpha[..., None]) + candidate * alpha[..., None]).clip(0, 255).astype('uint8')
    merged[~allowed] = original[~allowed]
    protected = ~allowed
    evidence = {'mask_supplied': True, 'mask_white_means': 'allowed edit',
                'protected_fraction': float(protected.mean()),
                'protected_pixels_exact': bool(np.array_equal(merged[protected], original[protected])),
                'mean_allowed_change': float(abs(merged.astype('float32') - original)[allowed].mean() / 255),
                'instruction_success': None, 'feather_pixels': feather}
    return Image.fromarray(merged), evidence
