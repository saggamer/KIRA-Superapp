"""All-requirement visual instruction reviews tied to exact output bytes.

Requirements and observations are explicit. This module is not an object detector
or an automatic semantic judge, and does not rewrite the user's instruction.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,re

CATEGORIES={'action','object','count','attribute','relation','forbidden','preservation','text'}


@dataclass(frozen=True)
class AdherenceRequirement:
    id: str
    category: str
    assertion: str


@dataclass(frozen=True)
class AdherenceContract:
    prompt: str
    mode: str
    requirements: tuple
    source_sha256: str | None = None

    def __post_init__(self):
        if self.mode not in {'editing','creativity'} or not isinstance(self.prompt,str) or not self.prompt.strip():
            raise ValueError('Provide the exact instruction and visual mode')
        if not 1<=len(self.requirements)<=32:
            raise ValueError('Use one to 32 explicit requirements')
        ids=set()
        for r in self.requirements:
            if not isinstance(r,AdherenceRequirement) or not isinstance(r.id,str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,47}',r.id) or r.id in ids:
                raise ValueError('Requirements need unique short identifiers')
            if r.category not in CATEGORIES or not isinstance(r.assertion,str) or not r.assertion.strip() or len(r.assertion)>1000:
                raise ValueError('Invalid requirement category or assertion')
            ids.add(r.id)
        if self.mode=='editing' and (not isinstance(self.source_sha256,str) or not re.fullmatch('[0-9a-f]{64}',self.source_sha256)):
            raise ValueError('Editing contracts must identify the exact source file')
        if self.mode=='creativity' and self.source_sha256 is not None:
            raise ValueError('Generation contracts cannot identify an edit source')

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or set(value)-{'prompt','mode','requirements','source_sha256'}:
            raise ValueError('Unknown adherence contract fields')
        try:
            return cls(value['prompt'],value['mode'],tuple(AdherenceRequirement(**r) for r in value['requirements']),value.get('source_sha256'))
        except (KeyError,TypeError) as exc:
            raise ValueError('Invalid adherence contract') from exc

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def validate_request(self,prompt,mode,source=None):
        if prompt!=self.prompt or mode!=self.mode:
            raise ValueError('The request must match its complete instruction contract')
        if self.mode=='editing':
            if source is None:raise ValueError('An edit source is required')
            with Path(source).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=self.source_sha256:raise ValueError('The source differs from the instruction contract')
        elif source is not None:raise ValueError('Unexpected source for generation contract')

    def assess(self,observations,output):
        if not isinstance(observations,dict) or set(observations)-{'contract_sha256','output_sha256','reviewer','checks','aesthetic_score'}:
            raise ValueError('Invalid adherence observations')
        with Path(output).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if observations.get('contract_sha256')!=self.fingerprint or observations.get('output_sha256')!=actual:
            raise ValueError('Review must match the exact instruction contract and output bytes')
        reviewer=observations.get('reviewer')
        if reviewer not in {'human','assistant_visual','agent_visual','programmatic'}:raise ValueError('Unknown reviewer provenance')
        supplied=observations.get('checks',{})
        if not isinstance(supplied,dict) or set(supplied)-{r.id for r in self.requirements}:
            raise ValueError('Review contains unknown requirement identifiers')
        result={}
        for requirement in self.requirements:
            item=supplied.get(requirement.id,{'status':'unknown','evidence':''})
            if not isinstance(item,dict) or set(item)-{'status','evidence'} or item.get('status') not in {'pass','fail','unknown'}:
                raise ValueError('Checks need explicit pass/fail/unknown observations')
            if not isinstance(item.get('evidence'),str) or (item['status']!='unknown' and not item['evidence'].strip()):
                raise ValueError('A positive or negative judgment needs a concrete observation')
            if reviewer=='programmatic' and item['status']=='pass':
                raise ValueError('Programmatic pixel/similarity checks cannot certify semantic adherence')
            result[requirement.id]={'category':requirement.category,'assertion':requirement.assertion,**item}
        failures=[key for key,value in result.items() if value['status']=='fail']
        unknown=[key for key,value in result.items() if value['status']=='unknown']
        return {'contract_sha256':self.fingerprint,'output_sha256':actual,'reviewer':reviewer,'checks':result,
                'instruction_success':not failures and not unknown,'failures':failures,'unknown':unknown,
                'status':'fail' if failures else 'unknown' if unknown else 'pass',
                'aesthetic_score_used_for_adherence':False}
