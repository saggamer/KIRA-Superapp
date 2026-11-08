"""BAGEL-inspired reference attention layout; no neural or pixel inference."""
from dataclasses import dataclass

TOKEN_KINDS = {'text','semantic_vision','clean_latent','noisy_latent','relation'}


@dataclass(frozen=True)
class ContextSpan:
    kind: str
    length: int
    block: int
    sample: int = 0
    image_id: str | None = None


def expand_layout(spans, max_tokens=2048):
    tokens, blocks = [], {}
    last = {}
    for span in spans:
        if span.kind not in TOKEN_KINDS or type(span.length) is not int or span.length < 1:
            raise ValueError('Invalid context modality or token count')
        if type(span.block) is not int or span.block < 0 or type(span.sample) is not int or span.sample < 0:
            raise ValueError('Context block and sample IDs must be nonnegative integers')
        if span.kind in {'semantic_vision','clean_latent','noisy_latent'} and (
                not isinstance(span.image_id,str) or not 1 <= len(span.image_id) <= 128):
            raise ValueError('Visual spans need explicit image identity')
        if span.kind in {'text','relation'} and span.image_id is not None:
            raise ValueError('Text/relation spans cannot impersonate image context')
        key = (span.sample,span.block)
        if key in blocks or span.block <= last.get(span.sample,-1):
            raise ValueError('Each sample needs strictly increasing, unique context blocks')
        blocks[key] = span.kind
        last[span.sample] = span.block
        for position in range(span.length):
            tokens.append((span,position))
        if len(tokens) > max_tokens:
            raise ValueError('Reference context exceeds its quadratic-attention token limit')
    if not tokens:
        raise ValueError('Empty multimodal context')
    return tokens


def generalized_attention_mask(spans, max_tokens=2048):
    tokens = expand_layout(spans,max_tokens)
    allowed = []
    for query,position in tokens:
        row = []
        for key,other_position in tokens:
            same_block = query.block == key.block
            visible = query.sample == key.sample and (
                key.block < query.block or same_block and
                (query.kind != 'text' or other_position <= position))
            # Noisy targets are not persistent conditioning for subsequent blocks.
            if key.kind == 'noisy_latent' and not same_block:
                visible = False
            # A noised image must never see its own clean/semantic target features.
            if query.kind == 'noisy_latent' and query.image_id == key.image_id and key.kind in {'clean_latent','semantic_vision'}:
                visible = False
            row.append(visible)
        allowed.append(row)
    return allowed
