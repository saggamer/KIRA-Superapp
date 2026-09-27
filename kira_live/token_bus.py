from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral


def remap_token_ids(
    token_ids: Sequence[int],
    source_tokenizer,
    target_tokenizer,
) -> tuple[int, ...]:
    """Translate byte-BPE pieces without decoding/re-prompting the transcript.

    The two Qwen vocabularies have different merges. Missing merged pieces
    (including partial UTF-8 sequences) must be split into smaller target
    pieces, not converted from None or decoded independently as Unicode.
    """
    source = [int(item) for item in token_ids]
    pieces = source_tokenizer.convert_ids_to_tokens(source)
    if isinstance(pieces, str):
        pieces = [pieces]
    else:
        try:
            pieces = list(pieces)
        except TypeError as exc:
            raise ValueError("Listener token bus produced incomplete source pieces.") from exc
    if len(pieces) != len(source):
        raise ValueError("Listener token bus produced incomplete source pieces.")
    mapped = target_tokenizer.convert_tokens_to_ids(pieces)
    if isinstance(mapped, int):
        mapped = [mapped]
    if not isinstance(mapped, (list, tuple)) or len(mapped) != len(source):
        raise ValueError("Listener/Thinker token bus produced an incomplete mapping.")
    unknown_id = getattr(target_tokenizer, "unk_token_id", None)
    def valid(value):
        return isinstance(value, Integral) and value >= 0 and value != unknown_id

    result = []
    split_cache = {}
    for piece, target_id in zip(pieces, mapped):
        if valid(target_id):
            result.append(int(target_id))
            continue
        if not isinstance(piece, str) or not piece or piece.startswith("<|"):
            raise ValueError("Listener semantic token is unavailable in the Thinker vocabulary.")
        if piece not in split_cache:
            split = []
            start = 0
            while start < len(piece):
                for end in range(len(piece), start, -1):
                    candidate = target_tokenizer.convert_tokens_to_ids(piece[start:end])
                    if valid(candidate):
                        split.append(int(candidate))
                        start = end
                        break
                else:
                    raise ValueError("Listener byte piece cannot be represented in the Thinker vocabulary.")
            split_cache[piece] = tuple(split)
        result.extend(split_cache[piece])
    return tuple(result)
