"""Preserve complete user instructions at the byte-token conditioning boundary."""
from dataclasses import dataclass,asdict
import hashlib


class InstructionBudgetError(ValueError):
    pass


@dataclass(frozen=True)
class EncodedInstruction:
    token_ids: tuple
    prompt_sha256: str
    prompt_bytes: int
    byte_limit: int
    total_context_tokens: int

    def evidence(self):
        return {**{k:v for k,v in asdict(self).items() if k!='token_ids'},
                'complete_instruction_preserved':True,'truncated':False}


def encode_instruction(prompt, *, max_prompt_bytes, max_context_tokens=None, visual_tokens=0):
    """UTF-8 vocabulary: bytes 0..255, BOS 256, EOS 257; never truncate.

    This is for Horizon's compact byte-token research core. The native FLUX/Qwen
    path uses its own tokenizer/template budget validator.
    """
    if not isinstance(prompt,str) or not prompt.strip():
        raise ValueError('Provide a nonempty image instruction')
    if type(max_prompt_bytes) is not int or max_prompt_bytes<1:
        raise ValueError('Checkpoint prompt-byte limit must be a positive integer')
    if type(visual_tokens) is not int or visual_tokens<0:
        raise ValueError('Visual token count must be a nonnegative integer')
    if max_context_tokens is not None and (type(max_context_tokens) is not int or max_context_tokens<1):
        raise ValueError('Context limit must be a positive integer')
    data=prompt.encode('utf8');total=len(data)+2+visual_tokens
    if len(data)>max_prompt_bytes:
        raise InstructionBudgetError(f'The complete instruction needs {len(data)} UTF-8 bytes; this checkpoint supports {max_prompt_bytes}. No instruction was truncated. Use an explicitly shorter instruction or a checkpoint trained for the longer context.')
    if max_context_tokens is not None and total>max_context_tokens:
        raise InstructionBudgetError(f'The complete instruction and images need {total} context tokens, exceeding {max_context_tokens}. No instruction was truncated.')
    return EncodedInstruction((256,*data,257),hashlib.sha256(data).hexdigest(),len(data),max_prompt_bytes,total)
