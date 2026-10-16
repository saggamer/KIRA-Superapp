"""Resolve the finished research checkpoint without implying release approval."""
import json
from pathlib import Path


def research_checkpoint(run):
    run=Path(run)
    status=json.loads((run/'status.json').read_text())
    stage=status.get('stage')
    if stage=='pilot_complete':
        name='retained_candidate.safetensors'
    elif stage=='pilot_rejected':
        name='sft.safetensors'
    else:
        raise RuntimeError(f'training has not reached a saved candidate: {stage}')
    checkpoint=run/name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint,stage
