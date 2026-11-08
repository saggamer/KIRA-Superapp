"""Pinned app assets; importing this catalog never initializes Metal."""
import hashlib
import json
import os
from pathlib import Path

BUNDLE_REPO = 'saggamer/horizon-1'
BUNDLE_REVISION = '027eb0f1e5511ccc8aa19fb17a0bea3424f973f5'
BASE_REPO = 'AITRADER/FLUX2-klein-base-4B-mlx-4bit'
BASE_REVISION = 'e04a5bc9307320a601ebb0047e3f47ec7dc4d432'
BASE_HASHES = {
    'text_encoder/0.safetensors':'7670e08db4e96efe705b63f001c3cf589796eff9522b7f7f9419b429c123ec11',
    'text_encoder/1.safetensors':'19a6fb703c218ba0f8b57807cdf74ff3cc8c06bdb01dcd2632bff9fd373e61e5',
    'tokenizer/tokenizer.json':'be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506',
    'transformer/0.safetensors':'c295458a2743eea0889bec2eec2aa9e326e8a4887421b659edb5aa1d0bc01a3f',
    'transformer/1.safetensors':'cb0fd80f39a645599b2fd0a9280c750fdbd188d7f0079d2da2519681df8bcd0c',
    'vae/0.safetensors':'23d5aa73c9203f50490508332813229673d818d315e6119b171d9c2df07dbdbc',
}


def paths(app_root):
    root=Path(app_root)
    return (Path(os.environ.get('KIRA_HORIZON_BUNDLE',root/'models/horizon_1_mlx')).expanduser(),
            Path(os.environ.get('KIRA_HORIZON_BASE',root/'models/horizon_1_base_4bit')).expanduser())


def _hash(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def validate_bundle(path, *, hashes=True):
    path=Path(path)
    required=['threshold.json','horizon_config.json','weights/editing_lora.safetensors',
              'weights/instruction_grounder.safetensors','weights/semantic_bridge.safetensors',
              'weights/region_embedding.safetensors','weights/creativity_lora.safetensors',
              'weights/creativity_bridge.safetensors']
    manifest=json.loads((path/'manifest.json').read_text())
    if not all(name in manifest for name in required):raise ValueError('Incomplete Horizon bundle manifest')
    for name,record in manifest.items():
        target=(path/name).resolve()
        target.relative_to(path.resolve())
        if not target.is_file() or target.stat().st_size!=record['bytes']:
            raise ValueError('Incomplete Horizon bundle: '+name)
        if hashes and _hash(target)!=record['sha256']:raise ValueError('Horizon file hash mismatch: '+name)


def validate_base(path, *, hashes=True):
    path=Path(path)
    for name in ['tokenizer/chat_template.jinja','tokenizer/tokenizer_config.json']:
        if not (path/name).is_file():raise ValueError('Incomplete Horizon tokenizer: '+name)
    for name,digest in BASE_HASHES.items():
        target=path/name
        if not target.is_file() or not target.stat().st_size:raise ValueError('Incomplete Horizon base: '+name)
        if hashes and _hash(target)!=digest:raise ValueError('Horizon base hash mismatch: '+name)
    for component in ['transformer','text_encoder','vae']:
        index=json.loads((path/component/'model.safetensors.index.json').read_text())
        files=set(index['weight_map'].values())
        if not files or any(not (path/component/file).is_file() for file in files):
            raise ValueError('Incomplete Horizon base index: '+component)
