"""Stage a complete, private-data-free three-weight KIRA Live release.

Runs locally; use hf upload on the printed staging directory after verification.
Only immutable public donor files and accepted KIRA checkpoints are included.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kira_live.weights import resolve_composite_weights

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    stage=Path(tempfile.mkdtemp(prefix='kira-live-complete-'))
    def include(source,relative):
        target=stage/relative; target.parent.mkdir(parents=True,exist_ok=True)
        try: os.link(source.resolve(),target)
        except OSError: shutil.copy2(source,target)
    weights=resolve_composite_weights()
    manifest=json.loads((ROOT/'kira_live/model_manifest.json').read_text())
    components={}
    for island in weights.by_role():
        for source in sorted(island.snapshot.rglob('*')):
            if not source.is_file(): continue
            rel=source.relative_to(island.snapshot)
            # No model cards containing unsupported claims; credit original
            # publishers separately. Only known runtime assets are included.
            if source.suffix not in {'.json','.safetensors','.txt','.model','.jinja'}: continue
            include(source,Path('weights')/island.spec.role/rel)
        components[island.spec.role]={'path':f'weights/{island.spec.role}',
            'model_id':island.spec.model_id,'revision':island.spec.revision,'license':'apache-2.0'}
    for name in ['thinker_ple_moe','live_emotion_stage1','live_semantic_bridge']:
        entry=manifest['training_status'][name]
        if not str(entry.get('decision','')).startswith('accepted'): raise RuntimeError(f'Unaccepted: {name}')
        rel=Path(entry['checkpoint'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe checkpoint path')
        include(ROOT/rel,rel)
    include(ROOT/'training_runs/kira_live_emotion_v1/feature_normalization.npz',Path('training_runs/kira_live_emotion_v1/feature_normalization.npz'))
    # Copy mutable code instead of linking it, to freeze the staged release.
    for source in sorted((ROOT/'kira_live').glob('*.py')):
        if source.name.startswith(('experimental_','sparse_student')): continue
        dest=stage/'kira_live'/source.name; dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source,dest)
    shutil.copy2(ROOT/'kira_live/model_manifest.json',stage/'kira_live/model_manifest.json')
    shutil.copy2(ROOT/'LICENSE',stage/'LICENSE')
    shutil.copy2(ROOT/'docs/KIRA_LIVE_1_MODEL_CARD.md',stage/'README.md')
    shutil.copy2(ROOT/'scripts/run_kira_live_bundle.py',stage/'run_kira_live.py')
    requirements=[]
    for name in ['mlx','mlx-lm','mlx-audio','transformers','numpy','sounddevice','soundfile','huggingface-hub','safetensors','ten-vad','scipy','pywebrtc-audio','torch','coremltools']:
        requirements.append(f'{name}=={importlib.metadata.version(name)}')
    (stage/'requirements.txt').write_text('\n'.join(requirements)+'\n')
    (stage/'THIRD_PARTY_NOTICES.md').write_text('# Third-party notices\n\nQwen3.5 and Qwen3-ASR: Copyright Alibaba/Qwen contributors.\nQwen3-TTS MLX conversion: Qwen and MLX community contributors.\nAll three weight sources declare Apache-2.0. The license is included as LICENSE.\nOriginal publisher URLs and exact revisions are recorded in bundle_manifest.json.\nThese weights retain their original attribution; KIRA does not claim authorship of the donor weights.\nTEN-VAD is an installed dependency, not a rehosted weight asset; its own license applies.\n')
    metadata={'format':'kira_live_three_weight_bundle_v1','project':'KIRA Live 1',
              'research_preview':True,'tensor_fusion':False,'joint_end_to_end_training':False,
              'weights':components,'runtime':'kira_live.native_session.KiraNativeModelBundle',
              'latest_runtime_fixes':['identity_guard','sentence_context_streaming','overlapped_playback',
                 'thread_bound_metal_generation','noise_and_speech_gate','playback_echo_filter',
                 'permission_aware_tree_hooks','bundled_local_weight_resolution',
                 'lossless_multilingual_token_mapping','webrtc_aec3_playback_tail_and_self_speech_guard'],
              'private_data_included':False}
    (stage/'bundle_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    checksums={str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    (stage/'checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    print(json.dumps({'stage':str(stage),'files':len(checksums)+1,'bytes':sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())},indent=2),flush=True)

if __name__=='__main__': main()
