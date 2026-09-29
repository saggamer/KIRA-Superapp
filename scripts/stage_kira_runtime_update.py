"""Stage an incremental Hub update without copying weights or private data.

The existing weight checksums are retained. Publishing remains an explicit step.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'saggamer/KIRA-LIVE-1'
RUNTIME = (
    'audio_devices', 'audio_pipeline', 'checkpoint_bundle', 'mlx_layers',
    'mlx_listener', 'mlx_thinker', 'mlx_voice', 'native_session', 'policy',
    'session', 'workflow_memory', 'action_contracts', 'expert_profiles',
    'expert_surgery', 'response_repetition', 'event_dispatch',
)


def main():
    revision = HfApi().model_info(REPO).sha
    stage = Path(tempfile.mkdtemp(prefix='kira-runtime-release-'))
    for name in ('README.md', 'checksums.json', 'bundle_manifest.json'):
        shutil.copy2(hf_hub_download(REPO, name, revision=revision), stage / name)
    checksums = json.loads((stage / 'checksums.json').read_text())
    def include(source, relative):
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in RUNTIME:
        include(ROOT / f'kira_live/{name}.py', f'kira_live/{name}.py')
    package = Path('training_runs/kira_live_emotion_v1/KiraLiveEmotionVerified.mlpackage')
    for source in sorted((ROOT / package).rglob('*')):
        if source.is_file():
            include(source, source.relative_to(ROOT))
    include(ROOT / 'docs/KIRA_LIVE_VOICE_LATENCY_UPDATE_2026_09_29.md', 'VOICE_RUNTIME_UPDATE.md')
    dependencies = ('mlx', 'mlx-lm', 'mlx-audio', 'transformers', 'numpy',
                    'sounddevice', 'soundfile', 'huggingface-hub', 'safetensors',
                    'ten-vad', 'scipy', 'pywebrtc-audio', 'torch', 'coremltools')
    (stage / 'requirements.txt').write_text(''.join(
        f'{name}=={importlib.metadata.version(name)}\n' for name in dependencies))
    readme = (stage / 'README.md').read_text().replace(
        '4. Launch `.venv-kira-live/bin/python interface.py`, select **Talk with KIRA Live 1**,',
        '4. Launch `RUN_KIRA_OS.command`, select **Talk with KIRA Live 1**,')
    readme += '\n## September 29 runtime update\n\n'
    readme += ('Updated Live conversation history, repetition control, language selection,\n'
               'THE TREE action contracts, nonblocking audio events and latency diagnostics.\n'
               'The Talker uses in-memory 8-bit linear layers by default; set\n'
               '`KIRA_LIVE_TALKER_BITS=0` for original BF16 precision. Saved weights\n'
               'are unchanged. The verified CoreML emotion package is included.\n'
               'See [measured timings and limitations](VOICE_RUNTIME_UPDATE.md).\n'
               'This is a runtime update, not a new trained model or quality promotion.\n')
    (stage / 'README.md').write_text(readme)
    manifest = json.loads((stage / 'bundle_manifest.json').read_text())
    manifest['runtime_update'] = '2026-09-29'
    manifest['latest_runtime_fixes'] = list(dict.fromkeys(
        manifest.get('latest_runtime_fixes', []) + [
            'live_conversation_history', 'reply_repetition_control', 'explicit_asr_language',
            'live_tree_action_contracts', 'in_memory_int8_talker', 'nonblocking_audio_events',
            'safe_audio_device_defaults', 'complete_playback_drain', 'turn_latency_metrics']))
    (stage / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for source in sorted(stage.rglob('*')):
        if source.is_file() and source.name != 'checksums.json':
            checksums[str(source.relative_to(stage))] = hashlib.sha256(source.read_bytes()).hexdigest()
    (stage / 'checksums.json').write_text(json.dumps(checksums, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'stage': str(stage), 'parent_revision': revision,
                      'files': sum(p.is_file() for p in stage.rglob('*'))}, indent=2))


if __name__ == '__main__':
    main()
