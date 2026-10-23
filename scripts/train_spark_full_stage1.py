#!/usr/bin/env python3
"""Bounded full-weight Spark format SFT before on-policy Kira RL.

Adafactor state is factored and updates are materialized one tensor at a time
so a 24 GB unified-memory Mac can train all 4.112B base parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training_models/Spark-MLX-LLM'))
MODEL = ROOT / 'training_models/Spark-X2.5-4B'
DATA = ROOT / 'training_data/orchestrator_v15_spark9b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def update_all(model, gradients, optimizer, states, step):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    optimizer.state['step'] = mx.array(step, mx.uint64)
    updated = 0
    for key, gradient in tree_flatten(gradients):
        # Build only a temporary parameter view. Holding all old arrays while
        # replacing weights would double the 7.7 GiB checkpoint footprint.
        parameter = dict(tree_flatten(model.parameters()))[key]
        state = states.setdefault(key, {})
        if not state:
            optimizer.init_single(parameter, state)
        new_parameter = mx.stop_gradient(optimizer.apply_single(gradient, parameter, state))
        mx.eval(new_parameter, *state.values())
        model.update(tree_unflatten([(key, new_parameter)]))
        del parameter, new_parameter
        updated += 1
    mx.clear_cache()
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-gib', type=float, default=19)
    parser.add_argument('--learning-rate', type=float, default=2e-6)
    args = parser.parse_args()
    if not 1 <= args.steps <= 128 or not 12 <= args.memory_gib <= 19:
        raise ValueError('bounded laptop pilot limits exceeded')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory for an immutable run')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for split in ('train', 'valid', 'test'):
        if sha(args.data / f'{split}.jsonl') != manifest['files'][split]['sha256']:
            raise ValueError(f'{split} data hash mismatch')
    rows = [r for r in load_rows(args.data / 'train.jsonl')
            if r['source'] in {'kira_spark_tool_authored', 'BreakingWeb/breakingweb-results-v2'}
            and len(r['token_ids']) <= 384
            and r['family'] in {'browser_fill', 'browser_inspect', 'browser_index',
                                'browser_stale', 'read_file', 'list_dir', 'web_search',
                                'web_fetch', 'project_command', 'mcp_discover',
                                'executed_browser_action', 'write_file', 'open_file',
                                'mcp_draft', 'mcp_apply', 'native_docx', 'native_pdf',
                                'schedule_task'}]
    rows.sort(key=lambda r: (r['family'], r['id']))
    if not rows:
        raise ValueError('No short Kira contract examples')
    import mlx.core as mx
    from mlx import nn, optimizers
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.utils import save_model
    from spark_mlx_llm.loader import load
    mx.set_cache_limit(128*1024**2)
    mx.set_memory_limit(int(args.memory_gib*1024**3))
    if not mx.metal.is_available():
        raise RuntimeError('Metal GPU required')
    args.output.mkdir(parents=True)
    model, tokenizer = load(args.model, lazy=True)
    for layer in model.layers:
        grad_checkpoint(layer)
    model.train()
    optimizer = optimizers.Adafactor(learning_rate=args.learning_rate,
                                   relative_step=False, scale_parameter=False,
                                   beta_1=None, weight_decay=0.0)
    states = {}
    def loss_fn(m, row):
        ids = mx.array([row['token_ids']], mx.int32)
        start = row['answer_start']
        logits = m(ids[:, :-1])
        losses = nn.losses.cross_entropy(logits, ids[:, 1:])
        return mx.mean(losses[:, start-1:])
    value_and_grad = nn.value_and_grad(model, loss_fn)
    started = time.monotonic()
    metrics = args.output / 'metrics.jsonl'
    selected = []
    families = sorted({r['family'] for r in rows})
    for step in range(1, args.steps+1):
        family = families[(step-1) % len(families)]
        subset = [r for r in rows if r['family'] == family]
        # Across cycles, rotate groups rather than replaying the same authored
        # scenario with only the family name changed.
        row = subset[((step-1) // len(families) * 11) % len(subset)]
        loss, gradients = value_and_grad(model, row)
        mx.eval(loss, gradients)
        loss_value = float(loss.item())
        del loss
        tensors = update_all(model, gradients, optimizer, states, step)
        del gradients
        post_loss = loss_fn(model, row)
        mx.eval(post_loss)
        post_loss_value = float(post_loss.item())
        del post_loss
        event = {'step': step, 'row_id': row['id'], 'family': family,
                 'loss': loss_value, 'post_update_loss': post_loss_value,
                 'updated_tensors': tensors,
                 'active_memory_gib': mx.get_active_memory()/1024**3,
                 'peak_memory_gib': mx.get_peak_memory()/1024**3,
                 'elapsed_seconds': round(time.monotonic()-started, 2)}
        with metrics.open('a') as handle:
            handle.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
        selected.append(row['id'])
        if post_loss_value > loss_value * 1.05:
            (args.output / 'report.json').write_text(json.dumps({
                'stage': 'rejected_regressing_update', 'step': step,
                'loss_before': loss_value, 'loss_after': post_loss_value,
                'checkpoint_saved': False, 'promoted': False}, indent=2))
            raise RuntimeError('post-update loss regressed by more than 5%; checkpoint rejected')
    checkpoint = args.output / 'checkpoint'
    save_model(checkpoint, model)
    for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'special_tokens_map.json',
                 'chat_template.jinja', 'vocab.json', 'merges.txt'):
        shutil.copy2(args.model / name, checkpoint / name)
    report = {'stage': 'full_weight_format_sft', 'completed_steps': args.steps,
              'full_weight_training': True, 'optimizer': 'factored Adafactor without first moment',
              'model_source': 'XHToken/Spark-X2.5-4B', 'initial_checkpoint': str(args.model),
              'data_manifest_sha256': sha(args.data / 'manifest.json'),
              'selected_row_ids': selected, 'checkpoint': str(checkpoint),
              'elapsed_seconds': round(time.monotonic()-started, 2),
              'rl_updates': 0, 'experts_trained': 0, 'promoted': False}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
