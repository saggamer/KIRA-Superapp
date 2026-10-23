#!/usr/bin/env python3
"""One bounded on-policy, full-weight Spark RL update with typed Kira rewards.

The update is skipped if sampled rewards have no variance. The checkpoint is
saved only when the update preserves three held-out exact-answer decisions.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training_models/Spark-MLX-LLM'))
from orchestrator_v15.contracts import parse_block, reward
from orchestrator_v15.spark_tools import native_browser_call_to_block
from scripts.train_spark_full_stage1 import load_rows, sha, update_all


def structured_reward(text, verifier, lexical_shaping=False):
    exact = reward(text, verifier)
    if exact > 0:
        return 1.0
    if verifier['kind'] != 'block':
        return -1.0
    parsed = parse_block(text)
    if parsed is None:
        if lexical_shaping:
            upper = text.upper()
            tool = verifier['tool']
            bonus = 0.15 if tool in upper else 0.0
            bonus += 0.15 if f'[{tool}]' in upper else 0.0
            bonus += 0.10 if f'[/{tool}]' in upper else 0.0
            bonus += sum(0.07 for key, value in verifier['fields'].items()
                         if f'{key}: {value}'.upper() in upper)
            return min(-0.2, -1.0 + bonus)
        return -1.0
    tool, fields = parsed
    if tool != verifier['tool']:
        return -0.75
    wanted = verifier['fields']
    matched = sum(fields.get(key) == value for key, value in wanted.items())
    unexpected = len(set(fields) - set(wanted))
    return min(0.75, -0.25 + matched/max(1, len(wanted)) - 0.1*unexpected)


def sample(model, tokenizer, prompt, max_tokens, sampler):
    from mlx_lm import stream_generate
    pieces = []; ids = []
    for response in stream_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=max_tokens, sampler=sampler):
        pieces.append(response.text)
        if response.finish_reason is None or response.finish_reason in {'stop', 'length'}:
            ids.append(int(response.token))
    return ''.join(pieces), ids


def select_candidates(rows, max_prompt_tokens=320):
    """Put short browser contracts first without excluding all browser rows."""
    order = ['browser_inspect', 'browser_fill', 'browser_index', 'read_file',
             'permission', 'truth', 'arithmetic']
    candidates = [r for r in rows
                  if r['answer_start'] <= max_prompt_tokens
                  and r['verifier']['kind'] in {'block', 'exact', 'integer'}
                  and r.get('family') in order]
    candidates.sort(key=lambda r: (order.index(r['family']), r['id']))
    first = []
    used = set()
    for family in order:
        row = next((item for item in candidates if item['family'] == family), None)
        if row is not None:
            first.append(row); used.add(row['id'])
    return first + [row for row in candidates if row['id'] not in used]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'training_data/orchestrator_v15_spark9b_r2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=int, default=8)
    parser.add_argument('--rollouts', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--temperature', type=float, default=1.15)
    parser.add_argument('--lexical-shaping', action='store_true')
    parser.add_argument('--force-browser-prefix', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('RL trial output must be new')
    if not 1 <= args.groups <= 32 or not 2 <= args.rollouts <= 8 or not 0.5 <= args.temperature <= 2.0:
        raise ValueError('bounded RL pilot limits exceeded')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for split in ('train', 'valid', 'test'):
        if sha(args.data / f'{split}.jsonl') != manifest['files'][split]['sha256']:
            raise ValueError('data hash mismatch')
    # The r6 browser examples begin at 293 tokens; the previous 280-token
    # cutoff silently excluded every browser row from a browser RL trial.
    # Keep enough room for a 64-token rollout (and an optional short prefix)
    # under the 384-token full-gradient bound.
    candidates = select_candidates(load_rows(args.data / 'rl_train.jsonl'))
    import mlx.core as mx
    from mlx import nn, optimizers
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.utils import save_model
    from spark_mlx_llm.loader import load
    mx.random.seed(1509)
    mx.set_cache_limit(128*1024**2)
    mx.set_memory_limit(19*1024**3)
    args.output.mkdir(parents=True)
    model, tokenizer = load(args.model, lazy=True)
    model.eval()
    sampler = make_sampler(temp=args.temperature, top_p=0.95)
    validation = []
    for family in ('truth', 'permission', 'arithmetic'):
        row = next((r for r in load_rows(args.data / 'valid.jsonl')
                    if r.get('family') == family and r.get('verifier')), None)
        if row is not None:
            validation.append(row)
    def heldout_scores():
        greedy = make_sampler(temp=0)
        return {row['id']: reward(sample(model, tokenizer,
            row['token_ids'][:row['answer_start']], 48, greedy)[0], row['verifier'])
            for row in validation}
    baseline_heldout = heldout_scores()
    started = time.monotonic()
    results = []
    chosen = None
    for row in candidates[:args.groups]:
        rollouts = []
        action_prefix = ('[BROWSER_TOOL]\nACTION: '
                         if args.force_browser_prefix and row['verifier']['kind'] == 'block'
                         and row['verifier']['tool'] == 'BROWSER_TOOL'
                         and not row.get('native_browser_tool') else '')
        sample_prompt = row['token_ids'][:row['answer_start']] + (
            tokenizer.encode(action_prefix, add_special_tokens=False) if action_prefix else [])
        for _ in range(args.rollouts):
            suffix, ids = sample(model, tokenizer, sample_prompt,
                                 args.max_tokens, sampler)
            text = action_prefix + suffix
            scored = ((native_browser_call_to_block(text) or text)
                      if row.get('native_browser_tool') else text)
            rollouts.append({'text': text, 'token_ids': ids,
                             'prompt_token_ids': sample_prompt,
                             'strict_reward': reward(scored, row['verifier']),
                             'shaped_reward': structured_reward(scored, row['verifier'],
                                                                 args.lexical_shaping)})
        values = [r['shaped_reward'] for r in rollouts]
        results.append({'row_id': row['id'], 'family': row['family'],
                        'rewards': values,
                        'sample_prefixes': [r['text'][:100] for r in rollouts],
                        'strict_successes': sum(r['strict_reward'] > 0 for r in rollouts)})
        if max(values) > min(values):
            # Pick the first positive advantage; the group mean is fixed before
            # the update, so this is a reward-weighted policy gradient step.
            baseline = sum(values)/len(values)
            index = next(i for i, value in enumerate(values) if value > baseline)
            chosen = row, rollouts[index], values[index]-baseline
            break
    report = {'stage': 'full_weight_on_policy_rl_trial', 'model': str(args.model),
              'data_manifest_sha256': sha(args.data / 'manifest.json'),
              'groups': results, 'rollouts_per_group': args.rollouts,
              'rl_updates': 0, 'experts_trained': 0, 'promoted': False,
              'baseline_heldout_scores': baseline_heldout,
              'reward_note': 'exact typed verifier; partial syntax rewards stay negative',
              'lexical_shaping': args.lexical_shaping,
              'temperature': args.temperature,
              'forced_browser_prefix': args.force_browser_prefix}
    if chosen is None:
        report['status'] = 'no_reward_variance_no_update'
    else:
        row, selected, advantage = chosen
        prompt = selected['prompt_token_ids']
        sequence = prompt + selected['token_ids']
        if len(sequence) > 384 or len(selected['token_ids']) == 0:
            report['status'] = 'sample_too_long_no_update'
        else:
            def log_prob(m, ids):
                sequence_array = mx.array([ids], mx.int32)
                logits = m(sequence_array[:, :-1])
                completion_logits = logits[:, len(prompt)-1:, :]
                targets = sequence_array[:, len(prompt):]
                selected_log_probs = (mx.take_along_axis(completion_logits,
                    targets[..., None], axis=-1)[..., 0]
                    - mx.logsumexp(completion_logits, axis=-1))
                return mx.mean(selected_log_probs)
            # The native model has no dropout; pre/post log probability is a
            # local policy-drift guard without keeping a second 8 GB copy.
            before = log_prob(model, sequence); mx.eval(before)
            before_value = float(before.item())
            for layer in model.layers:
                grad_checkpoint(layer)
            model.train()
            value_and_grad = nn.value_and_grad(model,
                lambda m, ids: -advantage * log_prob(m, ids))
            loss, gradients = value_and_grad(model, sequence)
            mx.eval(loss, gradients)
            optimizer = optimizers.Adafactor(learning_rate=args.learning_rate,
                relative_step=False, scale_parameter=False, beta_1=None,
                weight_decay=0.0)
            updated = update_all(model, gradients, optimizer, {}, 1)
            del gradients, loss
            model.eval()
            after = log_prob(model, sequence); mx.eval(after)
            after_value = float(after.item())
            drift = abs(after_value-before_value)
            report.update({'selected_row': row['id'], 'selected_reward': selected['shaped_reward'],
                           'advantage': advantage, 'updated_tensors': updated,
                           'sample_logprob_before': before_value,
                           'sample_logprob_after': after_value,
                           'local_logprob_drift': drift,
                           'peak_memory_gib': mx.get_peak_memory()/1024**3})
            after_heldout = heldout_scores()
            report['after_heldout_scores'] = after_heldout
            if drift > 0.05:
                report['status'] = 'rejected_policy_drift'
            elif any(value > 0 and after_heldout[key] < 0
                     for key, value in baseline_heldout.items()):
                report['status'] = 'rejected_heldout_regression'
            else:
                checkpoint = args.output / 'checkpoint'
                save_model(checkpoint, model)
                for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                             'tokenizer_config.json', 'special_tokens_map.json',
                             'chat_template.jinja', 'vocab.json', 'merges.txt'):
                    shutil.copy2(args.model / name, checkpoint / name)
                report.update({'status': 'candidate_saved', 'rl_updates': 1,
                               'checkpoint': str(checkpoint)})
    report['elapsed_seconds'] = round(time.monotonic()-started, 2)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
