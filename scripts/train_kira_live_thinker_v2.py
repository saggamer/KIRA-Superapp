#!/usr/bin/env python3
"""Train a gated KIRA Thinker candidate with untruncated runtime-format SFT."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from kira_live.mlx_thinker import KiraMLXThinker
from kira_live.weights import resolve_composite_weights


MIX = {
    "kira_agentic": 0.40,
    "kira_workflow": 0.15,
    "openassistant": 0.20,
    "ultrachat": 0.10,
    "argilla": 0.15,
}
TARGETED_MIX = {
    "kira_agentic": 0.50,
    "kira_workflow": 0.20,
    "openassistant": 0.10,
    "ultrachat": 0.05,
    "privacy": 0.10,
    "identity": 0.05,
}
TRUTH_MIX = {
    "kira_agentic": 0.35, "kira_workflow": 0.15,
    "openassistant": 0.08, "ultrachat": 0.03,
    "privacy": 0.05, "identity": 0.04, "truth": 0.30,
}
TREE_MIX = {
    "tree": 0.40, "truth": 0.15,
    "kira_agentic": 0.20, "kira_workflow": 0.10,
    "openassistant": 0.06, "ultrachat": 0.02,
    "privacy": 0.04, "identity": 0.03,
}
CONVERSATION_MIX = {
    "truth": 0.60, "kira_agentic": 0.20, "kira_workflow": 0.10,
    "openassistant": 0.03, "ultrachat": 0.01, "privacy": 0.04, "identity": 0.02,
}
EXPERT_REPAIR_MIX = {
    "tree": 0.35, "conversation": 0.20, "reasoning": 0.20,
    "truth": 0.10, "kira_agentic": 0.10, "kira_workflow": 0.05,
}
SPLIT_AGENTIC_MIX = {"tree": 0.35, "kira_agentic": 0.35, "kira_workflow": 0.15,
                     "truth": 0.05, "reasoning": 0.05, "conversation": 0.05}
SPLIT_CONVERSATION_MIX = {"conversation": 0.35, "truth": 0.20, "reasoning": 0.20,
                          "tree": 0.10, "kira_agentic": 0.10, "kira_workflow": 0.05}


def mask_preserved_channels(gradients, prefix):
    """Prevent gradients on the established channels; new capacity alone learns."""
    import re
    result = []
    for key, gradient in tree_flatten(gradients):
        match = re.fullmatch(r'expert_layers\.\d+\.(down|up)\.\d+\.weight', key)
        if match is None:
            raise ValueError(f'Unexpected trainable parameter in separated training: {key}')
        axis = 0 if match[1] == 'down' else 1
        if prefix <= 0 or prefix >= gradient.shape[axis]:
            raise ValueError('Preserved prefix must leave new trainable channels.')
        if axis == 0:
            gradient = mx.concatenate((mx.zeros_like(gradient[:prefix]), gradient[prefix:]), axis=0)
        else:
            gradient = mx.concatenate((mx.zeros_like(gradient[:, :prefix]), gradient[:, prefix:]), axis=1)
        result.append((key, gradient))
    return tree_unflatten(result)


def encode_complete(tokenizer, messages: list[dict], max_length: int) -> tuple[mx.array, mx.array, int] | None:
    """Use exactly the public-answer template used by generation; never splice tokens."""
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        return None
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    complete = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    if not complete.startswith(prompt):
        raise ValueError("Chat template did not preserve the assistant prefix.")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    all_ids = tokenizer.encode(complete, add_special_tokens=False)
    if all_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenization crossed the assistant prefix boundary.")
    if len(all_ids) > max_length or len(all_ids) <= len(prompt_ids) + 1:
        return None
    mask = [0.0] * len(prompt_ids) + [1.0] * (len(all_ids) - len(prompt_ids))
    return mx.array([all_ids], dtype=mx.int32), mx.array([mask], dtype=mx.float32), len(prompt_ids)


def precision_mask(tokenizer, row: dict, encoded: tuple[mx.array, mx.array, int],
                   permission_weight: float, close_weight: float) -> mx.array:
    """Upweight the safety decision and final JSON closure, not prompt tokens."""
    ids, mask, answer_start = encoded
    values = mask[0].tolist()
    answer = row["messages"][-1]["content"]
    if not isinstance(answer, str):
        return mask
    try:
        action = json.loads(answer)
    except json.JSONDecodeError:
        return mask
    if not isinstance(action, dict) or not isinstance(action.get("requires_confirmation"), bool):
        return mask
    serialized = "true" if action["requires_confirmation"] else "false"
    pattern = tokenizer.encode(f'"requires_confirmation": {serialized}', add_special_tokens=False)
    token_values = ids[0].tolist()
    variants = (pattern, pattern[1:]) if pattern and pattern[0] == 1 else (pattern,)
    matches = [
        start + len(variant) - 1
        for variant in variants if variant
        for start in range(answer_start, len(token_values) - len(variant) + 1)
        if token_values[start:start + len(variant)] == variant
    ]
    if len(set(matches)) != 1:
        raise ValueError(f"Could not locate one permission decision token in {row['id']}.")
    values[matches[0]] = permission_weight
    # Qwen's complete assistant JSON ends with the closing brace before the
    # chat end marker. Weight only that brace; never train on an incomplete JSON.
    for index in range(len(token_values) - 1, answer_start - 1, -1):
        if tokenizer.decode([token_values[index]], skip_special_tokens=False) == "}":
            values[index] = close_weight
            break
    return mx.array([values], dtype=mx.float32)


def prepare(path: Path, tokenizer, max_length: int, *, permission_weight: float = 1.0,
            close_weight: float = 1.0) -> tuple[dict[str, list[tuple[mx.array, mx.array, int]]], Counter]:
    rows = defaultdict(list)
    counts = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source = row["source"]
        counts[f"{source}_seen"] += 1
        encoded = encode_complete(tokenizer, row["messages"], max_length)
        if encoded is None:
            counts[f"{source}_skipped"] += 1
            continue
        if source == "kira_agentic" and (permission_weight > 1.0 or close_weight > 1.0):
            ids, _mask, answer_start = encoded
            encoded = (
                ids,
                precision_mask(tokenizer, row, encoded, permission_weight, close_weight),
                answer_start,
            )
            counts["permission_weighted"] += 1
        rows[source].append(encoded)
        counts[f"{source}_usable"] += 1
    return dict(rows), counts


def efficient_loss_fn(model, token_ids, assistant_mask, answer_start: int):
    if answer_start <= 0 or answer_start >= token_ids.shape[1]:
        raise ValueError("Assistant loss boundary must be within the full conversation.")
    embeddings = model.embed(token_ids)
    _hidden, logits = model.forward_hidden(
        embeddings,
        emotion_features=mx.zeros((1, 8), dtype=embeddings.dtype),
        ngram_ids=token_ids,
        logits_from=answer_start - 1,
    )
    # Only the final assistant's answer and EOS are supervised. Projecting the
    # whole (often 600-token) tool schema into 248k vocabulary logits wastes
    # GPU memory and latency without providing a training signal.
    answer_targets = token_ids[:, answer_start:]
    log_probs = nn.log_softmax(logits[:, :-1].astype(mx.float32), axis=-1)
    selected = mx.take_along_axis(log_probs, answer_targets[..., None], axis=-1)[..., 0]
    weights = assistant_mask[:, answer_start:]
    language_loss = -mx.sum(selected * weights) / mx.maximum(mx.sum(weights), 1.0)
    router = model.last_router_logits.astype(mx.float32)
    probabilities = mx.softmax(router, axis=-1)
    mean_probability = mx.mean(probabilities, axis=(0, 1, 2))
    from kira_live.expert_profiles import FAMILY_INDICES
    budgets = model.expert_layers[0].family_budgets
    target_values = [0.0] * 10
    for family, budget in zip(FAMILY_INDICES, budgets):
        for expert in family:
            target_values[expert] = budget / len(family)
    target = mx.array(target_values, dtype=mx.float32)
    balance = -mx.sum(target * mx.log(mx.maximum(mean_probability, 1e-7)))
    return language_loss + 0.015 * balance


def evaluate(model, validation, samples_per_source: int) -> dict[str, float]:
    scores = {}
    for source, rows in validation.items():
        values = []
        for ids, mask, answer_start in rows[:samples_per_source]:
            loss = efficient_loss_fn(model, ids, mask, answer_start)
            mx.eval(loss)
            values.append(float(loss.item()))
        scores[source] = sum(values) / len(values) if values else float("inf")
    scores["balanced_mean"] = sum(scores.values()) / len(scores)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "training_data/kira_live_thinker_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "training_runs/kira_live_thinker_v2")
    parser.add_argument("--base-checkpoint", type=Path, help="Pinned local Qwen3.5 model directory; defaults to KIRA's 0.8B donor.")
    parser.add_argument("--expert-rank", type=int, default=128, help="Low-rank width of each Thinker expert; 768 provides about 0.151B token-active MoE parameters.")
    parser.add_argument("--expert-profile", choices=("legacy", "split_50_50"), default="legacy")
    parser.add_argument("--expert-group", choices=("agentic", "conversation"), help="Separate optimizer pass; train only new channels in this expert group.")
    parser.add_argument("--preserve-expert-prefix", type=int, default=128)
    parser.add_argument("--init-addons", type=Path, help="Resume from a prior unpromoted KIRA adapter checkpoint.")
    parser.add_argument("--continue-addons", type=Path, help="Overlay a partial checkpoint after --init-addons.")
    parser.add_argument("--targeted-mix", action="store_true", help="Use the small permission-balanced laptop curriculum.")
    parser.add_argument("--truth-mix", action="store_true", help="Use the targeted curriculum plus evidence-honesty and emotion-calibration repair examples.")
    parser.add_argument("--tree-mix", action="store_true", help="Train exact Superapp Tree tool contracts with conversation/safety retention.")
    parser.add_argument("--conversation-mix", action="store_true", help="Conversation repair with permission/workflow retention.")
    parser.add_argument("--expert-repair-mix", action="store_true", help="Compact native-policy tool, reasoning and conversation expert repairs.")
    parser.add_argument("--conversation-experts-only", action="store_true", help="Update only conversation, memory, emotion and speech adapters; freeze agentic experts and routers.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-per-source", type=int, default=6)
    parser.add_argument("--keep-eval-checkpoints", action="store_true", help="Retain each improved evaluation checkpoint for generation-gate comparison.")
    parser.add_argument("--permission-token-weight", type=float, default=1.0)
    parser.add_argument("--json-close-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-last-experts", type=int, default=0,
        help="When positive, train only this many final expert adapters and keep PLE frozen; useful on memory-limited Macs.",
    )
    parser.add_argument("--train-experts-only", action="store_true", help="Train every expert adapter while keeping PLE frozen.")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if (args.steps <= 0 or args.eval_every <= 0 or args.train_last_experts < 0 or args.learning_rate <= 0
            or args.permission_token_weight < 1.0 or args.json_close_weight < 1.0):
        raise ValueError("Training steps and evaluation interval must be positive.")
    if args.train_last_experts and args.train_experts_only:
        raise ValueError("Choose either --train-last-experts or --train-experts-only.")
    if args.expert_group and (args.expert_profile != 'split_50_50' or not args.train_last_experts or args.conversation_experts_only or args.init_addons is None):
        raise ValueError('Separated training requires split_50_50, init-addons and train-last-experts, with no legacy conversation-only flag.')
    if sum((args.targeted_mix, args.truth_mix, args.tree_mix, args.conversation_mix, args.expert_repair_mix)) > 1:
        raise ValueError("Choose only one curriculum mix.")
    args.output.mkdir(parents=True, exist_ok=True)
    base_checkpoint = args.base_checkpoint or resolve_composite_weights().thinker.snapshot
    model = KiraMLXThinker.from_pretrained(base_checkpoint, expert_rank=args.expert_rank, expert_profile=args.expert_profile)
    if args.init_addons is not None:
        if not args.init_addons.is_file():
            raise FileNotFoundError(args.init_addons)
        model.load_weights(str(args.init_addons), strict=False)
    if args.continue_addons is not None:
        if args.init_addons is None or not args.continue_addons.is_file():
            raise ValueError("--continue-addons needs an existing --init-addons and checkpoint.")
        model.load_weights(str(args.continue_addons), strict=False)
    model.freeze()
    if args.train_last_experts:
        if args.train_last_experts > len(model.expert_layers):
            raise ValueError("Requested more final expert adapters than the model has.")
        for module in model.expert_layers[-args.train_last_experts:]:
            if args.expert_group:
                from kira_live.expert_profiles import AGENTIC_INDICES, CONVERSATION_INDICES
                selected = AGENTIC_INDICES if args.expert_group == 'agentic' else CONVERSATION_INDICES
                for expert_index in selected:
                    module.down[expert_index].unfreeze()
                    module.up[expert_index].unfreeze()
            elif args.conversation_experts_only:
                for expert_index in (0, 1, 4, 8):
                    module.down[expert_index].unfreeze()
                    module.up[expert_index].unfreeze()
            else:
                module.unfreeze()
    elif args.train_experts_only:
        for module in model.expert_layers:
            module.unfreeze()
    else:
        for module in model.ngram_layers + model.expert_layers:
            module.unfreeze()
    model.train()  # DeltaNet training scan has a gradient; inference kernel does not.
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if (args.train_experts_only or args.train_last_experts) and (not trainable or any(not key.startswith('expert_layers.') for key in trainable)):
        raise RuntimeError('Expert-only scope violated; refusing training.')
    prefix_reference = {}
    if args.expert_group:
        for key, value in trainable.items():
            prefix_reference[key] = value[:args.preserve_expert_prefix] if '.down.' in key else value[:, :args.preserve_expert_prefix]
    train, train_counts = prepare(
        args.data / "train.jsonl", model.tokenizer, args.max_length,
        permission_weight=args.permission_token_weight,
        close_weight=args.json_close_weight,
    )
    valid, valid_counts = prepare(args.data / "valid.jsonl", model.tokenizer, args.max_length)
    if args.conversation_experts_only and not args.train_last_experts:
        raise ValueError("Conversation-only training requires --train-last-experts.")
    mix = EXPERT_REPAIR_MIX if args.expert_repair_mix else CONVERSATION_MIX if args.conversation_mix else TREE_MIX if args.tree_mix else TRUTH_MIX if args.truth_mix else TARGETED_MIX if args.targeted_mix else MIX
    if args.expert_group:
        mix = SPLIT_AGENTIC_MIX if args.expert_group == 'agentic' else SPLIT_CONVERSATION_MIX
    if set(train) != set(mix) or set(valid) != set(mix):
        raise ValueError("A required train or validation source was removed by filtering.")
    if any(train_counts[f"{name}_skipped"] for name in mix):
        raise ValueError("Training examples were truncated; increase max-length or curate them.")
    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0 if args.expert_group else 0.01)
    loss_and_grad = nn.value_and_grad(model, efficient_loss_fn)
    source_names = tuple(mix)
    source_weights = tuple(mix.values())
    cursor = {name: 0 for name in mix}
    for rows in train.values():
        rng.shuffle(rows)
    baseline = evaluate(model, valid, args.eval_per_source)
    print(json.dumps({"step": 0, "validation": baseline, "usable": dict(train_counts)}), flush=True)
    best = baseline["balanced_mean"]
    history = [{"step": 0, "validation": baseline}]
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        source = rng.choices(source_names, weights=source_weights, k=1)[0]
        rows = train[source]
        index = cursor[source]
        if index >= len(rows):
            rng.shuffle(rows)
            index = 0
        ids, mask, answer_start = rows[index]
        cursor[source] = index + 1
        loss, gradients = loss_and_grad(model, ids, mask, answer_start)
        if args.expert_group:
            gradients = mask_preserved_channels(gradients, args.preserve_expert_prefix)
        optimizer.update(model, gradients)
        mx.eval(model.trainable_parameters(), optimizer.state)
        if args.expert_group:
            for key, value in tree_flatten(model.trainable_parameters()):
                prefix = value[:args.preserve_expert_prefix] if '.down.' in key else value[:, :args.preserve_expert_prefix]
                if not bool(mx.array_equal(prefix, prefix_reference[key]).item()):
                    raise RuntimeError(f'Established expert channels changed: {key}')
        if step == 1:
            print(json.dumps({"step": step, "source": source, "loss": float(loss.item()), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
        if step % 10 == 0:
            print(json.dumps({"step": step, "source": source, "loss": float(loss.item()), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            current = evaluate(model, valid, args.eval_per_source)
            entry = {"step": step, "validation": current}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            weights = dict(tree_flatten(model.trainable_parameters()))
            if args.keep_eval_checkpoints:
                mx.save_safetensors(str(args.output / f"step_{step}.safetensors"), weights)
            if current["balanced_mean"] < best:
                best = current["balanced_mean"]
                mx.save_safetensors(
                    str(args.output / "thinker_addons_candidate.safetensors"),
                    weights,
                )
    report = {
        "stage": "kira_live_thinker_v2_full_context_sft",
        "promoted": False,
        "base_frozen": True,
        "base_checkpoint": str(base_checkpoint),
        "expert_rank": args.expert_rank,
        "expert_profile": args.expert_profile,
        "expert_group": args.expert_group,
        "preserved_prefix_channels": args.preserve_expert_prefix if args.expert_group else None,
        "established_channels_preserved": bool(args.expert_group),
        "active_moe_parameters": model.active_moe_parameters,
        "init_addons": str(args.init_addons) if args.init_addons else None,
        "continue_addons": str(args.continue_addons) if args.continue_addons else None,
        "train_last_experts": args.train_last_experts,
        "conversation_experts_only": args.conversation_experts_only,
        "train_experts_only": args.train_experts_only,
        "trainable_tensor_names": list(trainable),
        "trainable_parameter_count": sum(value.size for value in trainable.values()),
        "effective_new_channel_parameter_count": sum(value.size - args.preserve_expert_prefix * (value.shape[1] if '.down.' in key else value.shape[0]) for key, value in trainable.items()) if args.expert_group else None,
        "optimizer_weight_decay": 0.0 if args.expert_group else 0.01,
        "permission_token_weight": args.permission_token_weight,
        "json_close_weight": args.json_close_weight,
        "runtime_chat_template": True,
        "truncation_policy": "reject complete examples longer than max_length",
        "max_length": args.max_length,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "mix": mix,
        "train_counts": dict(train_counts),
        "valid_counts": dict(valid_counts),
        "baseline_validation": baseline,
        "best_balanced_validation": best,
        "history": history,
        "candidate": str(args.output / "thinker_addons_candidate.safetensors"),
        "release_gate": "held-out generation, conversation, voice latency, and human live tests remain required",
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": report["candidate"], "best_balanced_validation": best, "promoted": False}), flush=True)


if __name__ == "__main__":
    main()
