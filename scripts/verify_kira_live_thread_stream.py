#!/usr/bin/env python3
"""Reproduce UI import then inference-thread generation on real Metal weights."""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
from mlx_lm.generate import generate_step  # deliberately imported on main thread
from kira_live.native_session import KiraNativeModelBundle


def probe(bundle):
    with mx.stream(bundle.mlx_cpu_stream), mx.stream(bundle.mlx_stream):
        ids = mx.array([bundle.thinker.tokenizer.encode('Hey Kira.', add_special_tokens=False)])
        prefix = bundle.thinker.build_live_prefix(bundle.thinker.embed(ids), listener_unit_ids=ids)
        result = bundle.thinker.generate_from_embeddings(prefix.embeddings, ngram_ids=prefix.ngram_ids, max_tokens=12, compute_response_hidden=False)
        return {'tokens': len(result.token_ids), 'reply': result.text}


with ThreadPoolExecutor(max_workers=1) as worker:
    bundle = worker.submit(KiraNativeModelBundle).result()
    for turn in range(2):
        result = worker.submit(probe, bundle).result()
        print({'turn': turn + 1, **result}, flush=True)
        if not result['tokens']:
            raise RuntimeError('No generated tokens')
