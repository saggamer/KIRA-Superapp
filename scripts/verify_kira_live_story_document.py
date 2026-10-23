"""Real Live Thinker body -> existing Tree document specialist.

Pipe --body (MLX environment) into --artifact (bundled document environment).
No private chat or microphone input is used.
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REQUEST = 'Create a small Word document with an original spooky horror story about a midnight camp.'


def main():
    if '--body' in sys.argv:
        import mlx.core as mx
        from kira_live.checkpoint_bundle import resolve_live_checkpoint
        from kira_live.mlx_thinker import KiraMLXThinker
        from kira_live.privacy import sanitize_public_output
        from kira_live.action_contracts import docx_contract
        checkpoint = resolve_live_checkpoint()
        thinker = KiraMLXThinker.from_pretrained(checkpoint.weights.thinker.snapshot)
        thinker.load_weights(str(checkpoint.thinker_addons), strict=False)
        thinker.eval()
        ids = mx.array([thinker.tokenizer.encode(REQUEST, add_special_tokens=False)])
        prefix = thinker.build_live_prefix(thinker.embed(ids), listener_unit_ids=ids, artifact_content=True)
        response = thinker.generate_from_embeddings(prefix.embeddings, ngram_ids=prefix.ngram_ids, max_tokens=384, compute_response_hidden=False)
        body = sanitize_public_output(response.text, live=True)
        print(json.dumps({'request': REQUEST, 'body': body, 'contract': docx_contract(body, REQUEST)}), flush=True)
    else:
        import types
        sys.modules.setdefault('webview', types.SimpleNamespace())
        sys.path.insert(0, str(ROOT / 'tests'))
        from test_tree_web_artifacts import HarnessActions
        packet = json.load(sys.stdin)
        root = ROOT / 'kira_voice_tmp/live_tree_story_test'
        root.mkdir(parents=True, exist_ok=True)
        actions = HarnessActions(root)
        result = actions._generate_native_docx_blocks(packet['contract'], packet['request'], 'story-fixture')
        print(json.dumps({'tree_result': result, 'artifacts': actions.last_artifacts}), flush=True)
        if 'DOCX generated:' not in result:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
