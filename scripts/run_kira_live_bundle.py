"""Standalone three-weight KIRA Live bundle: verify, GPU probe, or microphone.

THE TREE execution requires KIRA Superapp; standalone mode is conversation only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent
# Also usable from the Superapp scripts directory during release verification.
if not (ROOT/'kira_live').is_dir(): ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--bundle-root',type=Path,default=ROOT)
    args=parser.parse_args()
    root=args.bundle_root.resolve()
    os.environ['KIRA_LIVE_BUNDLE_ROOT']=str(root)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.verify:
        checks=json.loads((root/'checksums.json').read_text())
        for relative,expected in checks.items():
            rel=Path(relative)
            if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe checksum path')
            with (root/rel).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected: raise RuntimeError(f'Checksum mismatch: {relative}')
        print(f'PASS: {len(checks)} release files match SHA-256 checksums',flush=True)
        return
    from kira_live.native_session import KiraNativeModelBundle,KiraNativeLiveSession
    if args.probe:
        import mlx.core as mx
        def probe():
            bundle=KiraNativeModelBundle()
            with mx.stream(bundle.mlx_cpu_stream),mx.stream(bundle.mlx_stream):
                replies=[]
                for question in ['Hello. Say a short friendly greeting.','What is your name?']:
                    ids=mx.array([bundle.thinker.tokenizer.encode(question,add_special_tokens=False)])
                    prefix=bundle.thinker.build_live_prefix(bundle.thinker.embed(ids),listener_unit_ids=ids)
                    answer=bundle.thinker.generate_from_embeddings(prefix.embeddings,ngram_ids=prefix.ngram_ids,max_tokens=48,compute_response_hidden=False)
                    if not answer.token_ids: raise RuntimeError('Empty generation')
                    replies.append(answer.text)
                return {'three_islands_loaded':True,'replies':replies}
        with ThreadPoolExecutor(max_workers=1) as worker: print(json.dumps(worker.submit(probe).result(),indent=2),flush=True)
        return
    def event(kind,payload):
        if kind in {'TRANSCRIPT','RESPONSE_TEXT','LIVE_ERROR','STATUS','RESPONSE_FINISHED'}:
            print(kind, json.dumps(payload,ensure_ascii=False),flush=True)
    session=KiraNativeLiveSession(chat_id='standalone-live',on_event=event)
    try:
        session.start()
        print('Listening. Press Enter to stop. Standalone tool execution is disabled.',flush=True)
        input()
    finally: session.stop()

if __name__=='__main__': main()
