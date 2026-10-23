#!/usr/bin/env python3
"""Sealed paired audit: prior 8-expert v1.5 versus browser-first 16-expert run."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from orchestrator_v15.contracts import reward
from orchestrator_v15.checkpoints import research_checkpoint
from orchestrator_v15.regression import score_expected_json


DATA=ROOT/'training_data/orchestrator_v15_browser12b'
PRIOR=ROOT/'training_runs/orchestrator_v15/local_pilot_20261005/retained_candidate.safetensors'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_cases(per_family=3, legacy_count=20):
    manifest=json.loads((DATA/'manifest.json').read_text())
    for split in ('train','valid','test'):
        if sha(DATA/(split+'.jsonl'))!=manifest['files'][split]['sha256']:
            raise ValueError('corpus changed after preparation')
    grouped=defaultdict(list)
    for row in read_rows(DATA/'test.jsonl'):
        if row.get('verifier') and row['source'] in {'kira_v15_authored','kira_diagonal_browser_authored'}:
            grouped[row['family']].append(row)
    selected=[]
    for family,group in sorted(grouped.items()):
        selected.extend(sorted(group,key=lambda r:r['id'])[:per_family])
    legacy=read_rows(ROOT/'training_data/kira_agentic_sft_v1/test.jsonl')
    work=min(legacy_count//2,32)
    indices=list(range(work))+list(range(32,min(len(legacy),32+legacy_count-work)))
    selected.extend({'id':f'original-heldout/{i}',
                     'family':'original_workflow' if i<32 else 'original_app_action',
                     'source':'original_kira_holdout',
                     'messages':legacy[i]['messages']} for i in indices)
    return manifest,selected


def evaluate_variant(variant,run,per_family,legacy_count):
    import mlx.core as mx
    from mlx_lm import load,generate
    from mlx_lm.sample_utils import make_sampler
    from orchestrator_v15.moe import attach

    manifest,selected=selected_cases(per_family,legacy_count)
    if variant=='prior':
        checkpoint=PRIOR;experts=8;ceiling=10_000_000_000
    else:
        checkpoint,_=research_checkpoint(run);experts=16;ceiling=12_000_000_000
        run_manifest=json.loads((run/'data_manifest.json').read_text())
        if run_manifest!=manifest:
            raise ValueError('run was not trained on this exact corpus')
    if not checkpoint.exists():raise FileNotFoundError(checkpoint)
    mx.set_cache_limit(256*1024**2);mx.set_memory_limit(18*1024**3)
    model,tokenizer=load(manifest['base_snapshot'],lazy=False)
    modules,budget=attach(model,experts=experts,ceiling=ceiling)
    model.load_weights(str(checkpoint),strict=False);mx.eval(model.parameters())
    for module in modules:module._enabled=True;module._route_label=None
    model.eval()
    checkpoint_sha=sha(checkpoint)
    output=[]
    path=run/f'browser12b_{variant}_sealed.json'
    for index,row in enumerate(selected,1):
        if 'token_ids' in row:
            prompt=row['token_ids'][:row['answer_start']]
        else:
            rendered=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=False,
                                                    add_generation_prompt=True,enable_thinking=False)
            prompt=tokenizer.encode(rendered,add_special_tokens=False)
        answer=generate(model,tokenizer,prompt=prompt,max_tokens=128,
                        sampler=make_sampler(temp=0),verbose=False)
        if row.get('verifier'):
            verdict={'full':reward(answer,row['verifier'])==1.0}
        else:
            expected=json.loads(row['messages'][-1]['content'])
            verdict=score_expected_json(answer,expected)
        output.append({'id':row['id'],'family':row['family'],'source':row['source'],
                       'prompt_tokens':len(prompt),'answer':answer,'verdict':verdict})
        path.write_text(json.dumps({'variant':variant,'checkpoint':str(checkpoint),
                                    'checkpoint_sha256':checkpoint_sha,
                                    'architecture':budget,'cases':output},indent=2,ensure_ascii=False))
        if index%10==0:mx.clear_cache()
        print(f'{variant} {index}/{len(selected)} {row["family"]} pass={verdict["full"]}',flush=True)


def summarize(run):
    old=json.loads((run/'browser12b_prior_sealed.json').read_text())
    new=json.loads((run/'browser12b_new_sealed.json').read_text())
    before={r['id']:r for r in old['cases']};after={r['id']:r for r in new['cases']}
    if set(before)!=set(after):raise ValueError('paired case set differs')
    groups=defaultdict(list)
    losses=[];gains=[]
    for case_id in before:
        a,b=before[case_id],after[case_id]
        if a['family']!=b['family']:raise ValueError('family mismatch')
        groups[a['family']].append((a,b))
        if a['verdict']['full'] and not b['verdict']['full']:losses.append(case_id)
        if not a['verdict']['full'] and b['verdict']['full']:gains.append(case_id)
    summary={'suite':'sealed local paired decision audit, prior 8 experts vs new 16 experts',
             'prior_sha256':old['checkpoint_sha256'],'new_sha256':new['checkpoint_sha256'],
             'cases':len(before),'prior_passes':sum(r['verdict']['full'] for r in before.values()),
             'new_passes':sum(r['verdict']['full'] for r in after.values()),
             'lost_prior_cases':losses,'gained_cases':gains,
             'by_family':{family:{'cases':len(rows),
                                  'prior_passes':sum(a['verdict']['full'] for a,_ in rows),
                                  'new_passes':sum(b['verdict']['full'] for _,b in rows)}
                          for family,rows in sorted(groups.items())},
             'release_pass':False,
             'limitation':'typed, mostly synthetic decision prompts; separate executed browser tasks required'}
    (run/'browser12b_paired_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--variant',choices=('prior','new','summarize'),required=True)
    parser.add_argument('--per-family',type=int,default=3)
    parser.add_argument('--legacy-count',type=int,default=20)
    args=parser.parse_args();run=args.run.resolve()
    if args.variant=='summarize':summarize(run)
    else:evaluate_variant(args.variant,run,args.per_family,args.legacy_count)


if __name__=='__main__':main()
