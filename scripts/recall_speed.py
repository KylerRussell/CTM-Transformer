"""How fast does in-context recall emerge in continued pretraining? Dans vs CTsqS at d704 (2026-11-17).

scripts/recall_in_text.py ended at ceiling: after 100M tokens both Dans and CTsqS answer recall inside text at 100%
on N 2-16 and 97-100% on N 17-32. At width 128 (scripts/recall_memory.py) the CTsqS advantage was in learning the
binding (19 of 25 runs vs Dans 4 of 25), not in what a run that learned it could do. The language-model analogue is
how many tokens each model needs before recall emerges. This reruns recall_in_text exactly (same data, checkpoints,
seeds and training), keeping a snapshot every 25 updates (6.6M tokens), and evaluates every snapshot.

Evaluation of each snapshot: recall_in_text's ID (N 2-16) and OOD (N 17-32) problems, 512 per split after 96 tokens
of held-out text, the value's argmax at the "=" position, at 16 ticks; accuracy by N. Snapshots at updates 25, 50,
..., 375 and 381. The curriculum reaches N 9 at update 96, 13 at 191 and 16 at 286 (stages of 95-96 updates).

Rules, fixed before any run (two replicates, paired by seed):
- A run's emergence point is the first snapshot whose ID accuracy is at least 80% (381 + 25 if none).
- A CTM win: CTsqS's emergence point is at least 50 updates (two snapshots) earlier than Dans's on both replicates;
  a CTM cost: the reverse. Otherwise neither.
- Secondary: CTsqS's mean ID accuracy over all snapshots is at least 5 points above Dans's on both replicates (a win
  on the whole curve); the reverse is a cost.
- Reported: ID and OOD curves; accuracy by N at the emergence point; the final snapshot against recall_in_text's
  result for the same run (reproducibility of the rerun).
- If either rule gives a win, the follow-up is more continued-training replicates, with rules fixed before them.

Usage: python -m scripts.recall_speed --queue k/2 | --summarize
"""
import argparse,json,os,random
from pathlib import Path
import numpy as np

import scripts.reasoning_in_text as R
import scripts.recall_in_text as T

RUNS=Path('research/runs/recall_speed');OUT=Path('research/results/recall_speed');R.RUNS=RUNS;EVERY=25;TOTAL=381
SNAPSHOTS=[*range(EVERY,TOTAL,EVERY),TOTAL]


def config(arm,replicate):
    run=T.config(arm,replicate);run['run_directory']=str(RUNS/f'{arm}_r{replicate}');run['train'].update(checkpoint_interval=EVERY,snapshot_interval=EVERY);return run


R.config=config


def evaluate(run):
    import torch
    from scripts.code_math_probe import load
    out=OUT/f'{run}.json';done=json.loads(out.read_text()) if out.exists() else {'run':run,'snapshots':{}}
    text=np.memmap(R.SRC/'validation_000.bin',dtype=np.uint16,mode='r');device=torch.device('cuda')
    sets={}
    for split,(lo,hi) in T.SPLITS.items():
        rng=random.Random(f'recall-in-text-eval-{split}');rows=[]
        for _ in range(T.EVAL_N):
            start=rng.randrange(len(text)-T.PREFIX);n=rng.randint(lo,hi);q,v,_=T.problem(n,rng);rows.append((n,[*text[start:start+T.PREFIX].tolist(),*q],v))
        L=max(len(s) for _,s,_ in rows);x=torch.zeros((T.EVAL_N,L),dtype=torch.long)
        for i,(_,s,_) in enumerate(rows):x[i,:len(s)]=torch.tensor(s)
        sets[split]=(rows,x,torch.tensor([len(s)-1 for _,s,_ in rows]),torch.tensor([v for _,_,v in rows]))
    cfg=json.loads((RUNS/run/'config.json').read_text())
    for step in SNAPSHOTS:
        if str(step) in done['snapshots']:continue
        cfg['train']['total_steps']=step;tmp=RUNS/run/'eval_config';tmp.mkdir(exist_ok=True);(tmp/'config.json').write_text(json.dumps(cfg))
        (tmp/f'step_{step:07d}.pt').unlink(missing_ok=True);(tmp/f'step_{step:07d}.pt').symlink_to((RUNS/run/f'step_{step:07d}.pt').resolve())
        model,_=load(tmp,device);(tmp/f'step_{step:07d}.pt').unlink();r={}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for split,(rows,x,at,val) in sets.items():
                hit=[]
                for i in range(0,T.EVAL_N,32):
                    logits=model(x[i:i+32].to(device),max_thought_steps=16)['logits'];hit+=(logits[torch.arange(len(logits)),at[i:i+32].to(device)].argmax(-1).cpu()==val[i:i+32]).tolist()
                by={}
                for (n,_,_),h in zip(rows,hit):b=by.setdefault(n,[0,0]);b[0]+=h;b[1]+=1
                r[split]=100*sum(hit)/T.EVAL_N;r[f'{split}_by_n']={str(n):round(100*h/c,1) for n,(h,c) in sorted(by.items())}
        done['snapshots'][str(step)]=r;OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(done,indent=1)+'\n')
        print(run,step,'id',round(r['id'],1),'ood',round(r['ood'],1),flush=True);del model;torch.cuda.empty_cache()


def summarize():
    res={(a,k):json.loads(f.read_text()) for a in T.ARMS for k in T.REPLICATES if (f:=OUT/f'{a}_r{k}.json').exists()}
    emerge=lambda r:next((s for s in SNAPSHOTS if r['snapshots'].get(str(s),{}).get('id',0)>=80),TOTAL+EVERY)
    mean=lambda v:sum(v)/len(v)
    L=['# When in-context recall emerges (d704, continued pretraining)','','Protocol and rules: `scripts/recall_speed.py`. ID accuracy (%) at 16 ticks per snapshot (updates); OOD below.','',
       '| Run | '+' | '.join(map(str,SNAPSHOTS))+' | Emerges |','|---|'+'---:|'*(len(SNAPSHOTS)+1)]
    for (a,k),r in sorted(res.items()):L.append(f'| {a}_r{k} | '+' | '.join(f"{r['snapshots'][str(s)]['id']:.0f}" if str(s) in r['snapshots'] else '' for s in SNAPSHOTS)+f' | {emerge(r)} |')
    L+=['','| Run (OOD) | '+' | '.join(map(str,SNAPSHOTS))+' |','|---|'+'---:|'*len(SNAPSHOTS)]
    for (a,k),r in sorted(res.items()):L.append(f'| {a}_r{k} | '+' | '.join(f"{r['snapshots'][str(s)]['ood']:.0f}" if str(s) in r['snapshots'] else '' for s in SNAPSHOTS)+' |')
    full=len(res)==4 and all(len(r['snapshots'])==len(SNAPSHOTS) for r in res.values())
    if full:
        a,b='CTsqS_sampler','Dans_sampler';de=[emerge(res[(b,k)])-emerge(res[(a,k)]) for k in T.REPLICATES]
        dm=[mean([res[(a,k)]['snapshots'][str(s)]['id'] for s in SNAPSHOTS])-mean([res[(b,k)]['snapshots'][str(s)]['id'] for s in SNAPSHOTS]) for k in T.REPLICATES]
        L+=['','## Rule outcomes','',
            f"- **Emergence (primary):** {'CTM win' if all(x>=50 for x in de) else 'CTM cost' if all(x<=-50 for x in de) else 'neither'} (Dans minus CTsqS, updates: {de})",
            f"- **Whole curve (secondary):** {'CTM win' if all(x>=5 for x in dm) else 'CTM cost' if all(x<=-5 for x in dm) else 'neither'} (CTsqS minus Dans, mean ID: {', '.join(f'{x:+.1f}' for x in dm)})"]
        for (x,k),r in sorted(res.items()):
            e=emerge(r);prev=T.OUT/f'{x}_r{k}.json'
            if str(e) in r['snapshots']:L.append(f"- {x}_r{k} at emergence ({e}): by N "+str(r['snapshots'][str(e)]['id_by_n']))
            if prev.exists():p=json.loads(prev.read_text());L.append(f"- {x}_r{k} final vs recall_in_text: ID {r['snapshots'][str(TOTAL)]['id']:.1f} vs {p['id']['16']:.1f}, OOD {r['snapshots'][str(TOTAL)]['ood']:.1f} vs {p['ood']['16']:.1f}")
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'RECALL_SPEED.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    k,n=map(int,a.queue.split('/'));os.environ['CUDA_VISIBLE_DEVICES']=str(k)
    jobs=[(arm,rep) for rep in T.REPLICATES for arm in T.ARMS][k::n];print('queue',k,[f'{x}_r{y}' for x,y in jobs],flush=True)
    for arm,rep in jobs:R.train(arm,rep,k);evaluate(f'{arm}_r{rep}')


if __name__=='__main__':main()
