"""Does the CTM Transformer learn reliably where Dans does not? Addition in text, more replicates (2026-11-06).

In the reasoning-in-text experiment (scripts/reasoning_in_text.py, sampler arms) all four CTsq runs learned reversed
addition (98-100% exact match on 1-10 digits), while Dans replicate 2 failed (18%) and Dans replicate 1 learned it
(98%). Both Dans runs had gradient-norm spikes after warmup (3.1 at step 353; 6.3 at step 65, 2.6 at 237); neither
CTsq run exceeded 0.65. One failure is an anecdote: this asks whether it is a property, on fresh replicates.

Setup: exactly the reasoning-in-text sampler arms (the same data, 100M tokens of continued pretraining at peak lr 1e-3,
CTsq compiled at full micro-batch), replicates 3-10 for Dans_sampler and CTsq_sampler. Replicate r starts from the
400M-token checkpoint of seed 1 (odd r) or seed 2 (even r), and its data order and tick sampler use seed r.
Evaluation: as reasoning_in_text (256 problems per task-split, teacher-forced exact match).

Rules, fixed before the runs. Judged on the fresh replicates 3-10 only (8 per model); replicates 1-2 suggested the
hypothesis and are reported pooled, not judged.
- A run learns addition if its ID addition accuracy at 16 ticks is at least 90%.
- Reliability: CTsq is more reliable if it learns in at least 3 more of the 8 replicates than Dans (and the reverse).
  The one-sided Fisher exact p is reported.
- Stability: a run spikes if its gradient norm (before clipping) exceeds 2.0 at any update after warmup (step > 50).
  CTsq is more stable if it spikes in at least 3 fewer of the 8 replicates than Dans (and the reverse).
- Reported: whether failures and spikes coincide; results by starting checkpoint; the other tasks' accuracy.

Usage: python -m scripts.addition_reliability --queue k/2 | --summarize
"""
import argparse,json,math,os
from pathlib import Path

import scripts.reasoning_in_text as R

ARMS=('Dans_sampler','CTsq_sampler');FRESH=range(3,11);ALL=range(1,11);LEARNED,SPIKE,WARMUP=90.0,2.0,50
OUT=R.OUT


def config(arm,replicate):
    run=R.config(arm,2-replicate%2);run.update(name=f'{arm}_r{replicate}',seed=replicate,run_directory=str(R.RUNS/f'{arm}_r{replicate}'))
    return run


def train(arm,replicate,gpu):
    import subprocess,sys
    d=R.RUNS/f'{arm}_r{replicate}'
    if (d/'complete.json').exists():return
    d.mkdir(parents=True,exist_ok=True);(d/'config.json').write_text(json.dumps(config(arm,replicate),indent=2)+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
    with open(d/'train.log','a') as f:subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(d/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env)
    if not (d/'complete.json').exists():raise RuntimeError(f'{arm}_r{replicate}: training did not complete')


def queue(k,n):
    assert n==2;return [(ARMS[(r+k)%2],r) for r in FRESH]  # each GPU alternates models: 4 runs of each per GPU


def fisher_one_sided(a,b,n):
    """P(X >= a) for X the first group's successes under the hypergeometric null (a of n vs b of n)."""
    total=a+b;p=lambda x:math.comb(n,x)*math.comb(n,total-x)/math.comb(2*n,total)
    return sum(p(x) for x in range(a,min(n,total)+1))


def run_facts(arm,r):
    f=OUT/f'{arm}_r{r}.json'
    if not f.exists():return None
    acc=json.loads(f.read_text())['accuracy'];rows=[json.loads(x) for x in (R.RUNS/f'{arm}_r{r}'/'metrics.jsonl').read_text().splitlines() if x.strip()]
    late=[x['grad_norm'] for x in rows if x['step']>WARMUP]
    return {'add':acc['add_id']['fixed16'],'add4':acc['add_id']['fixed4'],'learned':acc['add_id']['fixed16']>=LEARNED,'max_grad':max(late),
            'spikes':sum(g>SPIKE for g in late),'spiked':max(late)>SPIKE,'base':2-r%2,'other':{k:acc[k]['fixed16'] for k in ('track_id','parity_id','eval_id')}}


def summarize():
    facts={(a,r):run_facts(a,r) for a in ARMS for r in ALL};L=['# Addition reliability (d704, 100M tokens of continued pretraining)','',
       'Protocol and rules: `scripts/addition_reliability.py` (docstring). Learned: ID addition at 16 ticks at least 90%. Spike: gradient norm above 2.0 after warmup.','',
       '| Arm | Replicate | Start | Addition ID at 4 / 16 ticks | Learned | Max gradient norm after warmup | Spike updates | Tracking / parity / expressions ID |','|---|---:|---:|---|---|---:|---:|---|']
    for a in ARMS:
        for r in ALL:
            v=facts[(a,r)]
            if v:L.append(f'| {a} | {r}{"" if r in FRESH else " (not judged)"} | s{v["base"]} | {v["add4"]:.0f} / {v["add"]:.0f} | {"yes" if v["learned"] else "no"} | {v["max_grad"]:.2f} | {v["spikes"]} | '+' / '.join(f'{x:.0f}' for x in v['other'].values())+' |')
    fresh={a:[facts[(a,r)] for r in FRESH] for a in ARMS}
    if not all(all(fresh[a]) for a in ARMS):
        L+=['','- incomplete: '+', '.join(f'{a} {sum(1 for v in fresh[a] if v)}/8' for a in ARMS)];(OUT/'ADDITION_RELIABILITY.md').write_text('\n'.join(L)+'\n');print('\n'.join(L));return
    d,c=(sum(v['learned'] for v in fresh[a]) for a in ARMS);ds,cs=(sum(v['spiked'] for v in fresh[a]) for a in ARMS)
    pooled={a:sum(facts[(a,r)]['learned'] for r in ALL if facts[(a,r)]) for a in ARMS}
    coincide={a:sum(v['spiked'] and not v['learned'] for v in fresh[a]) for a in ARMS}
    L+=['','## Rule outcomes (replicates 3-10)','',
        f'- **Learned:** Dans {d}/8, CTsq {c}/8 (one-sided Fisher p: CTsq more {fisher_one_sided(c,d,8):.3f}, Dans more {fisher_one_sided(d,c,8):.3f})',
        f'- **Reliability:** '+('CTsq is more reliable' if c>=d+3 else 'Dans is more reliable' if d>=c+3 else 'no difference by the rule'),
        f'- **Spiked:** Dans {ds}/8, CTsq {cs}/8; stability: '+('CTsq is more stable' if ds>=cs+3 else 'Dans is more stable' if cs>=ds+3 else 'no difference by the rule'),
        f'- **Failed runs that spiked:** Dans {coincide["Dans_sampler"]} of {8-d} failures, CTsq {coincide["CTsq_sampler"]} of {8-c}',
        f'- **Pooled with replicates 1-2 (reported):** Dans {pooled["Dans_sampler"]}/10, CTsq {pooled["CTsq_sampler"]}/10',
        '- **By starting checkpoint (learned, fresh):** '+', '.join(f'{a} s{b} {sum(v["learned"] for v in fresh[a] if v["base"]==b)}/4' for a in ARMS for b in (1,2))]
    (OUT/'ADDITION_RELIABILITY.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    k,n=map(int,a.queue.split('/'));os.environ['CUDA_VISIBLE_DEVICES']=str(k);jobs=queue(k,n);print('queue',k,[f'{x}_r{y}' for x,y in jobs],flush=True)
    for arm,rep in jobs:train(arm,rep,k);R.evaluate(f'{arm}_r{rep}')


if __name__=='__main__':main()
