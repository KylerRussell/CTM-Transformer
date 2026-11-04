"""Can ticks beyond 8 contribute? A screen of three training changes in the language model (2026-11-05).

Every language model here (Dans, every CTM Transformer variant, per-token budgets, the neuron-level activation)
reaches a fixed point by about 8 ticks: loss at 16, 32, 64 and 96 ticks is the same to 0.001
(research/CTM_TRANSFORMER_DESIGN.md). Three changes, each aimed at a different reason this could happen:
- film: the recurrence is the same map every tick, so a contraction settles. A per-tick scale and shift of the
  injected input (rdt_recipe `tick_film` 48: zero-initialized, ticks beyond 48 reuse the last) lets successive ticks
  do different work, as distinct layers do. (T20, 20 distinct layers, beats Dans by about 0.01 at 400M tokens.)
- mtp: next-token prediction may not need more computation at this scale. Multi-token prediction (`mtp` 2: also
  predict 2 and 3 tokens ahead, weight 0.5) asks more of the state; evaluation is next-token only.
- m32: the training ticks may be too few for longer computation to be learned. Ticks of mean 32, at most 96 (the
  usual mean 15, at most 48), micro-batch 8 above 40 ticks.

Arms: Dans and CTsq with each change, the pilot's setting (scripts/per_token_pilot.py: d704, 100M tokens, the same
data; CTsq compiled at full micro-batch as in scripts/ctm_speed_check.py). Screen: seed 1. References, seed 1:
Dans_sampler_s1 (Dans) and CTsq_fast_s1 (CTsq). Evaluation as the pilot (128 windows, fixed ticks 1-96).

Rules, fixed before the runs. For an arm and seed against its model's reference of the same seed:
- ticks pay: loss at 32 and at 64 ticks each at least 0.01 below its loss at 16 ticks;
- no cost: loss at 16 ticks at most 0.01 above the reference's;
- a win: loss at 16 ticks at least 0.02 below the reference's.
An arm passes the screen if (ticks pay and no cost) or a win holds on seed 1. A passing arm is confirmed on seed 2
(against Dans_sampler_s2 / CTsq_fast_s2) by the same rule, before it counts. A change helps the CTM Transformer
specifically if its CTsq arm passes and its Dans arm does not.

Usage: python -m scripts.tick_contribution --queue k/2 [--seed 1] | --summarize
"""
import argparse,json,os,subprocess,sys
from pathlib import Path

import scripts.per_token_pilot as pilot
import scripts.ctm_speed_check as speed

CHANGES=('film','mtp','m32');MODELS=('Dans','CTsq');OUT=pilot.OUT
HOURS={'film':1.0,'mtp':1.1,'m32':2.1}  # relative cost, for balancing the queues
REFERENCE={'Dans':'Dans_sampler','CTsq':'CTsq_fast'}


def name(model,change,seed):return f'{model}_{change}_s{seed}'


def config(model,change,seed):
    run=speed.config(seed) if model=='CTsq' else pilot.config('Dans_sampler',seed);t=run['train']
    run.update(name=name(model,change,seed),run_directory=str(pilot.RUNS/name(model,change,seed)));recipe=dict(run['rdt_recipe'])
    if change=='film':recipe['tick_film']=48
    if change=='mtp':recipe['mtp']=2
    if change=='m32':run['depth_sampler']={**run['depth_sampler'],'mean':32,'maximum':96};t['micro_batch_by_depth']=[[40,8]]
    run['rdt_recipe']=recipe;return run


def train(model,change,seed,gpu):
    d=pilot.RUNS/name(model,change,seed)
    if not (d/'complete.json').exists():
        d.mkdir(parents=True,exist_ok=True);(d/'config.json').write_text(json.dumps(config(model,change,seed),indent=2)+'\n')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
        with open(d/'train.log','a') as f:subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(d/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env)
        if not (d/'complete.json').exists():raise RuntimeError(f'{name(model,change,seed)}: training did not complete')
    import torch
    from scripts.exit_calibration import analyse
    analyse(name(model,change,seed),torch.device('cuda'),run_dir=d,out_dir=OUT,t_max=pilot.T_EVAL)
    torch.cuda.empty_cache()


def queue(k,n):
    jobs=sorted(((m,c) for c in CHANGES for m in MODELS),key=lambda j:HOURS[j[1]]*(1.1 if j[0]=='CTsq' else 1),reverse=True);loads=[0.0]*n;mine=[]
    for j in jobs:
        q=loads.index(min(loads));loads[q]+=HOURS[j[1]]
        if q==k:mine.append(j)
    return mine


def verdict(model,change,seed):
    f=OUT/f'{name(model,change,seed)}.json';ref=OUT/f'{REFERENCE[model]}_s{seed}.json'
    if not f.exists() or not ref.exists():return None
    x=json.loads(f.read_text())['fixed_ce'];r=json.loads(ref.read_text())['fixed_ce']
    pays=x['32']<=x['16']-0.01 and x['64']<=x['16']-0.01;cost=x['16']-r['16']
    return {'loss':{t:x[str(t)] for t in (1,4,8,16,32,64,96)},'reference16':r['16'],'pays':pays,'cost':cost,
            'passes':(pays and cost<=0.01) or cost<=-0.02}


def summarize():
    L=['# Can ticks beyond 8 contribute? (100M-token screen)','','Protocol and rules: `scripts/tick_contribution.py` (docstring).','',
       '| Arm | Seed | 1 | 4 | 8 | 16 | 32 | 64 | 96 | Reference at 16 | Ticks pay | vs reference | Passes |','|---|---:|'+'---:|'*8+'---|---:|---|']
    for c in CHANGES:
        for m in MODELS:
            for s in (1,2):
                v=verdict(m,c,s)
                if v:L.append(f'| {m}_{c} | {s} | '+' | '.join(f'{v["loss"][t]:.3f}' for t in (1,4,8,16,32,64,96))+f' | {v["reference16"]:.3f} | {"yes" if v["pays"] else "no"} | {v["cost"]:+.3f} | {"yes" if v["passes"] else "no"} |')
    L+=['','## Outcomes','']
    for c in CHANGES:
        state={m:[verdict(m,c,s) for s in (1,2)] for m in MODELS}
        text=[]
        for m in MODELS:
            v1,v2=state[m]
            text.append(f'{m}: '+('not run' if v1 is None else 'fails the screen' if not v1['passes'] else 'passes the screen; confirmation pending' if v2 is None else 'confirmed' if v2['passes'] else 'not confirmed on seed 2'))
        L.append(f'- **{c}:** '+'; '.join(text))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'TICK_CONTRIBUTION.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue');p.add_argument('--seed',type=int,default=1)
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    k,n=map(int,a.queue.split('/'));os.environ['CUDA_VISIBLE_DEVICES']=str(k);jobs=queue(k,n);print('queue',k,[name(m,c,a.seed) for m,c in jobs],flush=True)
    for m,c in jobs:train(m,c,a.seed,k)


if __name__=='__main__':main()
