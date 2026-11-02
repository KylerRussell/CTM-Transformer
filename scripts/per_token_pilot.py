"""Stage B of research/PER_TOKEN_TICKS_DESIGN.md: per-token ticks in the language model, a 100M-token pilot (2026-11-02).

Stage A (scripts/parity_in_text.py) passed its rule for CTsq: with the parity problem's positions given ticks
that scale with its length, 4 of 10 CTsq seeds solved longer problems by thinking longer (uniform ticks: 0),
at no cost in text loss. This asks whether per-token ticks, with each token's difficulty measured by the
model itself, make extra ticks pay on web text, where every model so far is flat from about 8 ticks.

Arms (d704, 2 + 4 + 2 layers, as the 400M-token runs; 100M tokens: 381 updates of 262,144 tokens, warmup 50,
cosine to 10%, peak lr 2e-3; seeds 1 and 2, the same data order for every arm of a seed):
- Dans_sampler, CTsq_sampler: the usual tick sampler (log-normal Poisson, mean 15, at most 48);
- Dans_pertoken, CTsq_pertoken: per-token budgets (scripts/pretrain.py `per_token_ticks`): difficulty = the
  token's cross-entropy from a no-gradient 2-tick pass; budget 1 + round(24 q) + U{0..6} for the token's
  difficulty quantile q within its micro-batch (mean about 16 ticks per token, at most 31).

Evaluation (scripts/exit_calibration.analyse, extended to 96 ticks): 128 held-out windows (offset 40,000);
loss at every fixed tick count 1-96 for every position; the convergence exit (a token stops at the first tick
t >= 2 with KL(p_t || p_{t-1}) < eps, swept over eps), as (mean exit tick, loss) points.

Rules, fixed before any run. Two-seed means; a comparison holds only if both seeds agree in direction.
- Ticks pay: loss at 32 ticks and at 64 ticks each at least 0.01 below the loss at 16 ticks (both seeds).
- No cost: the per-token arm's 16-tick loss is no more than 0.01 above its sampler arm's.
- Better use of compute: at a mean of 4 and of 8 convergence-exit ticks (interpolated), the per-token arm's
  loss is at least 0.02 below its sampler arm's.
- Per-token ticks work for a model if "ticks pay" and "no cost" hold for its per-token arm.
- The CTM matters if CTsq's per-token gain exceeds Dans's by at least 0.02, on the loss at 32 ticks or under the
  exit at a mean of 8 ticks (gain = sampler arm minus per-token arm).
- Decision (research/PER_TOKEN_TICKS_DESIGN.md): if per-token ticks work, the next step is 400M tokens with two
  seeds for Dans and CTsq.

Usage: python -m scripts.per_token_pilot --queue k/2 | --summarize
"""
import argparse,json,math,os,subprocess,sys
from pathlib import Path

RUNS=Path('research/runs/per_token_pilot');OUT=Path('research/results/per_token_pilot')
BASE={'Dans':'research/runs/rdt_recipe/Dans_rand_lr0.002_s2/config.json','CTsq':'research/runs/ctm_transformer/CTsq_lr0.002_s2/config.json'}
ARMS=('Dans_sampler','CTsq_sampler','Dans_pertoken','CTsq_pertoken');SEEDS=(1,2)
PER_TOKEN={'difficulty_ticks':2,'scale':24,'jitter':6};T_EVAL=96;FIXED=(1,2,4,8,16,32,64,96)
HOURS={'Dans_sampler':2.6,'CTsq_sampler':3.9,'Dans_pertoken':5.6,'CTsq_pertoken':8.6}  # estimates, for balancing the queues


def name(arm,seed):return f'{arm}_s{seed}'


def config(arm,seed):
    model,kind=arm.split('_');run=json.loads(Path(BASE[model]).read_text());t=run['train']
    run.update(name=name(arm,seed),seed=seed,run_directory=str(RUNS/name(arm,seed)))
    t.update(total_steps=381,warmup_steps=50,eval_interval=381,eval_windows=64,final_eval_windows=256,checkpoint_interval=50,
             snapshot_interval=10**12,delete_checkpoint_on_complete=False)
    if kind=='pertoken':run['per_token_ticks']=PER_TOKEN
    return run


def train(arm,seed,gpu):
    d=RUNS/name(arm,seed)
    if (d/'complete.json').exists():return
    d.mkdir(parents=True,exist_ok=True);(d/'config.json').write_text(json.dumps(config(arm,seed),indent=2)+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
    print('train',name(arm,seed),flush=True)
    with open(d/'train.log','a') as f:code=subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(d/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    print('train',name(arm,seed),'exit',code,flush=True)
    if not (d/'complete.json').exists():raise RuntimeError(f'{name(arm,seed)}: training did not complete (exit {code})')


def evaluate(arm,seed):
    import torch
    from scripts.exit_calibration import analyse
    analyse(name(arm,seed),torch.device('cuda'),run_dir=RUNS/name(arm,seed),out_dir=OUT,t_max=T_EVAL)
    torch.cuda.empty_cache()


def queue(k,n):
    jobs=sorted(((a,s) for a in ARMS for s in SEEDS),key=lambda j:HOURS[j[0]],reverse=True);loads=[0.0]*n;mine=[]
    for j in jobs:
        q=loads.index(min(loads));loads[q]+=HOURS[j[0]]
        if q==k:mine.append(j)
    return mine


# ---------------------------------------------------------------- summary

def measures(r):
    from scripts.exit_calibration import envelope_at
    m={f'fixed{t}':r['fixed_ce'][str(t)] for t in FIXED}
    conv=list(r['convergence_exit'].values())
    for t in (4,8):m[f'exit{t}']=envelope_at(conv,t)
    return m


def summarize():
    res={(a,s):measures(json.loads((OUT/f'{name(a,s)}.json').read_text())) for a in ARMS for s in SEEDS if (OUT/f'{name(a,s)}.json').exists()}
    mean=lambda v:sum(v)/len(v)
    L=['# Per-token ticks: language-model pilot (100M tokens)','','Protocol and rules: `scripts/per_token_pilot.py` (docstring). Loss in nats; two-seed means (per seed in brackets).','',
       '| Measure | '+' | '.join(ARMS)+' |','|---|'+'---:|'*len(ARMS)]
    keys=[f'fixed{t}' for t in FIXED]+['exit4','exit8']
    for k in keys:
        row=[]
        for a in ARMS:
            v=[res[(a,s)][k] for s in SEEDS if (a,s) in res]
            row.append(f'{mean(v):.3f} ('+'/'.join(f'{x:.3f}' for x in v)+')' if v else '')
        L.append(f'| {k.replace("fixed","loss at ").replace("exit","exit, mean ")} | '+' | '.join(row)+' |')
    L+=['','## Rule outcomes','']
    complete=lambda *arms:all((a,s) in res for a in arms for s in SEEDS)
    def pays(a):return all(res[(a,s)][f'fixed{t}']<=res[(a,s)]['fixed16']-0.01 for s in SEEDS for t in (32,64))
    gains={}
    for m in ('Dans','CTsq'):
        pt,sm=f'{m}_pertoken',f'{m}_sampler'
        if not complete(pt,sm):L.append(f'- **{m}:** incomplete');continue
        cost=mean([res[(pt,s)]['fixed16'] for s in SEEDS])-mean([res[(sm,s)]['fixed16'] for s in SEEDS])
        better={t:mean([res[(sm,s)][f'exit{t}']-res[(pt,s)][f'exit{t}'] for s in SEEDS]) for t in (4,8)}
        better_ok={t:better[t]>=0.02 and all(res[(sm,s)][f'exit{t}']>res[(pt,s)][f'exit{t}'] for s in SEEDS) for t in (4,8)}
        works=pays(pt) and cost<=0.01
        gains[m]={'fixed32':[res[(sm,s)]['fixed32']-res[(pt,s)]['fixed32'] for s in SEEDS],'exit8':[res[(sm,s)]['exit8']-res[(pt,s)]['exit8'] for s in SEEDS]}
        L.append(f'- **{m}:** ticks pay (per-token arm): {"yes" if pays(pt) else "no"} (sampler arm: {"yes" if pays(sm) else "no"}); '
                 f'16-tick cost {cost:+.3f} ({"ok" if cost<=0.01 else "fails"}); exit gain at mean 4 / 8 ticks {better[4]:+.3f} / {better[8]:+.3f} '
                 f'(better use of compute: {"yes" if all(better_ok.values()) else "no"}); per-token ticks work: {"yes" if works else "no"}')
    if len(gains)==2:
        ctm=any(mean(gains['CTsq'][k])-mean(gains['Dans'][k])>=0.02 and all(c>d for c,d in zip(gains['CTsq'][k],gains['Dans'][k])) for k in ('fixed32','exit8'))
        L.append(f'- **The CTM matters (CTsq per-token gain exceeds Dans by at least 0.02):** {"yes" if ctm else "no"}')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'PER_TOKEN_PILOT.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    k,n=map(int,a.queue.split('/'));jobs=queue(k,n);print('queue',k,[name(*j) for j in jobs],flush=True)
    os.environ['CUDA_VISIBLE_DEVICES']=str(k)
    for arm,seed in jobs:train(arm,seed,k);evaluate(arm,seed)


if __name__=='__main__':main()
