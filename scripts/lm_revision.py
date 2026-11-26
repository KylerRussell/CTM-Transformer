"""Language-model revision runs (2026-11-21, after external review, concerns 3 and 4).

The review: the "ties" with Dans rest on two seeds and a superiority-only rule; the learning-rate grid {1e-3, 2e-3,
4e-3} is coarse against the 0.02 margin and was searched only for Dans and CTsq; the 8-layer Transformer baseline is
not compute-matched. Arms: Dans, CTsq, CTsqS (the CTM Transformer's best), T8 (8-layer Transformer, Dans's parameters),
T20 (20-layer Transformer, Dans's compute at 4 ticks). All at 400M tokens unless stated, the existing recipe and data,
CTM variants compiled (`compile_ctm`, adopted after scripts/ctm_speed_check.py). Measure: held-out loss on the 1,024
final-evaluation windows (validation 40,000-41,023) at 16 ticks (Transformers: one pass), the training run's final
evaluation, identical to scripts/lm_unified_eval.py at 16 ticks.

Phase A, finer learning-rate grid (seed 1234): every arm at lr 1.4e-3 and 2.8e-3 (half-steps around 2e-3, the best of
the old grid for every arm). Rules, fixed before any run:
- An arm's selected lr is the one with the lowest loss among all its seed-1234 runs, except that 2e-3 is kept unless
  another rate is lower by at least 0.005 (within seed noise, about 0.007-0.01, 2e-3 keeps its two existing seeds).
- If the selected lr is the lowest or highest rate tried for the arm, the next half-step out is run and the rule
  re-applied (only T8 lacks 1e-3).

Phase B, seeds: each arm at its selected lr on seeds 1234, 2, 3, 4, 5 (5 seeds; existing runs reused).

Phase C, compute-matched Transformer (T8x): T8 at its selected lr, trained for as many tokens as matches Dans's
training compute: steps = round(1526 x (2 + 4 x mean depth + 2) / 8), with mean depth the mean of Dans's sampled tick
counts over its 1,526 steps, the same warmup (100) and cosine schedule over the longer run; seeds 1234, 2, 3.

Rules for the comparisons (fixed before any phase-B or phase-C run; Welch t-tests on per-seed losses):
- Equivalence: a CTM arm is equivalent to Dans if the 90% confidence interval of the difference of means lies inside
  (-0.02, +0.02) (TOST, margin 0.02 nats, alpha 0.05 per side). It is worse than Dans if it is higher with one-sided
  p < 0.05, and beats Dans if at least 0.02 lower with one-sided p < 0.05 (Holm over CTsq and CTsqS in each case).
  Otherwise the comparison is inconclusive at this power, and is reported as such.
- Dans beats T8 (parameter-matched) if at least 0.02 lower with one-sided p < 0.05; likewise against T8x (parameter-
  and compute-matched). The T8 margin is presented as a win per unit of compute only if Dans also beats T8x.
- Reported: Dans at 4 ticks against T20 (equal compute), difference and 95% interval; every arm's mean and 95% interval.
- Other variants (CTfull, CTnlm, CTso, CTsqH, CTsqLD) stay at two seeds and lr 2e-3, and are reported as untuned screens.

Usage: python -m scripts.lm_revision --queue --gpu N     (a worker: runs ready runs in priority order, one at a time)
       | --summarize
"""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path

RUNS=Path('research/runs/revision');OUT=Path('research/results/lm_revision')
ARMS=('Dans','CTsq','CTsqS','T8','T20');SEEDS=(1234,2,3,4,5);T8X_SEEDS=(1234,2,3);GRID=(1e-3,1.4e-3,2e-3,2.8e-3,4e-3)
BASE={'Dans':'research/runs/rdt_recipe/Dans_rand_lr0.002','CTsq':'research/runs/ctm_transformer/CTsq_lr0.002','CTsqS':'research/runs/ctm_transformer/CTsqS_lr0.002',
      'T8':'research/runs/rdt_recipe/T8_lr0.002','T20':'research/runs/rdt_recipe/T20_lr0.002'}
OLD={('Dans',1e-3,1234):'rdt_recipe/Dans_rand_lr0.001',('Dans',2e-3,1234):'rdt_recipe/Dans_rand_lr0.002',('Dans',2e-3,2):'rdt_recipe/Dans_rand_lr0.002_s2',('Dans',4e-3,1234):'rdt_recipe/Dans_rand_lr0.004',
     ('CTsq',1e-3,1234):'ctm_transformer/CTsq_lr0.001',('CTsq',2e-3,1234):'ctm_transformer/CTsq_lr0.002',('CTsq',2e-3,2):'ctm_transformer/CTsq_lr0.002_s2',('CTsq',4e-3,1234):'ctm_transformer/CTsq_lr0.004',
     ('CTsqS',2e-3,1234):'ctm_transformer/CTsqS_lr0.002',('CTsqS',2e-3,2):'ctm_transformer/CTsqS_lr0.002_s2',
     ('T8',2e-3,1234):'rdt_recipe/T8_lr0.002',('T8',2e-3,2):'rdt_recipe/T8_lr0.002_s2',('T8',4e-3,1234):'rdt_recipe/T8_lr0.004',
     ('T20',1e-3,1234):'rdt_recipe/T20_lr0.001',('T20',2e-3,1234):'rdt_recipe/T20_lr0.002',('T20',4e-3,1234):'rdt_recipe/T20_lr0.004'}
COST={'Dans':17,'CTsq':18.5,'CTsqS':18.5,'T8':2.2,'T20':3.4,'T8x':18}  # GPU-hours, for ordering only
NEED_GIB=21  # an essentially empty GPU: recall workers peak at 2-3 GB each, so LM runs queue behind them


def name(arm,lr,seed):return f'{arm}_lr{lr:g}_s{seed}'


def directory(arm,lr,seed):
    return Path('research/runs')/OLD[(arm,lr,seed)] if (arm,lr,seed) in OLD else RUNS/name(arm,lr,seed)


def loss(d):
    """Final held-out loss (1,024 windows; 16 ticks or one pass), or None if the run is not complete."""
    if not (d/'complete.json').exists():return None
    rows=[json.loads(r) for r in (d/'metrics.jsonl').read_text().splitlines() if r.strip()]
    v=[r for r in rows if 'validation' in r][-1]['validation'];return v.get('depth_16',v.get('final'))


def dans_mean_depth():
    rows=[json.loads(r) for r in (Path(BASE['Dans'])/'metrics.jsonl').read_text().splitlines() if r.strip()]
    from ctm_transformer.depth_sampling import lognormal_poisson
    from ctm_transformer.pretrain import step_depth
    s=lognormal_poisson(15,0.5,48);return sum(step_depth(s,1234,k) for k in range(1526))/1526


def grid(arm):
    """(tried {lr: loss or None}, selected lr or None, edge run needed or None) for the phase-A rule on seed 1234."""
    tried={lr:loss(directory(arm,lr,1234)) for lr in GRID if (arm,lr,1234) in OLD or lr in (1.4e-3,2.8e-3) or directory(arm,lr,1234).exists()}
    if any(v is None for v in tried.values()):return tried,None,None
    best=min(tried,key=tried.get);lr=best if tried[best]<=tried[2e-3]-0.005 else 2e-3
    if lr!=2e-3 and lr in (min(tried),max(tried)):
        k=GRID.index(lr)+(-1 if lr==min(tried) else 1)
        if 0<=k<len(GRID):return tried,None,GRID[k]  # GRID[k] is untried, else lr would not be at the edge
    return tried,lr,None


def selected(arm):
    _,lr,_=grid(arm);return lr,lr is not None


def edge_runs(arm):
    _,_,e=grid(arm);return [(arm,e,1234)] if e else []


def plan():
    """Ready runs in priority order: phase A, then edge runs, then phase B, then phase C."""
    out=[(a,lr,1234) for lr in (1.4e-3,2.8e-3) for a in ARMS]
    for a in ARMS:out+=edge_runs(a)
    for a in ARMS:
        lr,ok=selected(a)
        if ok:out+=[(a,lr,s) for s in SEEDS]
    lr,ok=selected('T8')
    if ok:out+=[('T8x',lr,s) for s in T8X_SEEDS]
    seen=set();ready=[]
    for r in out:
        if r in seen or r in OLD or loss(directory(*r)) is not None:continue
        seen.add(r);ready.append(r)
    return ready


def config(arm,lr,seed):
    base=arm.removesuffix('x');run=json.loads((Path(BASE[base])/'config.json').read_text());t=run['train']
    d=RUNS/name(arm,lr,seed);run.update(name=name(arm,lr,seed),run_directory=str(d),seed=seed);t['lr']=lr
    if run.get('ctm_transformer') is not None:t['compile_ctm']=True;t.pop('micro_batch_by_depth',None)
    t['snapshot_interval']=t.get('snapshot_interval',400)
    if arm=='T8x':
        steps=round(1526*(2+4*dans_mean_depth()+2)/8);t['total_steps']=steps;run['compute_matched_to']='Dans (mean training depth)'
    return run


def worker(gpu):
    import fcntl
    RUNS.mkdir(parents=True,exist_ok=True)
    while True:
        with open(RUNS/'.queue.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);ready=[r for r in plan() if not (RUNS/name(*r)/'claimed').exists()]
            if not ready:return print('queue empty',flush=True)
            r=ready[0];d=RUNS/name(*r);d.mkdir(parents=True,exist_ok=True);(d/'claimed').write_text(f'gpu {gpu} {time.ctime()}\n')
        while True:  # wait until this GPU has room (recall workers or evaluations may be using it)
            free=int(subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',str(gpu)],capture_output=True,text=True).stdout.strip() or 0)/1024
            if free>=NEED_GIB:break
            time.sleep(120)
        (d/'config.json').write_text(json.dumps(config(*r),indent=2)+'\n')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8',
             'TORCHINDUCTOR_CACHE_DIR':str(Path.home()/f'.cache/torchinductor_lmrev_gpu{gpu}')}
        print('run',name(*r),'gpu',gpu,time.ctime(),flush=True)
        with open(d/'train.log','a') as f:
            code=subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(d/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
        print('run',name(*r),'exit',code,loss(d),time.ctime(),flush=True)
        if code!=0:(d/'claimed').unlink(missing_ok=True);(d/'failed').write_text(f'exit {code}\n');return


def summarize():
    from scipy import stats
    L=['# Language-model revision','Protocol and rules: `scripts/lm_revision.py`. Held-out loss, 1,024 windows, 16 ticks (Transformers: one pass).','',
       '## Phase A, learning-rate grid (seed 1234)','','| Arm | '+' | '.join(f'{lr:g}' for lr in GRID)+' | Selected |','|---|'+'---:|'*len(GRID)+'---|']
    sel={}
    for a in ARMS:
        cells=[f'{v:.4f}' if (v:=loss(directory(a,lr,1234)) if ((a,lr,1234) in OLD or directory(a,lr,1234).exists()) else None) is not None else '-' for lr in GRID]
        lr,ok=selected(a);sel[a]=lr if ok else None;L.append(f'| {a} | '+' | '.join(cells)+f" | {f'{lr:g}' if ok else 'pending'} |")
    vals={}
    for a in ARMS:
        if sel[a]:vals[a]=[v for s in SEEDS if (v:=loss(directory(a,sel[a],s))) is not None]
    if sel.get('T8'):vals['T8x']=[v for s in T8X_SEEDS if (v:=loss(RUNS/name('T8x',sel['T8'],s))) is not None]
    mean=lambda v:sum(v)/len(v)
    def ci(v,level=0.95):
        if len(v)<2:return float('nan'),float('nan')
        m=mean(v);se=stats.sem(v);h=se*stats.t.ppf(0.5+level/2,len(v)-1);return m-h,m+h
    def welch(a,b,level=0.90):
        """Difference a-b, its two-sided CI at `level` (Welch), and one-sided p for a<b and a>b."""
        d=mean(a)-mean(b);va,vb=stats.tvar(a)/len(a),stats.tvar(b)/len(b);se=(va+vb)**0.5
        df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1));h=se*stats.t.ppf(0.5+level/2,df)
        return d,(d-h,d+h),stats.t.cdf(d/se,df),stats.t.sf(d/se,df),se,df
    L+=['','## Phase B and C, seeds','','| Arm | lr | Seeds | Mean | 95% interval |','|---|---:|---:|---:|---|']
    for a,v in vals.items():
        lo,hi=ci(v);L.append(f"| {a} | {sel['T8' if a=='T8x' else a]:g} | {len(v)} | {mean(v):.4f} | {lo:.4f} to {hi:.4f} |" if v else f'| {a} | - | 0 | - | - |')
    full=lambda a:a in vals and len(vals[a])==(len(T8X_SEEDS) if a=='T8x' else len(SEEDS))
    if full('Dans'):
        L+=['','## Rule outcomes','']
        tests=[a for a in ('CTsq','CTsqS') if full(a)];res={a:welch(vals[a],vals['Dans']) for a in tests}
        def holm(ps):
            order=sorted(range(len(ps)),key=lambda i:ps[i]);adj=[0.0]*len(ps);run=0.0
            for j,i in enumerate(order):run=max(run,min(1.0,(len(ps)-j)*ps[i]));adj[i]=run
            return adj
        worse=dict(zip(tests,holm([res[a][3] for a in tests])))
        better=dict(zip(tests,holm([stats.t.cdf((res[a][0]+0.02)/res[a][4],res[a][5]) for a in tests])))
        for a in tests:
            d,(lo,hi),_,_,_,_=res[a];eq=-0.02<lo and hi<0.02
            verdict='beats Dans' if better[a]<0.05 else 'worse than Dans' if worse[a]<0.05 else 'equivalent to Dans' if eq else 'inconclusive at this power'
            L.append(f'- **{a} against Dans:** {verdict} (difference {d:+.4f}, 90% CI {lo:+.4f} to {hi:+.4f}; Holm p worse {worse[a]:.3f}, p beats-by-0.02 {better[a]:.3f})')
        for b in ('T8','T8x'):
            if full(b):
                d,_,_,_,se,df=welch(vals['Dans'],vals[b]);p=stats.t.cdf((d+0.02)/se,df);_,(lo,hi),*_=welch(vals['Dans'],vals[b],0.95)
                L.append(f"- **Dans against {b}:** {'Dans beats it' if p<0.05 else 'Dans does not beat it by 0.02'} (difference {d:+.4f}, 95% CI {lo:+.4f} to {hi:+.4f}, p = {p:.3f})")
    if full('T20'):  # reported (not a rule): Dans at 4 ticks against T20, whose compute equals Dans at 4 ticks
        U=Path('research/results/lm_unified');d4=[json.loads(f.read_text())['loss']['4'] for s in SEEDS if (f:=U/f"{directory('Dans',sel['Dans'],s).name}.json").exists()]
        if len(d4)==len(SEEDS):
            d,_,_,_,se,df=welch(d4,vals['T20']);_,(lo,hi),*_=welch(d4,vals['T20'],0.95)
            L.append(f"- Reported, Dans at 4 ticks ({mean(d4):.4f}) against T20 ({mean(vals['T20']):.4f}), equal compute: difference {d:+.4f}, 95% CI {lo:+.4f} to {hi:+.4f}")
        else:L.append(f'- Reported, Dans at 4 ticks against T20: pending ({len(d4)} of {len(SEEDS)} Dans seeds evaluated at 4 ticks)')
    (OUT/'LM_REVISION.md').parent.mkdir(parents=True,exist_ok=True);(OUT/'LM_REVISION.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue',action='store_true');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--summarize',action='store_true');p.add_argument('--plan',action='store_true');a=p.parse_args()
    if a.summarize:summarize()
    elif a.plan:print('\n'.join(name(*r) for r in plan()))
    elif a.queue:worker(a.gpu)


if __name__=='__main__':main()
