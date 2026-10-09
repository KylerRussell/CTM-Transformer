"""RDT recipe study, round 1 (research/RDT_RECIPE.md, 2026-10-08).

Base: the RDT-heavy sweep run at d 704 (prelude/core/coda 2/4/2, 48.2M non-embedding
parameters), trained for 400M tokens (1,526 steps of 262,144 tokens, warmup 100, cosine to
10%, one GPU). The current recipe scores 3.783 with randomized depth and 3.828 at depth 1
(research/CTM_TICK_DIAGNOSTICS.md).

| Variant | Recipe (ctm_transformer/rdt_recipe.py) |
|---|---|
| B | pre-norm prelude and coda, sandwich norm only in the core; zero initial state; standard embedding init |
| C | pre-norm everywhere; RMSNorm on the injected prelude output and at the core exit; random initial state (Huginn) |

Round 1b (2026-10-09), after B collapsed and C trailed: D (pre-norm everywhere, zero state, no
extra norms) and E (C with a zero initial state), at depth 1 with lr 2e-3 (the Transformer's best),
then D with randomized depth.

All variants drop the unit-scale input embeddings (embedding_init_std 1.0), which were a fix for
sandwich norm in the prelude.

Phase 1, learning rate on the cheap depth-1 twins: each variant at depth 1 with lr 1e-3, 2e-3
and 4e-3, and a matched-unique-parameter Transformer (8 layers at d 704, the RDT's unique
layers) at 2e-3 and 4e-3.
Phase 2: each variant with the sweep's randomized depth (log-normal Poisson, mean 15) at
the learning rate of its best depth-1 run; if that run diverges or collapses (held-out loss
of at least 7), at half that rate.

Rules, fixed before the runs:
- **Adopted:** a variant whose randomized-depth loss is below the current recipe's 3.783 and
  at least 0.04 below its own depth-1 twin at the same rate (recurrence still pays).
- **Diagnosis:** a variant whose depth-1 loss is within 0.03 of the 8-layer Transformer's
  best has closed the recipe gap.

Usage: python -m scripts.rdt_recipe --gpu N --phase1 names... | --phase2 VARIANT | --summarize
"""
import argparse,json,math,os,subprocess,sys
from pathlib import Path

RUNS=Path('research/runs/rdt_recipe');OUT=Path('research/results/rdt_recipe')
BASE=Path('research/runs/lr_sweep/rdt_heavy_d704_h1_k+0')
VARIANTS={'B':{'norm':'core_sandwich','normalize':False,'state_init':'zeros'},
          'C':{'norm':'prenorm','normalize':True,'state_init':'random'},
          # Round 1b (2026-10-09): D is the 8-layer Transformer plus the injection adapter; E is C with a zero state.
          'D':{'norm':'prenorm','normalize':False,'state_init':'zeros'},
          'E':{'norm':'prenorm','normalize':True,'state_init':'zeros'}}
LRS=(1e-3,2e-3,4e-3)
TRANSFORMER_LRS=(2e-3,4e-3)
COLLAPSED=7.0


def name(variant,depth,lr):return f'{variant}_{"d1" if depth==1 else "rand"}_lr{lr:g}'


def base_run(run_name,lr):
    run=json.loads((BASE/'config.json').read_text());t=run['train']
    run.update(name=run_name,run_directory=str(RUNS/run_name))
    t.update(total_steps=1526,eval_interval=381,warmup_steps=100,lr=lr,delete_checkpoint_on_complete=False)
    return run


def config(run_name):
    if run_name.startswith('T8_'):
        lr=float(run_name.split('lr')[1]);run=base_run(run_name,lr)
        for k in ('depth_sampler','eval_depth','eval_depths','embedding_init_std'):run.pop(k,None)
        run['family']='transformer';run['model']={'d_model':704,'n_layers':8,'ffn_hidden_dim':1856,'init_std':run['model']['init_std']}
        return run
    variant,kind,lr=run_name.split('_');lr=float(lr[2:]);run=base_run(run_name,lr)
    run.pop('embedding_init_std',None);run['rdt_recipe']=VARIANTS[variant]
    if kind=='d1':run.pop('depth_sampler');run.update(eval_depth=1,eval_depths=[1])
    return run


def final(run_name):
    d=RUNS/run_name
    if (d/'diverged.json').exists():return math.inf
    if not (d/'complete.json').exists():return None
    rows=[json.loads(r) for r in (d/'metrics.jsonl').read_text().splitlines() if r.strip()]
    loss=next(iter([r for r in rows if 'validation' in r][-1]['validation'].values()))
    return math.inf if loss>=COLLAPSED else loss


def train(run_name,gpu):
    out=RUNS/run_name;out.mkdir(parents=True,exist_ok=True)
    if final(run_name) is not None:return final(run_name)
    (out/'config.json').write_text(json.dumps(config(run_name),indent=2)+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
    print('run',run_name,flush=True)
    with open(out/'train.log','a') as f:
        code=subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(out/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    print('run',run_name,'exit',code,final(run_name),flush=True)
    return final(run_name)


def phase2(variant,gpu):
    scores={lr:final(name(variant,1,lr)) for lr in LRS}
    if any(v is None for v in scores.values()):raise SystemExit(f'phase 1 of {variant} is incomplete: {scores}')
    lr=min(scores,key=scores.get);result=train(name(variant,16,lr),gpu)
    if result==math.inf:
        lr=lr/2;train(name(variant,1,lr),gpu);train(name(variant,16,lr),gpu)


def summarize():
    rows=[]
    for d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        if (d/'config.json').exists():rows.append((d.name,final(d.name)))
    L=['# RDT recipe study, round 1','','Protocol: `scripts/rdt_recipe.py` (docstring). Held-out loss at 400M tokens on 1,024 windows (randomized-depth runs evaluated at depth 16).','',
       'Current recipe (sandwich norm everywhere, unit embeddings): randomized depth 3.783, depth 1 3.828 (lr 1e-3).','','| Run | Held-out loss |','|---|---:|']
    L+=[f'| {n} | {"running" if v is None else ("diverged or collapsed" if v==math.inf else f"{v:.3f}")} |' for n,v in rows]
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'RDT_RECIPE.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int);p.add_argument('--phase1',nargs='*');p.add_argument('--phase2')
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    for run_name in a.phase1 or []:train(run_name,a.gpu)
    if a.phase2:phase2(a.phase2,a.gpu)


if __name__=='__main__':main()
