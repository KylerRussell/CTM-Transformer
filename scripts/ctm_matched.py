"""The CTM arm matched to the RDT's best (research/RDT_RECIPE.md, 2026-10-13).

The paper compares the CTM's best with the RDT's best, with the Transformer as the baseline.
The RDT arm is Dans (47.2M non-embedding parameters, d 704: 3.569 and 3.579 on two seeds at
400M tokens; its depth-1 twin is the 8-layer Transformer, 3.618 and 3.638). This study brings
the CTM to the same standard.

Model: the best language-modelling CTM so far (token-conditioned start, sparse tick loss,
cross-position ticks, residual readout; research/runs/tick_use/RS_400M) at d 704 with a 4-layer
backbone, 1,408 neurons and 1,408 synchronization pairs (U-Net width 2,816): 45.1M
non-embedding parameters. The 4 backbone layers mirror Dans's 4 non-recurrent layers
(prelude 2, coda 2). Same data, seed, schedule (1,526 steps of 262,144 tokens, warmup 100,
cosine to 10%) and evaluation as Dans.

Phase 1: 1 tick at lr 1e-3, 2e-3 and 4e-3. Phase 2: 16 ticks at the rate of the best 1-tick
run (half that rate if it diverges or collapses), and a second seed (2) of both the 1-tick and
16-tick runs at that rate.

Rules, fixed before the runs:
- **Ticks contribute** if the 16-tick run is at least 0.04 below its 1-tick twin on both seeds
  (Dans's threshold against its depth-1 twin).
- **CTM against RDT:** the mean over the two seeds is compared with Dans's mean (3.574); a
  difference of less than 0.02 (about one seed spread) is reported as a tie.

Usage: python -m scripts.ctm_matched --gpu N --phase1 names... [--then main | seed2] | --summarize
"""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path

RUNS=Path('research/runs/ctm_matched');OUT=Path('research/results/ctm_matched')
BASE=Path('research/runs/tick_use/RS_400M')
MODEL={'d_model':704,'n_heads':11,'n_layers':4,'d_latent':1408,'sync_sparse_pairs':1408}
LRS=(1e-3,2e-3,4e-3)
COLLAPSED=7.0
DANS=(3.569,3.579)


def name(ticks,lr,seed=None):return f'M_T{ticks}_lr{lr:g}'+(f'_s{seed}' if seed else '')


def config(run_name):
    parts=run_name.split('_');ticks=int(parts[1][1:]);lr=float(parts[2][2:])
    run=json.loads((BASE/'config.json').read_text())
    run.update(name=run_name,run_directory=str(RUNS/run_name),eval_depth=ticks)
    if len(parts)>3:run['seed']=int(parts[3][1:])
    run['model'].update(MODEL,max_thought_steps=ticks)
    run['train'].update(lr=lr,delete_checkpoint_on_complete=False)
    return run


def final(run_name):
    d=RUNS/run_name
    if (d/'diverged.json').exists():return math.inf
    if not (d/'complete.json').exists():return None
    rows=[json.loads(r) for r in (d/'metrics.jsonl').read_text().splitlines() if r.strip()]
    loss=[r for r in rows if 'validation' in r][-1]['validation']['final_tick']
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


def best_lr():
    """Wait until every phase-1 run has finished (on either GPU), then return the best rate."""
    while any(final(name(1,lr)) is None for lr in LRS):time.sleep(300)
    scores={lr:final(name(1,lr)) for lr in LRS}
    if all(v==math.inf for v in scores.values()):raise SystemExit(f'every 1-tick run diverged or collapsed: {scores}')
    return min(scores,key=scores.get)


def main_run(gpu):
    lr=best_lr()
    if train(name(16,lr),gpu)==math.inf:
        lr=lr/2;train(name(1,lr),gpu);train(name(16,lr),gpu)


def seed2(gpu):
    lr=best_lr();train(name(1,lr,2),gpu);train(name(16,lr,2),gpu)


def summarize():
    rows=[(d.name,final(d.name)) for d in sorted(RUNS.iterdir()) if (d/'config.json').exists()] if RUNS.exists() else []
    L=['# Matched CTM arm','','Protocol: `scripts/ctm_matched.py` (docstring). Held-out loss at 400M tokens on 1,024 windows, final tick.','',
       f'References: Dans {DANS[0]} / {DANS[1]} (mean {sum(DANS)/2:.3f}); 8-layer Transformer 3.618 / 3.638.','','| Run | Held-out loss |','|---|---:|']
    L+=[f'| {n} | {"running" if v is None else ("diverged or collapsed" if v==math.inf else f"{v:.3f}")} |' for n,v in rows]
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'CTM_MATCHED.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int);p.add_argument('--phase1',nargs='*')
    p.add_argument('--then',choices=('main','seed2'));p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    for run_name in a.phase1 or []:train(run_name,a.gpu)
    if a.then=='main':main_run(a.gpu)
    if a.then=='seed2':seed2(a.gpu)


if __name__=='__main__':main()
