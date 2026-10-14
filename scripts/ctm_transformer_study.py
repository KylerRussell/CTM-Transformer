"""CTM Transformer on language modelling (research/CTM_TRANSFORMER_DESIGN.md, 2026-10-14).

Base: Dans (research/runs/rdt_recipe/Dans_rand_lr0.002): d 704, layers 2 + 4 thinking + 2, randomized
tick count (log-normal Poisson, mean 15, at most 48), final-tick loss, 400M tokens (1,526 steps of
262,144 tokens, warmup 100, cosine to 10%), lr 2e-3, the same data and evaluation. The CTM
Transformer adds its mechanisms (ctm_transformer/ctm_transformer.py), zero-initialized, so it
starts as Dans and, at one tick, as the 8-layer Transformer.

| Variant | Mechanisms |
|---|---|
| CTfull | neuron_models + sync_query + sync_output |
| CTnlm, CTsq, CTso | one mechanism each (ablations, after CTfull) |

Run names: <variant>_lr<lr>[_s<seed>]. References: Dans 3.569 / 3.579 (mean 3.574), the 8-layer
Transformer 3.618 / 3.638, the 20-layer Transformer (equal compute to 4 ticks) 3.562.

Rule, fixed before the runs: the CTM mechanisms add value if CTfull's two-seed mean is at least
0.02 below Dans's mean (3.574); a smaller difference is a tie. Liveness: `--diagnose` removes each
mechanism at evaluation (its zero-initialized parameter set back to zero) and reports the loss
increase; a mechanism whose removal costs less than 0.005 is reported as unused.

Usage: python -m scripts.ctm_transformer_study --gpu N --runs NAMES... | --diagnose RUN [--checkpoint F] | --summarize
"""
import argparse,json,math,os,subprocess,sys
from pathlib import Path

RUNS=Path('research/runs/ctm_transformer');OUT=Path('research/results/ctm_transformer')
VARIANTS={'CTfull':['neuron_models','sync_query','sync_output'],'CTnlm':['neuron_models'],'CTsq':['sync_query'],'CTso':['sync_output']}
MICRO={'CTfull':[[28,8]],'CTnlm':[[28,8]],'CTsq':[[28,8]],'CTso':[[28,8]]}  # micro-batch 16, and 8 for steps deeper than 28 ticks (memory)
COLLAPSED=7.0


def config(run_name):
    from scripts.rdt_recipe import config as recipe_config
    parts=run_name.split('_');variant=parts[0];lr=float(parts[1][2:])
    run=recipe_config('Dans_rand_lr0.002')
    run.update(name=run_name,run_directory=str(RUNS/run_name),ctm_transformer=VARIANTS[variant])
    if len(parts)>2:run['seed']=int(parts[2][1:])
    run['train']['lr']=lr
    if variant in MICRO:run['train']['micro_batch_by_depth']=MICRO[variant]
    run['train']['snapshot_interval']=381  # step_0000381.pt etc. for the liveness diagnostics
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


def diagnose(run_name,checkpoint=None,windows=8,depths=(1,4,8,16,32)):
    """Loss by tick count, and the loss increase when each mechanism is removed at evaluation."""
    import torch
    from scripts.pretrain import build_config
    from ctm_transformer.ctm_transformer import ctm_transformer_factory
    from ctm_transformer.pretrain import TokenWindows,evaluate_lm
    run=json.loads((RUNS/run_name/'config.json').read_text());t=run['train']
    data=json.loads(Path(run['data']).read_text());root=Path(run['data']).parent
    model=ctm_transformer_factory(set(run['ctm_transformer']),run['rdt_recipe'])(build_config(run,data['vocab_size']))
    name=checkpoint or (f'step_{t["total_steps"]:07d}.pt' if (RUNS/run_name/f'step_{t["total_steps"]:07d}.pt').exists() else 'latest.pt')
    payload=torch.load(RUNS/run_name/name,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'])
    device=torch.device('cuda');model.to(device)
    valid=TokenWindows([root/s for s in data['validation']],t['seq_len'],0);offset=t.get('eval_offset',0)
    indices=list(range(offset,offset+windows))
    with torch.no_grad():
        by_depth=evaluate_lm(model,valid,indices,1,device,'rdt',depths=tuple(depths))
        full=by_depth['depth_16'] if 'depth_16' in by_depth else evaluate_lm(model,valid,indices,1,device,'rdt',depths=(16,))['depth_16']
        removed={}
        params={'neuron_models':['nlm_scale'],'sync_query':[f'sync_queries.{i}.weight' for i in range(len(model.core))],'sync_output':['sync_projection.weight']}
        state={k:v.clone() for k,v in model.state_dict().items()}
        for mechanism in run['ctm_transformer']:
            for p in params[mechanism]:dict(model.named_parameters())[p].zero_()
            removed[mechanism]=evaluate_lm(model,valid,indices,1,device,'rdt',depths=(16,))['depth_16']-full
            model.load_state_dict(state)
        sizes={m:sum(float(dict(model.named_parameters())[p].float().pow(2).sum()) for p in params[m])**0.5 for m in run['ctm_transformer']}
    result={'run':run_name,'step':payload['step'],'windows':windows,'loss_by_depth':{k.split('_')[1]:v for k,v in by_depth.items()},
            'loss_increase_when_removed_at_16':removed,'unused':[m for m,v in removed.items() if v<0.005],'parameter_norm':sizes}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'diagnose_{run_name}_step{payload["step"]}.json').write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps(result,indent=1))


def summarize():
    rows=[(d.name,final(d.name)) for d in sorted(RUNS.iterdir()) if (d/'config.json').exists()] if RUNS.exists() else []
    L=['# CTM Transformer on language modelling','','Protocol: `scripts/ctm_transformer_study.py` (docstring). Held-out loss at 400M tokens on 1,024 windows, 16 ticks.','',
       'References: Dans 3.569 / 3.579 (mean 3.574); 8-layer Transformer 3.618 / 3.638; 20-layer Transformer 3.562.','','| Run | Held-out loss |','|---|---:|']
    L+=[f'| {n} | {"running" if v is None else ("diverged or collapsed" if v==math.inf else f"{v:.3f}")} |' for n,v in rows]
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'CTM_TRANSFORMER.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int);p.add_argument('--runs',nargs='*')
    p.add_argument('--diagnose');p.add_argument('--checkpoint');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    if a.diagnose:diagnose(a.diagnose,a.checkpoint);return
    for run_name in a.runs or []:train(run_name,a.gpu)


if __name__=='__main__':main()
