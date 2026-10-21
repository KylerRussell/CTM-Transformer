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

Ablations (added 2026-10-15, after CTfull scored 3.586 / 3.590, mean 3.588): CTsq, CTso and CTnlm,
each on seeds 1234 and 2. Rules, fixed before these runs, on two-seed means:
- a variant beats Dans if it is at least 0.02 below 3.574;
- against CTfull (3.588), a difference under 0.02 is a tie. A single-mechanism variant that ties or beats
  CTfull carries the full model's effect at lower cost.

Item 3 (added 2026-10-18): CTsqS (CTsq + the CMM attention sink) and CTsqL (CTsq trained with the CTM
loss: per token, the mean of the cross-entropy at the minimum-loss tick and at the most certain tick).
Two seeds each. Rules, fixed before these runs, on two-seed means:
- a variant improves on CTsq if it is at least 0.02 below 3.582; it beats Dans if it is at least 0.02 below 3.574.
- For CTsqL, certainty becomes a usable stopping signal (scripts/adaptive_compute.py) if threshold exit at
  a mean of 4 ticks is within 0.02 of its fixed 4-tick loss.

CTsqL failed (2026-10-20): both seeds collapsed from about step 40 (research/CTM_TRANSFORMER_DESIGN.md).
Pilots (added 2026-10-21): run names ending in `_pilot` train with per-step diagnostics (state RMS per tick,
selected-tick shares, gradient norm per module) and stop at step 200 on the full 1,526-step schedule. They are:
- CTsqL_lr0.002_pilot: reproduces the collapse with diagnostics, to find what starts it;
- CTsqLF_lr0.002_pilot (fix a): CTsq trained with the mean of the CTM loss and the final-tick loss;
- CTsqL_lr0.001_pilot (fix b): the CTM loss at half the learning rate.
Rule, fixed before the pilots: a fix passes if its mean training loss over steps 191-200 is within 0.1 of
CTsq's over the same steps (same seed, so the same tick counts), and no gradient norm after step 100 exceeds 10.
A passing fix then runs on two seeds at 400M tokens under the item-3 rules.
All three failed (2026-10-21; research/CTM_TRANSFORMER_DESIGN.md): with the CTM loss the state's RMS grows to
40-76 by step 200 (healthy: 6-7). Second pilots, same rule:
- CTsqLD_lr0.002_pilot (fix c): the mean of the CTM loss on detached readouts (it trains only the coda and the
  output head) and the full-gradient final-tick loss (it trains the recurrence as in Dans);
- CTsqLN_lr0.002_pilot (fix d): the CTM loss alone with the shared norm on the coda input (coda_norm 'shared').

Usage: python -m scripts.ctm_transformer_study --gpu N --runs NAMES... | --diagnose RUN [--checkpoint F] | --summarize | --pilots
"""
import argparse,json,math,os,subprocess,sys
from pathlib import Path

RUNS=Path('research/runs/ctm_transformer');OUT=Path('research/results/ctm_transformer')
VARIANTS={'CTfull':['neuron_models','sync_query','sync_output'],'CTnlm':['neuron_models'],'CTsq':['sync_query'],'CTso':['sync_output'],
          # item 3 (2026-10-18): the CMM attention sink and the CTM loss, on CTsq
          'CTsqS':['sync_query','attention_sink'],'CTsqL':['sync_query','ctm_loss'],
          # fix (a) for CTsqL's collapse (2026-10-21): the mean of the CTM loss and the final-tick loss
          'CTsqLF':['sync_query','ctm_loss','final_tick_loss'],
          # fixes (c) and (d) (2026-10-21): CTM loss on detached readouts; the CTM loss with a norm on the coda input
          'CTsqLD':['sync_query','ctm_loss','final_tick_loss','detached_readouts'],'CTsqLN':['sync_query','ctm_loss']}
RECIPE_OVERRIDES={'CTsqLN':{'coda_norm':'shared'}}
MICRO={'CTfull':[[28,8]],'CTnlm':[[28,8]],'CTsq':[[28,8]],'CTso':[[28,8]],'CTsqS':[[36,8]],'CTsqL':[[28,8]],'CTsqLF':[[28,8]],'CTsqLD':[[28,8]],'CTsqLN':[[28,8]]}  # micro-batch 16, and 8 for steps deeper than the limit (memory; item 3 limits from the 2026-10-19 memory test)
COLLAPSED=7.0
PILOT_STEPS=200


def config(run_name):
    from scripts.rdt_recipe import config as recipe_config
    pilot=run_name.endswith('_pilot');parts=run_name.removesuffix('_pilot').split('_');variant=parts[0];lr=float(parts[1][2:])
    run=recipe_config('Dans_rand_lr0.002')
    run.update(name=run_name,run_directory=str(RUNS/run_name),ctm_transformer=VARIANTS[variant])
    if len(parts)>2:run['seed']=int(parts[2][1:])
    run['train']['lr']=lr
    if variant in MICRO:run['train']['micro_batch_by_depth']=MICRO[variant]
    if variant in RECIPE_OVERRIDES:run['rdt_recipe']={**run['rdt_recipe'],**RECIPE_OVERRIDES[variant]}
    run['train']['snapshot_interval']=400  # a multiple of checkpoint_interval (50): snapshots are written only on checkpoint steps
    if pilot:run['train']['diagnostics']=True
    return run


def pilot_done(run_name):
    d=RUNS/run_name/'metrics.jsonl'
    return d.exists() and any(json.loads(r)['step']>=PILOT_STEPS for r in d.read_text().splitlines() if r.strip())


def final(run_name):
    if run_name.endswith('_pilot'):return 'done' if pilot_done(run_name) else None
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
        extra=['--max-steps',str(PILOT_STEPS)] if run_name.endswith('_pilot') else []
        code=subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(out/'config.json'),*extra],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    print('run',run_name,'exit',code,final(run_name),flush=True)


def diagnose(run_name,checkpoint=None,windows=8,depths=(1,4,8,16,32),device='cuda'):
    """Loss by tick count, and the loss increase when each mechanism is removed at evaluation."""
    import torch
    from scripts.pretrain import build_config
    from ctm_transformer.ctm_transformer import ctm_transformer_factory
    from ctm_transformer.pretrain import TokenWindows,evaluate_lm
    run=json.loads((RUNS/run_name/'config.json').read_text());t=run['train']
    data=json.loads(Path(run['data']).read_text());root=Path(run['data']).parent
    model=ctm_transformer_factory(set(run['ctm_transformer']),run['rdt_recipe'])(build_config(run,data['vocab_size']))
    name=checkpoint or (f'step_{t["total_steps"]:07d}.pt' if (RUNS/run_name/f'step_{t["total_steps"]:07d}.pt').exists() else 'latest.pt')
    path=Path(name) if Path(name).is_absolute() else RUNS/run_name/name
    payload=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'])
    device=torch.device(device);model.to(device)
    valid=TokenWindows([root/s for s in data['validation']],t['seq_len'],0);offset=t.get('eval_offset',0)
    indices=list(range(offset,offset+windows))
    with torch.no_grad():
        by_depth=evaluate_lm(model,valid,indices,1,device,'rdt',depths=tuple(depths))
        full=by_depth['depth_16'] if 'depth_16' in by_depth else evaluate_lm(model,valid,indices,1,device,'rdt',depths=(16,))['depth_16']
        removed={}
        params={'neuron_models':['nlm_scale'],'sync_query':[f'sync_queries.{i}.weight' for i in range(len(model.core))],'sync_output':['sync_projection.weight'],'attention_sink':['sink_gate']}
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


def pilots():
    """The pilot rule: mean training loss over steps 191-200 against CTsq's (the same seed and tick counts), and the
    largest gradient norm after step 100."""
    rows=lambda n:[json.loads(r) for r in (RUNS/n/'metrics.jsonl').read_text().splitlines() if r.strip()]
    window=lambda rs:sum(r['loss'] for r in rs if PILOT_STEPS-9<=r['step']<=PILOT_STEPS)/10
    reference=window(rows('CTsq_lr0.002'))
    L=['| Pilot | Train loss, steps 191-200 | vs CTsq | Max gradient norm after step 100 | Passes |','|---|---:|---:|---:|---|']
    for d in sorted(RUNS.glob('*_pilot')):
        if not pilot_done(d.name):L.append(f'| {d.name} | running | | | |');continue
        rs=rows(d.name);loss=window(rs);peak=max(r['grad_norm'] for r in rs if 100<r['step']<=PILOT_STEPS)
        L.append(f'| {d.name} | {loss:.3f} | {loss-reference:+.3f} | {peak:.3g} | {"yes" if loss-reference<=0.1 and peak<=10 else "no"} |')
    L.insert(0,f'CTsq (CTsq_lr0.002) train loss, steps 191-200: {reference:.3f}\n')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'PILOTS.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def summarize():
    rows=[(d.name,final(d.name)) for d in sorted(RUNS.iterdir()) if (d/'config.json').exists() and not d.name.endswith('_pilot')] if RUNS.exists() else []
    L=['# CTM Transformer on language modelling','','Protocol: `scripts/ctm_transformer_study.py` (docstring). Held-out loss at 400M tokens on 1,024 windows, 16 ticks.','',
       'References: Dans 3.569 / 3.579 (mean 3.574); 8-layer Transformer 3.618 / 3.638; 20-layer Transformer 3.562.','','| Run | Held-out loss |','|---|---:|']
    L+=[f'| {n} | {"running" if v is None else ("diverged or collapsed" if v==math.inf else f"{v:.3f}")} |' for n,v in rows]
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'CTM_TRANSFORMER.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int);p.add_argument('--runs',nargs='*')
    p.add_argument('--diagnose');p.add_argument('--checkpoint');p.add_argument('--device',default='cuda');p.add_argument('--windows',type=int,default=8)
    p.add_argument('--summarize',action='store_true');p.add_argument('--pilots',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    if a.pilots:pilots();return
    if a.diagnose:diagnose(a.diagnose,a.checkpoint,a.windows,device=a.device);return
    for run_name in a.runs or []:train(run_name,a.gpu)


if __name__=='__main__':main()
