"""Do the adapted CTM-LM's ticks contribute on a task that needs serial computation? (research/CTM_TICK_DIAGNOSTICS.md, 2026-10-07)

At LM scale, a CTM trained with one tick matched the 16-tick model, and the same was true
of the RDT. This study asks the question on the S3 word problem (running products over S3),
where each label depends on every earlier token and a fixed shallow network is known to
fail past about position 5 (research/RELIABILITY_S3.md).

Setup: the reliability study's recipe and data, on its first 10 seeds (131-181), so its
Transformer, RDT and faithful CTM-LM runs are paired references (same seed, same data):
LR 3e-4, 10,000 updates of 32 words of lengths 1-16, warmup 100, cosine to 10%, BF16
autocast, runner v3 (`train_dense_experiment`). The model is the reliability study's tiny
CTM-LM (d 128, 2 backbone layers, D 128, 128 pairs) as the adapted CTM-LM: RoPE in place of
the learned position table, token-conditioned start state, sparse-tick loss (ticks 4/8/12/16,
or the only tick when T = 1).

| Cell | Change |
|---|---|
| adapt | the adapted CTM-LM, T = 16 |
| adapt_t1 | the same, trained and evaluated with T = 1 |
| adapt_cross | adapt with cross-position ticks (keys and values also read every causal position's current state) |
| adapt_cross_t1 | adapt_cross with T = 1 |
| adapt_wide | adapt with the synapse U-Net width doubled (256 to 512) and the neuron-level models' width doubled (16 to 32) |
| adapt_wide_t1 | adapt_wide with T = 1 |

Like-for-like references (added 2026-10-07): the reliability study's Transformer (2 layers)
and RDT (1/2/1, T = 16, confidence readout) differ from these cells in position encoding.
`transformer_rope`, `rdt_rope` and `rdt_rope_t1` retrain them with RoPE and no position
table (`position_factory`), otherwise with their reliability configs; `rdt_rope_t1` trains and
evaluates the RDT at depth 1. Rule, fixed before these runs: adapt_cross exceeds rdt_rope if
it is higher on at least 8 of 10 seeds with a median gain of at least 5 points.

From Continuous Memory Machines (Regan et al., arXiv 2610.07907; added 2026-10-07):
`cross_norm_sink` is adapt_cross with mean-normalized synchronization (alpha / beta) and a
zero-value sink key in the tick attention; `cross_memory` is adapt_cross with CMM's joint
memory update over [sink; 8 long-term slots; the 8-tick pre-activation history]. Each has a
T = 1 control. Rules, fixed before these runs: a cell improves on adapt_cross if it is higher
on at least 8 of 10 seeds with a median gain of at least 3 points; the ticks rule above applies
to each against its own T = 1 control. Accuracy at T = 32 against T = 16 is reported, since the
normalization is claimed to help beyond the trained tick count.

Drift fixes (deep-research review, added 2026-10-08): `cross_anchor` is adapt_cross with the
backbone feature re-injected every tick (observe_token) and a gated state update
(gated_state); `cross_clamp` is adapt_cross with the official code's in-place decay clamp
(decay_data_clamp). Same improvement rule as above against adapt_cross; the drift test is
accuracy at T = 32 against T = 16.

RDT + CTM hybrids (added 2026-10-11, after the RDT recipe study, research/RDT_RECIPE.md):
`dans` is the reliability RDT (1/2/1, d 96, T = 16, confidence readout) with the adopted recipe
(pre-norm, additive injection, no coda norm, a norm on the recurrent state); `dans_t1` is its
T = 1 control. `dans_sync`, `dans_nlm` and `dans_carry` add one CTM mechanism each
(ctm_transformer/hybrid_rdt.py: synchronization readout, neuron-level models, state carry),
zero-initialized so that each starts as dans; `dans_all` adds all three. Rules, fixed before
these runs (same thresholds as above): dans improves on rdt_rope, and a hybrid improves on
dans, if higher on at least 8 of 10 seeds with a median gain of at least 3 points; dans's ticks
contribute under the ticks rule against dans_t1. Comparisons with cross_anchor are exploratory.

Evaluation: the final checkpoint (step 10,000) on each seed's 1,024 held-out words of length
32, final-tick readout, at T = 1, 4, 8, 16, 32 (T = 1 cells at T = 1 only).

Primary endpoint: mean accuracy over positions 1-16 at the trained T. Rule fixed before the
runs: a cell's ticks contribute if it beats its T = 1 control on at least 8 of 10 seeds with a
median paired gain of at least 5 accuracy points.

Usage: python -m scripts.tick_s3 --device cuda:N --cells adapt adapt_t1 ... [--seeds 131 137 ...]
       python -m scripts.tick_s3 --summarize
"""
import argparse,gc,json,statistics,traceback
from dataclasses import replace
from pathlib import Path

import torch

RUNS=Path('research/runs/tick_s3');OUT=Path('research/results/tick_s3')
RELIABILITY=Path('research/results/reliability_s3_v1')
SEEDS=(131,137,139,149,151,157,163,167,173,179)
BASE='research/configs/reliability_s3_v1/ctm_lm_seed{seed}.json'
LOSS=['token_start','sparse_tick_loss']
CELLS={'adapt':{'adaptations':LOSS,'T':16},'adapt_t1':{'adaptations':LOSS,'T':1},
       'adapt_cross':{'adaptations':LOSS+['cross_position'],'T':16},'adapt_cross_t1':{'adaptations':LOSS+['cross_position'],'T':1},
       'adapt_wide':{'adaptations':LOSS,'T':16,'unet_width':512,'nlm_hidden_dim':32},
       'adapt_wide_t1':{'adaptations':LOSS,'T':1,'unet_width':512,'nlm_hidden_dim':32},
       'cross_norm_sink':{'adaptations':LOSS+['cross_position','sync_mean','attention_sink'],'T':16},
       'cross_norm_sink_t1':{'adaptations':LOSS+['cross_position','sync_mean','attention_sink'],'T':1},
       'cross_memory':{'adaptations':LOSS+['cross_position','tick_memory'],'T':16},
       'cross_memory_t1':{'adaptations':LOSS+['cross_position','tick_memory'],'T':1},
       'cross_anchor':{'adaptations':LOSS+['cross_position','observe_token','gated_state'],'T':16},
       'cross_anchor_t1':{'adaptations':LOSS+['cross_position','observe_token','gated_state'],'T':1},
       'cross_clamp':{'adaptations':LOSS+['cross_position','decay_data_clamp'],'T':16},
       'transformer_rope':{'reference':'transformer','T':1,'readout':'final'},
       'rdt_rope':{'reference':'rdt','T':16,'readout':'confidence'},
       'rdt_rope_t1':{'reference':'rdt','T':1,'readout':'confidence'},
       'dans':{'hybrid':[],'T':16,'readout':'confidence'},'dans_t1':{'hybrid':[],'T':1,'readout':'confidence'},
       'dans_sync':{'hybrid':['sync_readout'],'T':16,'readout':'confidence'},'dans_nlm':{'hybrid':['neuron_models'],'T':16,'readout':'confidence'},
       'dans_carry':{'hybrid':['state_carry'],'T':16,'readout':'confidence'},
       'dans_all':{'hybrid':['sync_readout','neuron_models','state_carry'],'T':16,'readout':'confidence'}}
EVAL_TICKS=(1,4,8,16,32)


def factory(cell):
    spec=CELLS[cell]
    if 'hybrid' in spec:
        from ctm_transformer.hybrid_rdt import hybrid_factory
        return hybrid_factory(set(spec['hybrid']))
    if 'reference' in spec:
        from ctm_transformer.positions import position_factory
        return position_factory(spec['reference'],'rope')
    from ctm_transformer.ctm_lm_adapt import adapted_factory
    return adapted_factory(set(spec['adaptations']),unet_width=spec.get('unet_width'))


def cell_config(cell,seed,device):
    from ctm_transformer.research import load_research_config
    spec=CELLS[cell];ref=spec.get('reference','rdt' if 'hybrid' in spec else None)
    base=BASE if ref is None else f'research/configs/reliability_s3_v1/{ref}_seed{{seed}}.json'
    c,identity=load_research_config(base.format(seed=seed))
    c=replace(c,use_positional_encoding=False,max_thought_steps=spec['T'],device=device,checkpoint_dir=str((RUNS/cell/f'seed{seed}').resolve()))
    if ref is not None:
        if ref=='rdt':c=replace(c,train_depth_min=spec['T'],train_depth_max=spec['T'])
    else:c=replace(c,nlm_hidden_dim=spec.get('nlm_hidden_dim',c.nlm_hidden_dim))
    return c,identity


def run(cell,seed,device,smoke=False):
    from ctm_transformer.dense_experiment import train_dense_experiment
    from ctm_transformer.dense_pointer import evaluate_dense
    from ctm_transformer.group_suite import load_group_split
    out=OUT/f'{cell}_seed{seed}.json'
    if out.exists():return
    registry=json.loads((RELIABILITY/'registry.json').read_text());dataset=registry['dataset']
    c,identity=cell_config(cell,seed,device)
    train=load_group_split(dataset,seed,'train',c.seq_len);valid=load_group_split(dataset,seed,'validation',c.seq_len)
    c=replace(c,data_path=train.metadata['path'],eval_data_path=valid.metadata['path'])
    if smoke:c=replace(c,max_steps=20,eval_interval=10,warmup_steps=5)
    readout=CELLS[cell].get('readout','final')
    result=train_dense_experiment(c,identity,seed,train,valid,evaluate_dense,[readout],save_validation_checkpoints=False,model_factory=factory(cell),
        data_policy='S3 running products; reliability_s3_v1 data; adapted CTM-LM tick study')
    model=factory(cell)(c);payload=torch.load(Path(c.checkpoint_dir)/'final.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model_state_dict']);model=model.to(device).eval()
    data=load_group_split(dataset,seed,'evaluation',c.seq_len);results={}
    for ticks in (EVAL_TICKS if CELLS[cell]['T']>1 else (1,)):
        m=evaluate_dense(model,data,replace(c,max_thought_steps=ticks),device,(readout,))[readout]
        results[str(ticks)]={'answer_accuracy':m['answer_accuracy'],'loss':m['loss'],'by_position':m['answer_accuracy_by_position']}
    OUT.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'cell':cell,'seed':seed,'spec':CELLS[cell],'parameters':result.get('parameters'),'training_seconds':result.get('training_seconds'),
                               'results':results},indent=1)+'\n')
    print(cell,seed,{k:round(v['answer_accuracy'],4) for k,v in results.items()},flush=True)
    del model,train;gc.collect();torch.cuda.empty_cache()


def first16(by_position):return sum(by_position[:16])/16


def prefix(by_position,threshold=0.9):
    n=0
    for a in by_position:
        if a<threshold:break
        n+=1
    return n


def reference(cell,seed):
    """The reliability study's evaluation of a reference cell at its trained budget."""
    e=json.loads((RELIABILITY/f'{cell}_seed{seed}.evaluation.json').read_text());T='1' if cell=='transformer' else '16'
    return e['results'][T]['by_position']


def summarize():
    rows={};lines=['# Adapted CTM-LM tick use on the S3 word problem','','Protocol: `scripts/tick_s3.py` (docstring). Mean held-out accuracy over positions 1–16 of length-32 words, final readout, final checkpoint.','']
    for cell,spec in CELLS.items():
        for seed in SEEDS:
            p=OUT/f'{cell}_seed{seed}.json'
            if p.exists():rows.setdefault(cell,{})[seed]=json.loads(p.read_text())
    lines+=['| Cell | Seeds | T trained | Mean accuracy 1–16 | Median correct prefix | Escapes (9–16 ≥ 0.9) | Accuracy 1–16 at T = 1 / 4 / 8 / 16 / 32 |','|---|---:|---:|---:|---:|---:|---|']
    for ref in ('transformer','rdt','ctm_lm'):  # absolute positions
        accs=[reference(ref,s) for s in SEEDS]
        lines.append(f'| {ref} (reliability study, paired) | {len(accs)} | {1 if ref=="transformer" else 16} | {statistics.mean(first16(a) for a in accs):.3f} | {statistics.median(prefix(a) for a in accs)} | {sum(sum(a[8:16])/8>=0.9 for a in accs)} | |')
    for cell,by_seed in rows.items():
        T=str(CELLS[cell]['T']);accs=[r['results'][T]['by_position'] for r in by_seed.values()]
        curve=' / '.join(f'{statistics.mean(first16(r["results"][t]["by_position"]) for r in by_seed.values()):.3f}' if t in next(iter(by_seed.values()))['results'] else '' for t in map(str,EVAL_TICKS))
        lines.append(f'| {cell} | {len(accs)} | {T} | {statistics.mean(first16(a) for a in accs):.3f} | {statistics.median(prefix(a) for a in accs)} | {sum(sum(a[8:16])/8>=0.9 for a in accs)} | {curve} |')
    lines+=['','**Paired against the T = 1 control (rule: higher on ≥ 8 of 10 seeds and median gain ≥ 5 points):**','']
    if 'adapt_cross' in rows and 'rdt_rope' in rows:
        seeds=sorted(set(rows['adapt_cross'])&set(rows['rdt_rope']))
        diffs=[first16(rows['adapt_cross'][s]['results']['16']['by_position'])-first16(rows['rdt_rope'][s]['results']['16']['by_position']) for s in seeds]
        verdict='exceeds' if len(seeds)==10 and sum(d>0 for d in diffs)>=8 and statistics.median(diffs)>=0.05 else ('incomplete' if len(seeds)<10 else 'does not exceed')
        lines.append(f'- **adapt_cross vs rdt_rope (same position encoding):** higher on {sum(d>0 for d in diffs)}/{len(seeds)} seeds, median gain {100*statistics.median(diffs):+.1f} points: {verdict}.')
    for cell in ('cross_norm_sink','cross_memory','cross_anchor','cross_clamp'):
        if cell not in rows or 'adapt_cross' not in rows:continue
        seeds=sorted(set(rows[cell])&set(rows['adapt_cross']))
        diffs=[first16(rows[cell][s]['results']['16']['by_position'])-first16(rows['adapt_cross'][s]['results']['16']['by_position']) for s in seeds]
        verdict='improves' if len(seeds)==10 and sum(d>0 for d in diffs)>=8 and statistics.median(diffs)>=0.03 else ('incomplete' if len(seeds)<10 else 'does not improve')
        lines.append(f'- **{cell} vs adapt_cross:** higher on {sum(d>0 for d in diffs)}/{len(seeds)} seeds, median gain {100*statistics.median(diffs):+.1f} points: {verdict}.')
    def paired(a,b,ta='16',tb='16'):
        seeds=sorted(set(rows.get(a,{}))&set(rows.get(b,{})))
        return seeds,[first16(rows[a][s]['results'][ta]['by_position'])-first16(rows[b][s]['results'][tb]['by_position']) for s in seeds]
    for a,b in (('dans','rdt_rope'),('dans_sync','dans'),('dans_nlm','dans'),('dans_carry','dans'),('dans_all','dans')):
        seeds,diffs=paired(a,b)
        if not diffs:continue
        verdict='improves' if len(seeds)==10 and sum(d>0 for d in diffs)>=8 and statistics.median(diffs)>=0.03 else ('incomplete' if len(seeds)<10 else 'does not improve')
        lines.append(f'- **{a} vs {b}:** higher on {sum(d>0 for d in diffs)}/{len(seeds)} seeds, median gain {100*statistics.median(diffs):+.1f} points: {verdict}.')
    for a in ('dans','dans_sync','dans_nlm','dans_carry','dans_all'):
        seeds,diffs=paired(a,'cross_anchor')
        if diffs:lines.append(f'- *Exploratory:* **{a} vs cross_anchor:** higher on {sum(d>0 for d in diffs)}/{len(seeds)} seeds, median gain {100*statistics.median(diffs):+.1f} points.')
    for cell in ('adapt','adapt_cross','adapt_wide','rdt_rope','cross_norm_sink','cross_memory','cross_anchor','dans'):
        control=cell+'_t1'
        if cell not in rows or control not in rows:continue
        seeds=sorted(set(rows[cell])&set(rows[control]))
        diffs=[first16(rows[cell][s]['results']['16']['by_position'])-first16(rows[control][s]['results']['1']['by_position']) for s in seeds]
        if not diffs:continue
        verdict='ticks contribute' if len(seeds)==10 and sum(d>0 for d in diffs)>=8 and statistics.median(diffs)>=0.05 else ('incomplete' if len(seeds)<10 else 'ticks do not meet the rule')
        lines.append(f'- **{cell} vs {control}:** higher on {sum(d>0 for d in diffs)}/{len(seeds)} seeds, median gain {100*statistics.median(diffs):+.1f} points: {verdict}.')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'TICK_S3.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device');p.add_argument('--cells',nargs='*',default=[]);p.add_argument('--seeds',nargs='*',type=int,default=list(SEEDS))
    p.add_argument('--summarize',action='store_true');p.add_argument('--smoke',help='20-update check writing under this directory')
    a=p.parse_args()
    if a.smoke:
        global RUNS,OUT
        RUNS=Path(a.smoke)/'runs';OUT=Path(a.smoke)/'results'
    if a.summarize:summarize();return
    for cell in a.cells:
        for seed in a.seeds:
            try:run(cell,seed,a.device,smoke=bool(a.smoke))
            except Exception as exc:
                OUT.mkdir(parents=True,exist_ok=True)
                (OUT/f'{cell}_seed{seed}.failure.json').write_text(json.dumps({'error':repr(exc),'traceback':traceback.format_exc()},indent=1)+'\n');raise


if __name__=='__main__':main()
