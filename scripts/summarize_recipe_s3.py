"""Summarize the S3 recipe confirmation: escape, accuracy and extrapolation with RoPE; four declared paired tests, Holm-corrected."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ctm_transformer.experiment import file_hash
from scripts.summarize_group_s3 import correct_prefix
from scripts.summarize_reliability_s3 import ESCAPE,clopper_pearson,holm,mcnemar_exact

ROOT=Path('research/results/recipe_s3_v1')
CELLS=('transformer_rope','rdt_rope_fixed','rdt_rope_rand','ctm_lm_rope')
TRAINED={'transformer_rope':'1','rdt_rope_fixed':'16','rdt_rope_rand':'16','ctm_lm_rope':'16'}
EXTRAPOLATION={'transformer_rope':'1','rdt_rope_fixed':'32','rdt_rope_rand':'32','ctm_lm_rope':'32'}  # T >= evaluation length for recurrent cells
DRAWS,RNG_SEED=2_000_000,20260929


def read(p):return json.loads(Path(p).read_text())


def sign_flip_monte_carlo(diffs,draws=DRAWS,seed=RNG_SEED):
    """Two-sided paired sign-flip test on the mean difference; seeded Monte Carlo with the +1 correction."""
    d=np.asarray(diffs,dtype=float);observed=abs(d.mean());rng=np.random.default_rng(seed);hits=0;done=0
    while done<draws:
        n=min(100_000,draws-done);signs=rng.choice((-1.0,1.0),size=(n,len(d)))
        hits+=int((np.abs((signs*d).mean(1))>=observed-1e-12).sum());done+=n
    return {'mean_difference':float(d.mean()),'p':(hits+1)/(draws+1),'draws':draws,'rng_seed':seed}


def main():
    registry=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');models={t['cell']:t for t in frozen['models']}
    ev={c:read(ROOT/f'{c}.evaluation.json') for c in models}
    for e in ev.values():assert e['complete'] and e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
    seeds=registry['seeds'];n=len(seeds)
    positions=lambda cell,s,T:ev[f'{cell}_seed{s}']['results'][T]['by_position']
    mean=lambda cell,T,a,b:[statistics.mean(positions(cell,s,T)[a:b]) for s in seeds]
    area={c:mean(c,TRAINED[c],0,16) for c in CELLS};mid={c:mean(c,TRAINED[c],8,16) for c in CELLS}
    near={c:mean(c,EXTRAPOLATION[c],16,24) for c in CELLS};far={c:mean(c,EXTRAPOLATION[c],24,32) for c in CELLS}
    escape={c:[v>=ESCAPE for v in mid[c]] for c in CELLS}
    ticks={c:list(ev[f'{c}_seed{seeds[0]}']['results']) for c in CELLS}
    prefix={c:{T:[correct_prefix(positions(c,s,T)) for s in seeds] for T in ticks[c]} for c in CELLS}
    diff=lambda a,b:[x-y for x,y in zip(a,b)]
    tests=[{'id':'T1','comparison':'rdt_rope_rand_vs_rdt_rope_fixed','metric':'mean accuracy positions 1-16 at T=16 (paired sign-flip)',
            **sign_flip_monte_carlo(diff(area['rdt_rope_rand'],area['rdt_rope_fixed']))},
           {'id':'T2','comparison':'rdt_rope_rand_vs_rdt_rope_fixed','metric':'escape at T=16 (paired exact McNemar)',
            **mcnemar_exact(escape['rdt_rope_rand'],escape['rdt_rope_fixed'])},
           {'id':'T3','comparison':'rdt_rope_rand_vs_transformer_rope','metric':'mean accuracy positions 17-24 (RDT at T=32; paired sign-flip)',
            **sign_flip_monte_carlo(diff(near['rdt_rope_rand'],near['transformer_rope']))},
           {'id':'T4','comparison':'ctm_lm_rope_vs_rdt_rope_fixed','metric':'mean accuracy positions 1-16 at T=16 (paired sign-flip)',
            **sign_flip_monte_carlo(diff(area['ctm_lm_rope'],area['rdt_rope_fixed']))}]
    for t,adj in zip(tests,holm([t['p'] for t in tests])):t['p_holm']=adj
    secondary=[{'comparison':'rdt_rope_fixed_vs_transformer_rope','metric':'mean accuracy positions 1-16 at the trained budget (paired sign-flip; not corrected)',
                **sign_flip_monte_carlo(diff(area['rdt_rope_fixed'],area['transformer_rope']))},
               {'comparison':'rdt_rope_fixed_vs_transformer_rope','metric':'mean accuracy positions 17-24 (RDT at T=32; paired sign-flip; not corrected)',
                **sign_flip_monte_carlo(diff(near['rdt_rope_fixed'],near['transformer_rope']))}]
    report={'complete':True,'escape':{c:{'escaped':sum(escape[c]),'n':n,'ci95':clopper_pearson(sum(escape[c]),n)} for c in CELLS},
        'mean_accuracy_positions_1_16':area,'mean_accuracy_positions_9_16':mid,'mean_accuracy_positions_17_24':near,'mean_accuracy_positions_25_32':far,
        'correct_prefix':prefix,'tests':tests,'secondary_tests':secondary,
        'accuracy_positions_1_16_by_ticks':{c:{T:statistics.mean(mean(c,T,0,16)) for T in ticks[c]} for c in CELLS},
        'accuracy_positions_17_24_by_ticks':{c:{T:statistics.mean(mean(c,T,16,24)) for T in ticks[c]} for c in CELLS},
        'training_minutes':{c:statistics.mean(models[f'{c}_seed{s}']['training_seconds'] for s in seeds)/60 for c in CELLS},
        'parameters':{c:models[f'{c}_seed{seeds[0]}']['parameters'] for c in CELLS},
        'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in models}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

    pct=lambda x:f'{100*x:.1f}%'
    L=['# S3 recipe confirmation with RoPE','',
       f'Four cells × {n} new seeds, S₃ running products, training lengths 1–16, RoPE in every attention layer and no absolute position table. '
       f'Escape: mean accuracy over positions 9–16 ≥ {ESCAPE:.0%} at the trained budget (Transformer T=1, others T=16). Evaluation words have length 32; '
       'positions 17–32 measure length extrapolation (recurrent cells at T=32).','',
       '| Cell | Escaped | 95% CI | Positions 1–16 | Positions 17–24 | Positions 25–32 | Median prefix T=4/8/16/32/48/64 |','|---|---:|---|---:|---:|---:|---|']
    for c in CELLS:
        e=report['escape'][c];scaling='/'.join(f"{statistics.median(prefix[c][T]):g}" if T in prefix[c] else '—' for T in ('4','8','16','32','48','64'))
        if c=='transformer_rope':scaling=f"{statistics.median(prefix[c]['1']):g} (T=1)"
        L.append(f"| {c} | {e['escaped']}/{n} | {pct(e['ci95'][0])}–{pct(e['ci95'][1])} | {pct(statistics.mean(area[c]))} | {pct(statistics.mean(near[c]))} | {pct(statistics.mean(far[c]))} | {scaling} |")
    L+=['','## Declared paired tests (Holm over all four)','','| Test | Comparison | Metric | Effect | p | p (Holm) |','|---|---|---|---|---:|---:|']
    effect=lambda t:f"mean diff {100*t['mean_difference']:+.1f} points" if 'mean_difference' in t else f"{t['first_only']} vs {t['second_only']} discordant"
    for t in tests:L.append(f"| {t['id']} | {t['comparison'].replace('_vs_',' vs ')} | {t['metric']} | {effect(t)} | {t['p']:.4f} | {t['p_holm']:.4f} |")
    L+=['','## Secondary (not corrected)','','| Comparison | Metric | Effect | p |','|---|---|---|---:|']
    for t in secondary:L.append(f"| {t['comparison'].replace('_vs_',' vs ')} | {t['metric']} | {effect(t)} | {t['p']:.4f} |")
    L+=['','Sign-flip p-values use 2,000,000 seeded Monte Carlo sign assignments with the +1 correction; McNemar is exact.','',
        '![Accuracy by position](accuracy_by_position.png)','','See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and the per-run evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for c,style in zip(CELLS,('k:','b--','b-','r-')):
        for T,alpha in ((TRAINED[c],1.0),('64',0.4)):
            if T in ticks[c]:ax.plot(range(1,33),[statistics.mean(positions(c,s,T)[i] for s in seeds) for i in range(32)],style,alpha=alpha,label=f'{c} T={T}')
    ax.axvline(16.5,color='gray',ls=':',lw=1);ax.axhline(1/6,color='gray',lw=0.5);ax.set_ylim(0,1.02);ax.legend(fontsize=7)
    ax.set_xlabel('position');ax.set_ylabel('accuracy (mean over seeds)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'accuracy_by_position.{ext}',dpi=180)
    print('\n'.join(L[:12]),flush=True)

if __name__=='__main__':main()
