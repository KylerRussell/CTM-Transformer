"""Summarize the S3 reliability study: escape probabilities, exact paired tests with Holm correction."""
import itertools,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta,binomtest
from ctm_transformer.experiment import file_hash
from scripts.summarize_group_s3 import correct_prefix

ROOT=Path('research/results/reliability_s3_v1')
CELL_ORDER=('transformer','rdt','ctm_lm')
ESCAPE=0.9  # a run escapes if mean accuracy over positions 9-16 at the trained budget is >= 0.9
COMPARISONS=(('rdt','transformer'),('ctm_lm','rdt'))


def read(p):return json.loads(Path(p).read_text())


def clopper_pearson(k,n,level=0.95):
    a=(1-level)/2
    return (0.0 if k==0 else float(beta.ppf(a,k,n-k+1)),1.0 if k==n else float(beta.ppf(1-a,k+1,n-k)))


def mcnemar_exact(x,y):
    """Two-sided exact McNemar test on paired binary outcomes (binomial test on discordant pairs)."""
    b=sum(1 for i,j in zip(x,y) if i and not j);c=sum(1 for i,j in zip(x,y) if j and not i)
    return {'first_only':b,'second_only':c,'p':1.0 if b+c==0 else float(binomtest(b,b+c,0.5).pvalue)}


def sign_flip_exact(diffs):
    """Two-sided exact paired permutation (sign-flip) test on the mean difference."""
    d=np.asarray(diffs,dtype=float);observed=abs(d.mean())
    signs=np.array(list(itertools.product((1.0,-1.0),repeat=len(d))))
    means=np.abs((signs*d).mean(1))
    return {'mean_difference':float(d.mean()),'p':float((means>=observed-1e-12).mean())}


def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);adjusted=[0.0]*len(pvalues);running=0.0
    for rank,i in enumerate(order):
        running=max(running,min(1.0,(len(pvalues)-rank)*pvalues[i]));adjusted[i]=running
    return adjusted


def main():
    registry=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');models={t['cell']:t for t in frozen['models']}
    ev={c:read(ROOT/f'{c}.evaluation.json') for c in models}
    for e in ev.values():assert e['complete'] and e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
    seeds=registry['seeds'];n=len(seeds);name=lambda cell,s:f'{cell}_seed{s}'
    trained=lambda cell:'1' if cell=='transformer' else '16'
    positions=lambda cell,s,T:ev[name(cell,s)]['results'][T]['by_position']
    mid={cell:[statistics.mean(positions(cell,s,trained(cell))[8:16]) for s in seeds] for cell in CELL_ORDER}
    area={cell:[statistics.mean(positions(cell,s,trained(cell))[:16]) for s in seeds] for cell in CELL_ORDER}
    escaped={cell:[m>=ESCAPE for m in mid[cell]] for cell in CELL_ORDER}
    prefix={cell:{T:[correct_prefix(positions(cell,s,T)) for s in seeds] for T in ev[name(cell,seeds[0])]['results']} for cell in CELL_ORDER}
    escape={cell:{'escaped':sum(escaped[cell]),'n':n,'probability':sum(escaped[cell])/n,'ci95':clopper_pearson(sum(escaped[cell]),n)} for cell in CELL_ORDER}
    tests=[]
    for a,b in COMPARISONS:
        tests.append({'comparison':f'{a}_vs_{b}','metric':'mean accuracy positions 1-16 (paired sign-flip)',**sign_flip_exact([x-y for x,y in zip(area[a],area[b])])})
        tests.append({'comparison':f'{a}_vs_{b}','metric':'escape (paired exact McNemar)',**mcnemar_exact(escaped[a],escaped[b])})
    for t,adj in zip(tests,holm([t['p'] for t in tests])):t['p_holm']=adj
    report={'complete':True,'escape_rule':f'mean accuracy over positions 9-16 at the trained budget >= {ESCAPE}','escape':escape,
        'mean_accuracy_positions_9_16':mid,'mean_accuracy_positions_1_16':area,'correct_prefix':prefix,'tests':tests,
        'parameters':{cell:models[name(cell,seeds[0])]['parameters']['total'] for cell in CELL_ORDER},
        'training_minutes':{cell:statistics.mean(models[name(cell,s)]['training_seconds'] for s in seeds)/60 for cell in CELL_ORDER},
        'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in ev}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

    pct=lambda x:f'{100*x:.1f}%'
    L=['# S3 reliability: escape probability over twenty seeds','',
       f'{len(models)} runs ({len(CELL_ORDER)} cells × {n} seeds), S₃ running products, training lengths 1–16, 10,000 updates, peak LR 0.0003. '
       f'Evaluation: 1,024 held-out length-32 words per seed, evaluated once after the freeze. A run **escapes** if its mean accuracy over positions 9–16 at the trained budget (Transformer T=1, others T=16) is ≥ {ESCAPE:.0%}.','',
       '## Primary: escape probability','','| Cell | Parameters | Escaped | Probability | 95% CI (Clopper–Pearson) | Mean accuracy, positions 1–16 | Median correct prefix | Train min |','|---|---:|---:|---:|---|---:|---:|---:|']
    for cell in CELL_ORDER:
        e=escape[cell]
        L.append(f"| {cell} | {report['parameters'][cell]:,} | {e['escaped']}/{n} | {pct(e['probability'])} | {pct(e['ci95'][0])}–{pct(e['ci95'][1])} | "
                 f"{pct(statistics.mean(area[cell]))} | {statistics.median(prefix[cell][trained(cell)]):g} | {report['training_minutes'][cell]:.0f} |")
    L+=['','## Declared paired tests (Holm-corrected over all four)','','| Comparison | Metric | Effect | p | p (Holm) |','|---|---|---|---:|---:|']
    for t in tests:
        effect=f"mean diff {100*t['mean_difference']:+.1f} points" if 'mean_difference' in t else f"{t['first_only']} vs {t['second_only']} discordant"
        L.append(f"| {t['comparison'].replace('_vs_',' vs ')} | {t['metric']} | {effect} | {t['p']:.4f} | {t['p_holm']:.4f} |")
    L+=['','## Tick use (median correct prefix at T=4/8/16/32)','']
    for cell in CELL_ORDER[1:]:L.append(f"- **{cell}:** "+'/'.join(f"{statistics.median(prefix[cell][T]):g}" for T in ('4','8','16','32')))
    L+=['','![Distribution of positions 9–16 accuracy](escape_distribution.png)','','See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and per-run evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')
    fig,ax=plt.subplots(figsize=(7,4))
    for i,cell in enumerate(CELL_ORDER):ax.scatter([i+np.random.default_rng(0).uniform(-.15,.15) for _ in mid[cell]],mid[cell],s=18)
    ax.axhline(ESCAPE,color='gray',ls='--',lw=1);ax.set_xticks(range(len(CELL_ORDER)),CELL_ORDER);ax.set_ylim(0,1.02)
    ax.set_ylabel('mean accuracy, positions 9–16 (per seed)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'escape_distribution.{ext}',dpi=180)
    print('\n'.join(L[:12]),flush=True)

if __name__=='__main__':main()
