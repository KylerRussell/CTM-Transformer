"""Summarize the S3 learning-rate control: randomized vs fixed depth at the same learning rate; two declared paired tests, Holm-corrected."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctm_transformer.experiment import file_hash
from scripts.summarize_group_s3 import correct_prefix
from scripts.summarize_recipe_s3 import sign_flip_monte_carlo
from scripts.summarize_reliability_s3 import ESCAPE,clopper_pearson,holm,mcnemar_exact

ROOT=Path('research/results/lrcontrol_s3_v1')
NEW='rdt_rope_fixed_lr1e3';RAND='rdt_rope_rand';FIXED='rdt_rope_fixed'  # the last two come from the recipe confirmation
CELLS=(RAND,NEW,FIXED)
LABEL={RAND:'randomized depth, LR 1e-3 (recipe study)',NEW:'fixed depth, LR 1e-3 (this study)',FIXED:'fixed depth, LR 6e-4 (recipe study)'}


def read(p):return json.loads(Path(p).read_text())


def main():
    registry=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');models={t['cell']:t for t in frozen['models']}
    control=Path(registry['control_study'])
    assert file_hash(control/'registry.json')==registry['control_registry_sha256'] and file_hash(control/'summary.json')==registry['control_summary_sha256']
    recorded=read(control/'summary.json')['evaluations']
    ev={c:read(ROOT/f'{c}.evaluation.json') for c in models}
    for e in ev.values():assert e['complete'] and e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
    seeds=registry['seeds'];n=len(seeds)
    for cell in (RAND,FIXED):
        for s in seeds:
            rec=recorded[f'{cell}_seed{s}'];assert file_hash(rec['path'])==rec['sha256'];ev[f'{cell}_seed{s}']=read(rec['path'])
    positions=lambda cell,s,T:ev[f'{cell}_seed{s}']['results'][T]['by_position']
    mean=lambda cell,T,a,b:[statistics.mean(positions(cell,s,T)[a:b]) for s in seeds]
    area={c:mean(c,'16',0,16) for c in CELLS};mid={c:mean(c,'16',8,16) for c in CELLS};near={c:mean(c,'32',16,24) for c in CELLS}
    escape={c:[v>=ESCAPE for v in mid[c]] for c in CELLS}
    ticks=('4','8','16','32','48','64')
    prefix={c:{T:[correct_prefix(positions(c,s,T)) for s in seeds] for T in ticks} for c in CELLS}
    diff=lambda a,b:[x-y for x,y in zip(a,b)]
    tests=[{'id':'C1','comparison':f'{RAND}_vs_{NEW}','metric':'mean accuracy positions 1-16 at T=16 (paired sign-flip)',**sign_flip_monte_carlo(diff(area[RAND],area[NEW]))},
           {'id':'C2','comparison':f'{RAND}_vs_{NEW}','metric':'escape at T=16 (paired exact McNemar)',**mcnemar_exact(escape[RAND],escape[NEW])}]
    for t,adj in zip(tests,holm([t['p'] for t in tests])):t['p_holm']=adj
    secondary=[{'comparison':f'{NEW}_vs_{FIXED}','metric':'mean accuracy positions 1-16 at T=16 (paired sign-flip; not corrected)',**sign_flip_monte_carlo(diff(area[NEW],area[FIXED]))},
               {'comparison':f'{NEW}_vs_{FIXED}','metric':'escape at T=16 (paired exact McNemar; not corrected)',**mcnemar_exact(escape[NEW],escape[FIXED])},
               {'comparison':f'{RAND}_vs_{NEW}','metric':'mean accuracy positions 17-24 at T=32 (paired sign-flip; not corrected)',**sign_flip_monte_carlo(diff(near[RAND],near[NEW]))}]
    report={'complete':True,'escape':{c:{'escaped':sum(escape[c]),'n':n,'ci95':clopper_pearson(sum(escape[c]),n)} for c in CELLS},
        'mean_accuracy_positions_1_16':area,'mean_accuracy_positions_9_16':mid,'mean_accuracy_positions_17_24':near,'correct_prefix':prefix,
        'tests':tests,'secondary_tests':secondary,
        'accuracy_positions_1_16_by_ticks':{c:{T:statistics.mean(mean(c,T,0,16)) for T in ticks} for c in CELLS},
        'training_minutes':statistics.mean(models[f'{NEW}_seed{s}']['training_seconds'] for s in seeds)/60,
        'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in models}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    pct=lambda x:f'{100*x:.1f}%'
    effect=lambda t:f"mean diff {100*t['mean_difference']:+.1f} points" if 'mean_difference' in t else f"{t['first_only']} vs {t['second_only']} discordant"
    L=['# S3 learning-rate control: fixed depth at the randomized-depth learning rate','',
       f'Fixed-depth RoPE RDT trained at LR 1e-3 on the recipe confirmation\'s {n} seeds and data, paired with that study\'s randomized-depth (LR 1e-3) and fixed-depth (LR 6e-4) cells. '
       f'Escape: mean accuracy over positions 9–16 ≥ {ESCAPE:.0%} at T=16.','',
       '| Cell | Escaped | 95% CI | Positions 1–16 (T=16) | Positions 17–24 (T=32) | Median prefix T=4/8/16/32/48/64 |','|---|---:|---|---:|---:|---|']
    for c in CELLS:
        e=report['escape'][c]
        L.append(f"| {LABEL[c]} | {e['escaped']}/{n} | {pct(e['ci95'][0])}–{pct(e['ci95'][1])} | {pct(statistics.mean(area[c]))} | {pct(statistics.mean(near[c]))} | "
                 +'/'.join(f"{statistics.median(prefix[c][T]):g}" for T in ticks)+' |')
    L+=['','## Declared paired tests (Holm over both)','','| Test | Comparison | Metric | Effect | p | p (Holm) |','|---|---|---|---|---:|---:|']
    for t in tests:L.append(f"| {t['id']} | {t['comparison'].replace('_vs_',' vs ')} | {t['metric']} | {effect(t)} | {t['p']:.2e} | {t['p_holm']:.2e} |")
    L+=['','## Secondary (not corrected)','','| Comparison | Metric | Effect | p |','|---|---|---|---:|']
    for t in secondary:L.append(f"| {t['comparison'].replace('_vs_',' vs ')} | {t['metric']} | {effect(t)} | {t['p']:.2e} |")
    L+=['','Sign-flip p-values use 2,000,000 seeded Monte Carlo sign assignments with the +1 correction (floor 5.0e-07); McNemar is exact.','',
        '![Accuracy by position](accuracy_by_position.png)','','See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and the per-run evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for c,style in zip(CELLS,('b-','g-','b--')):
        ax.plot(range(1,33),[statistics.mean(positions(c,s,'32')[i] for s in seeds) for i in range(32)],style,label=f'{LABEL[c]}, T=32')
    ax.axvline(16.5,color='gray',ls=':',lw=1);ax.axhline(1/6,color='gray',lw=0.5);ax.set_ylim(0,1.02);ax.legend(fontsize=7)
    ax.set_xlabel('position');ax.set_ylabel('accuracy (mean over seeds)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'accuracy_by_position.{ext}',dpi=180)
    print('\n'.join(L[:9]),flush=True)

if __name__=='__main__':main()
