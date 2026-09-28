"""Summarize randomized-depth training against the paired fixed-depth reliability controls."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctm_transformer.experiment import file_hash
from scripts.summarize_group_s3 import correct_prefix
from scripts.summarize_reliability_s3 import ESCAPE,clopper_pearson,holm,mcnemar_exact,sign_flip_exact

ROOT=Path('research/results/randdepth_s3_v1')
PAIRS=(('rdt_rand','rdt'),('ctm_lm_rand','ctm_lm'))


def read(p):return json.loads(Path(p).read_text())


def main():
    registry=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');models={t['cell']:t for t in frozen['models']}
    control_root=Path(registry['control_study']);assert file_hash(control_root/'registry.json')==registry['control_registry_sha256']
    control=read(control_root/'summary.json')
    ev={c:read(ROOT/f'{c}.evaluation.json') for c in models}
    for e in ev.values():assert e['complete'] and e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
    for pair in PAIRS:
        for s in registry['seeds']:
            name=f'{pair[1]}_seed{s}';rec=control['evaluations'][name];assert file_hash(rec['path'])==rec['sha256'];ev[name]=read(rec['path'])
    seeds=registry['seeds'];n=len(seeds);cells=[c for p in PAIRS for c in p]
    positions=lambda cell,s,T:ev[f'{cell}_seed{s}']['results'][T]['by_position']
    mean=lambda cell,T,a,b:[statistics.mean(positions(cell,s,T)[a:b]) for s in seeds]
    mid16={c:mean(c,'16',8,16) for c in cells};area16={c:mean(c,'16',0,16) for c in cells}
    mid32={c:mean(c,'32',8,16) for c in cells};extra32={c:mean(c,'32',16,32) for c in cells}
    ticks={c:[T for T in ev[f'{c}_seed{seeds[0]}']['results']] for c in cells}
    prefix={c:{T:[correct_prefix(positions(c,s,T)) for s in seeds] for T in ticks[c]} for c in cells}
    escape=lambda values:[v>=ESCAPE for v in values]
    summary_escape={c:{T:{'escaped':sum(escape(v)),'n':n,'ci95':clopper_pearson(sum(escape(v)),n)} for T,v in (('16',mid16[c]),('32',mid32[c]))} for c in cells}
    tests=[]
    for a,b in PAIRS:
        tests.append({'comparison':f'{a}_vs_{b}','metric':'escape at T=16 (paired exact McNemar)',**mcnemar_exact(escape(mid16[a]),escape(mid16[b]))})
        tests.append({'comparison':f'{a}_vs_{b}','metric':'mean accuracy positions 1-16 at T=16 (paired sign-flip)',**sign_flip_exact([x-y for x,y in zip(area16[a],area16[b])])})
        tests.append({'comparison':f'{a}_vs_{b}','metric':'mean accuracy positions 17-32 at T=32 (paired sign-flip)',**sign_flip_exact([x-y for x,y in zip(extra32[a],extra32[b])])})
    for t,adj in zip(tests,holm([t['p'] for t in tests])):t['p_holm']=adj
    depth_counts={c:models[f'{c}_seed{seeds[0]}'].get('depth_counts') for c in ('rdt_rand','ctm_lm_rand')}
    report={'complete':True,'escape':summary_escape,'mean_accuracy_positions_9_16_T16':mid16,'mean_accuracy_positions_1_16_T16':area16,
        'mean_accuracy_positions_9_16_T32':mid32,'mean_accuracy_positions_17_32_T32':extra32,'correct_prefix':prefix,'tests':tests,
        'depth_sampler':registry['depth_sampler'],'training_minutes':{c:statistics.mean(models[f'{c}_seed{s}']['training_seconds'] for s in seeds)/60 for c in ('rdt_rand','ctm_lm_rand')},
        'control_summary_sha256':file_hash(control_root/'summary.json'),'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),
        'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in models}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

    pct=lambda x:f'{100*x:.1f}%'
    L=['# S3 randomized-depth training versus fixed depth','',
       f'RDT and CTM-LM trained with {registry["depth_sampler"]} (expected depth about 16), paired by seed and data with the fixed-T=16 runs of the reliability study ({n} seeds). '
       f'Escape: mean accuracy over positions 9–16 ≥ {ESCAPE:.0%}. Evaluation words have length 32.','',
       '| Cell | Escape at T=16 | 95% CI | Escape at T=32 | Positions 1–16 at T=16 | Positions 17–32 at T=32 | Median prefix T=4/8/16/32/48/64 |','|---|---:|---|---:|---:|---:|---|']
    for c in cells:
        e16,e32=summary_escape[c]['16'],summary_escape[c]['32']
        scaling='/'.join(f"{statistics.median(prefix[c][T]):g}" if T in prefix[c] else '—' for T in ('4','8','16','32','48','64'))
        L.append(f"| {c} | {e16['escaped']}/{n} | {pct(e16['ci95'][0])}–{pct(e16['ci95'][1])} | {e32['escaped']}/{n} | {pct(statistics.mean(area16[c]))} | {pct(statistics.mean(extra32[c]))} | {scaling} |")
    L+=['','## Declared paired tests (Holm over all six)','','| Comparison | Metric | Effect | p | p (Holm) |','|---|---|---|---:|---:|']
    for t in tests:
        effect=f"mean diff {100*t['mean_difference']:+.1f} points" if 'mean_difference' in t else f"{t['first_only']} vs {t['second_only']} discordant"
        L.append(f"| {t['comparison'].replace('_vs_',' vs ')} | {t['metric']} | {effect} | {t['p']:.4f} | {t['p_holm']:.4f} |")
    L+=['','![Accuracy by position](accuracy_by_position.png)','','See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and the per-run evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,(a,b) in zip(axes,PAIRS):
        for c,style in ((a,'-'),(b,'--')):
            for T in ('16','32'):
                if T in ticks[c]:ax.plot(range(1,33),[statistics.mean(positions(c,s,T)[i] for s in seeds) for i in range(32)],style,label=f'{c} T={T}')
        ax.axvline(16.5,color='gray',ls=':',lw=1);ax.set_ylim(0,1.02);ax.legend(fontsize=7);ax.set_xlabel('position')
    axes[0].set_ylabel('accuracy (mean over seeds)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'accuracy_by_position.{ext}',dpi=180)
    print('\n'.join(L[:10]),flush=True)

if __name__=='__main__':main()
