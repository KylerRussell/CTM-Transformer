"""Summarize the Sync-RDT retrieval study and apply its declared decision rules."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/syncrdt_retrieval_v1')
CELL_ORDER=('rdt','rdt_wide','history','history_lowgate','sync','sync_rdt')
# A comparison "favors" the first cell if it reaches the threshold earlier in at least
# MIN_SEEDS of 5 seeds and the median paired difference is at least MIN_UPDATES.
MIN_SEEDS,MIN_UPDATES=4,1000


def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.2f}%'


def steps_to(curve,threshold,key='position1'):
    """First validation update reaching the threshold; None if never reached (censored)."""
    return next((c['step'] for c in curve if c[key]>=threshold),None)


def censored_value(step,budget,interval):
    # A cell that never reaches the threshold is ranked after every cell that does.
    return budget+interval if step is None else step


def compare(first,second,budget,interval):
    """Paired comparison over seeds: positive difference means `first` is earlier."""
    diffs=[censored_value(b,budget,interval)-censored_value(a,budget,interval) for a,b in zip(first,second)]
    earlier=sum(d>0 for d in diffs);later=sum(d<0 for d in diffs)
    median=statistics.median(diffs)
    return {'differences':diffs,'first_earlier_seeds':earlier,'first_later_seeds':later,'median_difference':median,
            'favors_first':earlier>=MIN_SEEDS and median>=MIN_UPDATES,'favors_second':later>=MIN_SEEDS and -median>=MIN_UPDATES}


def area(curve,key='position1'):return statistics.mean(c[key] for c in curve)


def main():
    registry=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');models={t['cell']:t for t in frozen['models']}
    ev={c:read(ROOT/f'{c}.evaluation.json') for c in models}
    for e in ev.values():assert e['complete'] and e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
    seeds=registry['seeds'];budget,interval=registry['primary_checkpoint_step'],registry['eval_interval']
    name=lambda cell,seed:f'{cell}_seed{seed}'
    reach={t:{cell:[steps_to(models[name(cell,s)]['validation_curve'],t) for s in seeds] for cell in CELL_ORDER} for t in registry['thresholds']}
    primary=lambda cell,s:ev[name(cell,s)]['results']['confidence']
    final={cell:{'position1':[primary(cell,s)['answer_accuracy_by_position'][0] for s in seeds],
                 'answer_accuracy':[primary(cell,s)['answer_accuracy'] for s in seeds],
                 'sequence_exact':[primary(cell,s)['sequence_exact'] for s in seeds]} for cell in CELL_ORDER}
    auc={cell:[area(models[name(cell,s)]['validation_curve']) for s in seeds] for cell in CELL_ORDER}
    comparisons={f'{a}_vs_{b}':{str(t):compare(reach[t][a],reach[t][b],budget,interval) for t in registry['thresholds']}
        for a,b in (('sync','rdt'),('sync','rdt_wide'),('history','rdt'),('history_lowgate','history'),('history_lowgate','rdt'),
                    ('sync_rdt','rdt'),('sync_rdt','sync'),('rdt_wide','rdt'))}
    c90=lambda key:comparisons[key]['0.9']
    decisions={
        'sync_speeds_retrieval':c90('sync_vs_rdt')['favors_first'] and c90('sync_vs_rdt_wide')['favors_first'],
        'history_slows_retrieval':c90('history_vs_rdt')['favors_second'],
        'history_speeds_retrieval':c90('history_vs_rdt')['favors_first'],
        'lowgate_explains_history_slowdown':c90('history_vs_rdt')['favors_second'] and c90('history_lowgate_vs_history')['favors_first'],
    }
    report={'complete':True,'no_test_evaluation':True,'thresholds':registry['thresholds'],'budget':budget,'eval_interval':interval,
        'decision_rule':f'favors first: earlier at the 90% threshold in >= {MIN_SEEDS} of {len(seeds)} seeds and median paired difference >= {MIN_UPDATES} updates; censored runs rank after all that reach it',
        'steps_to_threshold':{str(t):v for t,v in reach.items()},'final_evaluation':final,'validation_auc_position1':auc,
        'comparisons':comparisons,'decisions':decisions,
        'parameters':{cell:models[name(cell,seeds[0])]['parameters']['total'] for cell in CELL_ORDER},
        'training_minutes':{cell:statistics.mean(models[name(cell,s)]['training_seconds'] for s in seeds)/60 for cell in CELL_ORDER},
        'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in ev}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

    fmt_steps=lambda xs:', '.join('>10k' if x is None else f'{x/1000:g}k' for x in xs)
    med=lambda xs:statistics.median(censored_value(x,budget,interval) for x in xs)
    L=['# Sync-RDT retrieval sample efficiency','',
       f'{len(models)} runs ({len(CELL_ORDER)} cells × {len(seeds)} seeds {seeds}), {budget:,} updates each, peak LR {registry["learning_rate"]}, identical data per seed across cells. '
       'All-keys one-hop MQAR; confidence readout. Position-1 accuracy is leak-free (chance 1/11). Development study; no test set.','',
       '## Primary: validation updates to reach position-1 accuracy thresholds','',
       f'Validation every {interval} updates on 256 maps. ">10k": not reached (censored; ranked after every run that reaches it).','',
       '| Cell | Parameters | Updates to 50% (per seed) | Median | Updates to 90% (per seed) | Median |','|---|---:|---|---:|---|---:|']
    for cell in CELL_ORDER:
        L.append(f"| {cell} | {report['parameters'][cell]:,} | {fmt_steps(reach[0.5][cell])} | {med(reach[0.5][cell])/1000:g}k | {fmt_steps(reach[0.9][cell])} | {med(reach[0.9][cell])/1000:g}k |")
    L+=['','## Declared decisions (90% threshold)','','| Comparison | Earlier seeds (first/second) | Median difference (updates, + favors first) | Favors |','|---|---:|---:|---|']
    for key,v in comparisons.items():
        x=v['0.9'];who='first' if x['favors_first'] else 'second' if x['favors_second'] else 'neither'
        L.append(f"| {key.replace('_vs_',' vs ')} | {x['first_earlier_seeds']}/{x['first_later_seeds']} | {x['median_difference']:+,.0f} | {who} |")
    L+=['']+[f"- **{k.replace('_',' ')}:** {'yes' if v else 'no'}" for k,v in decisions.items()]
    L+=['','## Secondary: final held-out accuracy (512 maps per seed) and validation area','',
        '| Cell | Position 1, mean ± SD | Answer accuracy | All 12 correct | Validation position-1 area | Train min |','|---|---:|---:|---:|---:|---:|']
    for cell in CELL_ORDER:
        p1=final[cell]['position1']
        L.append(f"| {cell} | {pct(statistics.mean(p1))} ± {100*statistics.stdev(p1):.2f} | {pct(statistics.mean(final[cell]['answer_accuracy']))} | "
                 f"{pct(statistics.mean(final[cell]['sequence_exact']))} | {statistics.mean(auc[cell]):.3f} | {report['training_minutes'][cell]:.0f} |")
    L+=['','![Validation position-1 accuracy by cell](learning_curves.png)','',
        'See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and the per-run evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')

    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax,cell in zip(axes.flat,CELL_ORDER):
        for s in seeds:
            curve=models[name(cell,s)]['validation_curve'];ax.plot([c['step'] for c in curve],[c['position1'] for c in curve],lw=1,label=f'seed {s}')
        ax.axhline(1/11,color='gray',ls=':',lw=1);ax.set_title(f"{cell} ({report['parameters'][cell]:,})");ax.set_ylim(0,1.02)
    for ax in axes[-1]:ax.set_xlabel('update')
    for ax in axes[:,0]:ax.set_ylabel('validation position-1 accuracy')
    axes[0,0].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'learning_curves.{ext}',dpi=180)
    print('\n'.join(L[:20]),flush=True)

if __name__=='__main__':main()
