"""Summarize the S3 group-word study and apply its declared decision rules."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/group_s3_v1')
CELL_ORDER=('transformer','ctm','rdt','rdt_wide','history','sync','sync_rdt')
RECURRENT=CELL_ORDER[1:]
THRESHOLD,MIN_SEEDS,MIN_POSITIONS=0.9,4,2


def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.1f}%'


def correct_prefix(by_position,threshold=THRESHOLD):
    """Number of leading positions whose accuracy is at least the threshold."""
    n=0
    for accuracy in by_position:
        if accuracy<threshold:break
        n+=1
    return n


def exceeds(a,b):
    """Paired over seeds: a exceeds b if larger in >= MIN_SEEDS seeds and the median difference is >= MIN_POSITIONS."""
    diffs=[x-y for x,y in zip(a,b)]
    return {'differences':diffs,'larger_seeds':sum(d>0 for d in diffs),'smaller_seeds':sum(d<0 for d in diffs),
            'median_difference':statistics.median(diffs),
            'first_exceeds':sum(d>0 for d in diffs)>=MIN_SEEDS and statistics.median(diffs)>=MIN_POSITIONS,
            'second_exceeds':sum(d<0 for d in diffs)>=MIN_SEEDS and -statistics.median(diffs)>=MIN_POSITIONS}


def main():
    registry=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');models={t['cell']:t for t in frozen['models']}
    ev={c:read(ROOT/f'{c}.evaluation.json') for c in models}
    for e in ev.values():assert e['complete'] and e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
    seeds=registry['seeds'];name=lambda cell,s:f'{cell}_seed{s}'
    trained=lambda cell:'1' if cell=='transformer' else '16'
    positions=lambda cell,s,T:ev[name(cell,s)]['results'][T]['by_position']
    prefix={cell:{T:[correct_prefix(positions(cell,s,T)) for s in seeds] for T in ev[name(cell,seeds[0])]['results']} for cell in CELL_ORDER}
    primary={cell:prefix[cell][trained(cell)] for cell in CELL_ORDER}
    mid={cell:[statistics.mean(positions(cell,s,trained(cell))[8:16]) for s in seeds] for cell in CELL_ORDER}
    extrapolation={cell:[statistics.mean(positions(cell,s,trained(cell))[16:32]) for s in seeds] for cell in CELL_ORDER}
    comparisons={f'{a}_vs_{b}':exceeds(primary[a],primary[b]) for a,b in (('rdt','transformer'),('rdt','ctm'),('ctm','transformer'),
        ('sync','rdt'),('sync','rdt_wide'),('history','rdt'),('history','rdt_wide'),('sync_rdt','rdt'),('sync_rdt','rdt_wide'),('rdt_wide','rdt'))}
    steps_used={cell:exceeds(prefix[cell]['16'],prefix[cell]['4']) for cell in RECURRENT}
    decisions={'recurrence_beats_fixed_depth':comparisons['rdt_vs_transformer']['first_exceeds'],
        'rdt_beats_ctm':comparisons['rdt_vs_ctm']['first_exceeds'],
        'sync_extends_state':comparisons['sync_vs_rdt']['first_exceeds'] and comparisons['sync_vs_rdt_wide']['first_exceeds'],
        'history_extends_state':comparisons['history_vs_rdt']['first_exceeds'] and comparisons['history_vs_rdt_wide']['first_exceeds'],
        'sync_rdt_extends_state':comparisons['sync_rdt_vs_rdt']['first_exceeds'] and comparisons['sync_rdt_vs_rdt_wide']['first_exceeds'],
        **{f'{cell}_uses_steps':steps_used[cell]['first_exceeds'] for cell in RECURRENT}}
    report={'complete':True,'no_test_evaluation':True,'threshold':THRESHOLD,
        'decision_rule':f'A exceeds B if its correct prefix is larger in >= {MIN_SEEDS} of {len(seeds)} seeds and the median paired difference is >= {MIN_POSITIONS} positions',
        'correct_prefix':prefix,'primary_correct_prefix':primary,'mean_accuracy_positions_9_16':mid,'mean_accuracy_positions_17_32':extrapolation,
        'comparisons':comparisons,'steps_used':steps_used,'decisions':decisions,
        'parameters':{cell:models[name(cell,seeds[0])]['parameters']['total'] for cell in CELL_ORDER},
        'training_minutes':{cell:statistics.mean(models[name(cell,s)]['training_seconds'] for s in seeds)/60 for cell in CELL_ORDER},
        'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in ev}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

    med=lambda xs:statistics.median(xs)
    L=['# S3 word problem: serial state across architectures','',
       f'{len(models)} runs ({len(CELL_ORDER)} cells × {len(seeds)} seeds {seeds}), {registry["primary_checkpoint_step"]:,} updates, peak LR {registry["learning_rate"]}. '
       'Training uses fresh words of lengths 1–16. Evaluation uses 1,024 held-out words of length 32 per seed, labeled at every position. Positions 1–16 are trained lengths; 17–32 are extrapolation. '
       f'Chance 1/6. **Correct prefix** = number of leading positions with accuracy ≥ {THRESHOLD:.0%}. Development data; no test set.','',
       '## Primary: correct prefix at the trained step budget (Transformer T=1, others T=16)','',
       '| Cell | Parameters | Per seed | Median | Mean accuracy, positions 9–16 | Mean accuracy, positions 17–32 | Train min |','|---|---:|---|---:|---:|---:|---:|']
    for cell in CELL_ORDER:
        L.append(f"| {cell} | {report['parameters'][cell]:,} | {', '.join(map(str,primary[cell]))} | {med(primary[cell]):g} | "
                 f"{pct(statistics.mean(mid[cell]))} | {pct(statistics.mean(extrapolation[cell]))} | {report['training_minutes'][cell]:.0f} |")
    L+=['','## Correct prefix by inference step budget (median across seeds)','','| Cell | T=4 | T=8 | T=16 | T=32 |','|---|---:|---:|---:|---:|']
    for cell in RECURRENT:L.append(f'| {cell} | '+' | '.join(f"{med(prefix[cell][T]):g}" for T in ('4','8','16','32'))+' |')
    L+=['','## Declared decisions','','| Comparison (correct prefix) | Larger seeds (first/second) | Median difference | Exceeds |','|---|---:|---:|---|']
    for key,x in comparisons.items():
        who='first' if x['first_exceeds'] else 'second' if x['second_exceeds'] else 'neither'
        L.append(f"| {key.replace('_vs_',' vs ')} | {x['larger_seeds']}/{x['smaller_seeds']} | {x['median_difference']:+g} | {who} |")
    for cell,x in steps_used.items():
        L.append(f"| {cell}: T=16 vs T=4 | {x['larger_seeds']}/{x['smaller_seeds']} | {x['median_difference']:+g} | {'yes' if x['first_exceeds'] else 'no'} |")
    L+=['']+[f"- **{k.replace('_',' ')}:** {'yes' if v else 'no'}" for k,v in decisions.items()]
    L+=['','![Accuracy by position](accuracy_by_position.png)','','See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and the per-run evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')

    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True)
    for ax,cell in zip(axes.flat,CELL_ORDER):
        for T in ev[name(cell,seeds[0])]['results']:
            curve=[statistics.mean(positions(cell,s,T)[i] for s in seeds) for i in range(32)]
            ax.plot(range(1,33),curve,label=f'T={T}')
        ax.axvline(16.5,color='gray',ls='--',lw=1);ax.axhline(1/6,color='gray',ls=':',lw=1);ax.set_title(cell);ax.set_ylim(0,1.02);ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    for ax in axes[-1]:ax.set_xlabel('position (dashed: end of trained lengths)')
    for ax in axes[:,0]:ax.set_ylabel('accuracy (mean over seeds)')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'accuracy_by_position.{ext}',dpi=180)
    print('\n'.join(L[:16]),flush=True)

if __name__=='__main__':main()
