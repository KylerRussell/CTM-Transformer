"""Summarize the locked confirmation and apply its declared decisions (C1, C2 and secondary)."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctm_transformer.experiment import file_hash
from scripts.summarize_group_s3 import correct_prefix

ROOT=Path('research/results/sync_confirmation_v1')
CELL_ORDER=('transformer','rdt','rdt_wide','sync','current')
RECURRENT=CELL_ORDER[1:]
MIN_SEEDS,MIN_POSITIONS=6,2


def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.1f}%'


def exceeds(a,b,min_seeds=MIN_SEEDS,min_positions=MIN_POSITIONS):
    """Paired over seeds: a exceeds b if larger in >= min_seeds seeds and the median difference is >= min_positions."""
    diffs=[x-y for x,y in zip(a,b)];median=statistics.median(diffs)
    return {'differences':diffs,'larger_seeds':sum(d>0 for d in diffs),'smaller_seeds':sum(d<0 for d in diffs),'median_difference':median,
            'first_exceeds':sum(d>0 for d in diffs)>=min_seeds and median>=min_positions,
            'second_exceeds':sum(d<0 for d in diffs)>=min_seeds and -median>=min_positions}


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
    pairs=(('sync','rdt'),('sync','rdt_wide'),('current','rdt'),('current','rdt_wide'),('sync','current'),('rdt','transformer'),('rdt_wide','rdt'))
    comparisons={f'{a}_vs_{b}':exceeds(primary[a],primary[b]) for a,b in pairs}
    steps_used={cell:exceeds(prefix[cell]['16'],prefix[cell]['4']) for cell in RECURRENT}
    decisions={'C1_sync_confirmed':comparisons['sync_vs_rdt']['first_exceeds'] and comparisons['sync_vs_rdt_wide']['first_exceeds'],
        'C2_current_confirmed':comparisons['current_vs_rdt']['first_exceeds'] and comparisons['current_vs_rdt_wide']['first_exceeds'],
        'secondary_sync_exceeds_current':comparisons['sync_vs_current']['first_exceeds'],
        'secondary_current_exceeds_sync':comparisons['sync_vs_current']['second_exceeds'],
        'secondary_rdt_exceeds_transformer':comparisons['rdt_vs_transformer']['first_exceeds'],
        **{f'secondary_{cell}_uses_steps':steps_used[cell]['first_exceeds'] for cell in RECURRENT}}
    decay={}
    import torch
    for s in seeds:
        r=torch.load(models[name('sync',s)]['checkpoint'],map_location='cpu',weights_only=False)['model_state_dict']['sync.r'].flatten()
        decay[str(s)]={'mean':float(r.mean()),'min':float(r.min()),'max':float(r.max())}
    report={'complete':True,'locked_test_evaluated_once':True,'rule':f'exceeds: larger in >= {MIN_SEEDS} of {len(seeds)} seeds and median paired difference >= {MIN_POSITIONS} positions',
        'correct_prefix':prefix,'primary_correct_prefix':primary,'mean_accuracy_positions_9_16':mid,'mean_accuracy_positions_17_32':extrapolation,
        'comparisons':comparisons,'steps_used':steps_used,'decisions':decisions,'sync_decay_rates':decay,
        'parameters':{cell:models[name(cell,seeds[0])]['parameters']['total'] for cell in CELL_ORDER},
        'training_minutes':{cell:statistics.mean(models[name(cell,s)]['training_seconds'] for s in seeds)/60 for cell in CELL_ORDER},
        'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in ev}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

    med=statistics.median
    L=['# Locked confirmation: second-order state features in the attention queries','',
       f'{len(models)} runs ({len(CELL_ORDER)} cells × {len(seeds)} unused seeds {seeds}), S₃ running products, training lengths 1–16. '
       'The locked test (1,024 length-32 words per seed) was evaluated once, after the checkpoint freeze. '
       f'Correct prefix = leading positions at ≥ 90% test accuracy. Rule: {report["rule"]}.','',
       '## Primary: correct prefix on the locked test (Transformer T=1, others T=16)','',
       '| Cell | Parameters | Per seed | Median | Positions 9–16 | Positions 17–32 | Median prefix T=4/8/16/32 |','|---|---:|---|---:|---:|---:|---|']
    for cell in CELL_ORDER:
        scaling='—' if cell=='transformer' else '/'.join(f"{med(prefix[cell][T]):g}" for T in ('4','8','16','32'))
        L.append(f"| {cell} | {report['parameters'][cell]:,} | {', '.join(map(str,primary[cell]))} | {med(primary[cell]):g} | "
                 f"{pct(statistics.mean(mid[cell]))} | {pct(statistics.mean(extrapolation[cell]))} | {scaling} |")
    L+=['','## Declared decisions','','| Comparison (correct prefix) | Larger seeds (first/second) | Median difference | Exceeds |','|---|---:|---:|---|']
    for key,x in comparisons.items():
        who='first' if x['first_exceeds'] else 'second' if x['second_exceeds'] else 'neither'
        L.append(f"| {key.replace('_vs_',' vs ')} | {x['larger_seeds']}/{x['smaller_seeds']} | {x['median_difference']:+g} | {who} |")
    for cell,x in steps_used.items():
        L.append(f"| {cell}: T=16 vs T=4 | {x['larger_seeds']}/{x['smaller_seeds']} | {x['median_difference']:+g} | {'yes' if x['first_exceeds'] else 'no'} |")
    L+=['']+[f"- **{k.replace('_',' ')}:** {'yes' if v else 'no'}" for k,v in decisions.items()]
    L+=['','![Accuracy by position](accuracy_by_position.png)','','See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and the per-run evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')

    fig,axes=plt.subplots(1,len(CELL_ORDER),figsize=(18,3.8),sharey=True)
    for ax,cell in zip(axes,CELL_ORDER):
        for T in ev[name(cell,seeds[0])]['results']:
            ax.plot(range(1,33),[statistics.mean(positions(cell,s,T)[i] for s in seeds) for i in range(32)],label=f'T={T}')
        ax.axvline(16.5,color='gray',ls='--',lw=1);ax.axhline(1/6,color='gray',ls=':',lw=1);ax.set_title(cell);ax.set_ylim(0,1.02);ax.legend(fontsize=7)
        ax.set_xlabel('position')
    axes[0].set_ylabel('locked-test accuracy (mean over seeds)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'accuracy_by_position.{ext}',dpi=180)
    print('\n'.join(L[:14]),flush=True)

if __name__=='__main__':main()
