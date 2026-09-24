"""Report declared tuning trials, frozen winners, paired test outcomes, and costs."""
import json
from pathlib import Path
import numpy as np
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/readout_comparison_v1')


def main():
    selection=json.loads((ROOT/'selection.json').read_text());evaluations={}
    for family in selection['winners']:
        ev=json.loads((ROOT/f'{family}.evaluation.json').read_text())
        assert ev['complete'] and ev['selection_sha256']==file_hash(ROOT/'selection.json')
        assert ev['checkpoint']['sha256']==selection['winners'][family]['checkpoint_sha256']
        for path,digest in ev['source_sha256'].items():assert file_hash(path)==digest
        evaluations[family]=ev
    # Paired map bootstrap. These intervals omit training-seed uncertainty.
    contrasts={}
    for split in ('test_id','test_shuffled'):
        ctm=evaluations['ctm']['results'][split]['metrics']['predictions']
        ids=[r['id'] for r in ctm]
        for family in ('transformer','recurrent_depth'):
            other=evaluations[family]['results'][split]['metrics']['predictions']
            assert [r['id'] for r in other]==ids
            assert all(a['answer']==b['answer'] for a,b in zip(ctm,other))
            delta=np.array([int(a['correct'])-int(b['correct']) for a,b in zip(ctm,other)])
            rng=np.random.default_rng(20260923)
            samples=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            contrasts[f'ctm_minus_{family}_{split}']={'difference':float(delta.mean()),
                'paired_map_bootstrap_95_percentile':np.quantile(samples,[.025,.975]).tolist(),
                'ctm_only_correct':int((delta==1).sum()),'baseline_only_correct':int((delta==-1).sum()),
                'maps':len(delta),'bootstrap_seed':20260923,'resamples':10000,
                'limitation':'Evaluation-map uncertainty only; single training seed; secondary shuffled split reuses primary maps.'}
    curves={}
    for trial in selection['trials']:
        cell=trial['cell'];p=Path('research/runs/readout_comparison_v1/seed17')/cell/'metrics.jsonl'
        assert file_hash(p)==trial['metrics_sha256']
        curves[cell]=[r for r in map(json.loads,p.read_text().splitlines()) if 'validation_readouts' in r]
    report={'complete':True,'selection':selection,'evaluations':evaluations,'paired_test_contrasts':contrasts,
        'analysis_source_sha256':file_hash('scripts/summarize_readout_comparison.py'),
        'scope':'Preliminary one-seed, approximate-parameter and exposure match with three new trials per family; unequal compute and prior tuning history.'}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Across-tick CTM recipe and baseline comparison','',
        'All nine declared trials completed. Three new tuning trials per family, one training seed, identical data exposure, and approximately matched trainable parameters. Validation selects trial, checkpoint and declared readout; frozen winners alone receive test evaluation. This is not a compute-matched or multi-seed architecture result. Earlier project work has inspected these test splits; fresh confirmatory data remain necessary.','',
        '## Frozen winners','',
        '| Family | Recipe | Readout | Selected update | Parameters | Validation CE | Ordered test accuracy | Shuffled test accuracy | Train min (full budget) | Generation ms / batch 32 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for family,w in selection['winners'].items():
        e=evaluations[family]
        lines.append(f"| {family} | {w['cell']} | {w['readout']} | {w['step']} | {w['parameters']['total']:,} | {w['validation_ce']:.6g} | {e['results']['test_id']['metrics']['generation']['exact_match']:.1%} | {e['results']['test_shuffled']['metrics']['generation']['exact_match']:.1%} | {w['training_seconds']/60:.1f} | {1000*e['generation_timing']['median_batch_seconds']:.1f} |")
    lines.extend(['','Both accuracies require unconstrained generation of the answer and EOS. Test CE and per-example predictions are retained in the evaluation JSON. Confidence readout executes all 16 ticks and selects separately for each generated token; it is not early stopping. Generation timing includes its overhead and repeated-prefix decoding on one RTX 3090. Training time covers all 3,000 updates, even when an earlier checkpoint wins.','',
        '## All declared tuning trials','',
        '| Trial | Selected readout | Update | Validation CE | Validation generation accuracy | Train min | Total run min | Peak GiB |','|---|---|---:|---:|---:|---:|---:|---:|'])
    for t in selection['trials']:
        lines.append(f"| {t['cell']} | {t['readout']} | {t['step']} | {t['validation_ce']:.6g} | {t['validation_generation']['exact_match']:.1%} | {t['training_seconds']/60:.1f} | {t['wall_seconds']/60:.1f} | {t['peak_allocated_bytes']/2**30:.3f} |")
    lines.extend(['','## Development cost','', '| Family | New trials | Training GPU min | Total run GPU min |','|---|---:|---:|---:|'])
    for family in selection['winners']:
        rows=[t for t in selection['trials'] if t['model_family']==family]
        lines.append(f"| {family} | {len(rows)} | {sum(t['training_seconds'] for t in rows)/60:.1f} | {sum(t['wall_seconds'] for t in rows)/60:.1f} |")
    lines.extend(['','Total run time includes validation and checkpoint work. These costs exclude earlier development, setup, profiling and final evaluation, so they are not total project costs. CTM has 32 training block applications per sequence; recurrent depth has 34 and Transformer 2. Their blocks have different costs; block counts are not FLOPs. Matching new trial counts does not match GPU budgets or lifetime tuning effort.','',
        '## Paired test differences','', '| CTM minus baseline | Accuracy difference (points) | Paired map bootstrap 95% interval |','|---|---:|---|'])
    for name,r in contrasts.items():
        low,high=r['paired_map_bootstrap_95_percentile']
        lines.append(f"| {name} | {100*r['difference']:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] |")
    lines.extend(['','These intervals resample 128 maps, not training seeds. Ordered and shuffled outcomes reuse the same maps; the intervals are descriptive and not adjusted for multiple comparisons.','',
        '## Validation trajectories','', '![Policy-aware validation curves](validation_curves.png)','',
        'See [tick dynamics](TICK_DYNAMICS.md) for the direct raw/confidence/oracle monotonicity diagnosis at all three CTM recipe checkpoints.','',
        '![Selected recipes: test quality and measured generation time](quality_cost.png)','',
        '## What this run establishes','',
        'At its selected checkpoint, dynamic CTM with confidence selection shows zero aggregate answer-token CE regressions and zero accuracy regressions over the 15 adjacent tick transitions. Uniform has 9 and 3 respectively. This reproduces the reported monotonic trajectory pattern under a label-free selector on this validation set; it is not a per-example or general guarantee. Nevertheless, uniform finishes at a better absolute validation score (76.6% generation accuracy versus 73.4% dynamic). The chosen monotonic penalty lowers dynamic accuracy to 64.1%.', '',
        'Optimizer settings strongly affect the baseline comparison: the standard Transformer reaches 100% validation accuracy at peak LR 3e-4, versus 61.7% at 1e-3 and 14.1% at 3e-3. CTM learning rate was fixed at 1e-3 while its temporal objective was varied. A dedicated CTM learning-rate study is a sensible next development step, before freezing a broader architecture comparison.', '',
        'All three frozen winners fail shuffled-order transfer (8.6–9.4%). The near-perfect ordered baseline results establish fixed-order lookup on this task, not general graph retrieval or multistep reasoning. This limitation motivates shuffled-order training and fresh confirmatory maps.', '',
        '## Interpretation limits','',
        '- Three CTM temporal recipes were compared with three learning rates for each final-CE baseline. This compares complete recipes; it does not isolate auxiliary supervision from architecture. A recurrent baseline with identical auxiliary temporal losses remains a required attribution control.',
        '- Single seed and one elementary ordered lookup task. Approximate parameter match and equal exposure are explicit; compute, architecture-specific state and historical tuning are unequal.',
        '- The exact earlier successful CTM recipe remains unidentified. The monotonic penalty is soft; minimum-entropy selection does not guarantee monotonic accuracy. Gold-aware envelopes are diagnostics only.',
        '- Fresh test maps, seed replication, harder composition tasks, and compute-matched baselines are needed before paper-level claims. No post-test retuning is part of this study.','',
        'See [frozen protocol](PLAN_BEFORE_RUNS.md), `selection.json`, `registry.json`, source snapshots, all retained validation checkpoints, and per-example evaluation files.'])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,family in zip(axes,('ctm','recurrent_depth','transformer')):
        for trial in selection['trials']:
            if trial['model_family']!=family:continue
            rows=curves[trial['cell']]
            for policy in ('final','confidence'):
                if policy not in rows[0]['validation_readouts']:continue
                ax.plot([r['step'] for r in rows],[r['validation_readouts'][policy]['loss'] for r in rows],
                    linestyle='--' if policy=='confidence' else '-',label=trial['cell'].replace(family+'_','')+'/'+policy)
        ax.set(title=family,xlabel='Optimizer updates',ylabel='Validation answer/EOS CE');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Declared trials: seed 17, matched exposure, approximately matched parameters')
    fig.savefig(ROOT/'validation_curves.png',dpi=180);fig.savefig(ROOT/'validation_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');families=list(selection['winners']);x=np.arange(3)
    for j,split in enumerate(('test_id','test_shuffled')):
        axes[0].bar(x+(j-.5)*.35,[100*evaluations[f]['results'][split]['metrics']['generation']['exact_match'] for f in families],.35,label=split)
    axes[0].set(xticks=x,xticklabels=families,ylabel='Exact answer + EOS (%)',ylim=(0,105));axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    axes[1].bar(x,[1000*evaluations[f]['generation_timing']['median_batch_seconds'] for f in families]);axes[1].set(xticks=x,xticklabels=families,ylabel='Median generation ms / batch 32');axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Frozen winners: test quality and measured generation cost on RTX 3090')
    fig.savefig(ROOT/'quality_cost.png',dpi=180);fig.savefig(ROOT/'quality_cost.pdf');plt.close(fig)
    print('\n'.join(lines[:14]))

if __name__=='__main__':main()
