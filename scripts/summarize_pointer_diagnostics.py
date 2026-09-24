"""Report fixed-budget pointer diagnostics and plot their learning curves."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root',type=Path,required=True)
    parser.add_argument('--run_root',type=Path,required=True)
    parser.add_argument('--dataset_root',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    families=['transformer','recurrent_depth','ctm']
    names={'transformer':'Transformer','recurrent_depth':'Recurrent depth','ctm':'CTM'}
    records, curves, baselines={}, {}, {}
    for mode in ('tiny','onehop'):
        records[mode],curves[mode]={},{}
        for family in families:
            record=json.loads((args.results_root/f'{mode}_{family}_seed{args.seed}.summary.json').read_text())
            assert record['training']['complete'] and record['training']['steps']==600
            records[mode][family]=record
            path=args.run_root/f'{mode}_seed{args.seed}'/family/'metrics.jsonl'
            curves[mode][family]=[json.loads(line) for line in path.read_text().splitlines()]
        budgets={(r['training']['examples_seen'],r['training']['tokens_seen'],r['training']['supervised_tokens_seen'])
                 for r in records[mode].values()}
        assert len(budgets)==1
        train=[json.loads(line) for line in (args.dataset_root/mode/'pointer/train.jsonl').read_text().splitlines()]
        test=[json.loads(line) for line in (args.dataset_root/mode/'pointer/test_id.jsonl').read_text().splitlines()]
        majority=Counter(r['answer'] for r in train).most_common(1)[0][0]
        baselines[mode]={'majority_answer':majority,'heldout_majority_accuracy':sum(r['answer']==majority for r in test)/len(test),
                         'uniform_node_expected_accuracy':1/8,'exclude_start_expected_accuracy':1/7}
    def values(mode,family,split):
        return next(iter(records[mode][family]['metrics'][split].values()))
    gates={mode:{family:values(mode,family,'train' if mode=='tiny' else 'test_id')['exact_match'] >= (0.95 if mode=='tiny' else 0.9)
                 for family in families} for mode in records}
    persistence={}
    for family in families:
        evaluation=json.loads((args.results_root/f'tiny_{family}_seed{args.seed}.eval.json').read_text())
        def predictions(split):
            return next(iter(evaluation['results'][split]['depths'].values()))['predictions']
        original={row['id']:row['answer'] for row in predictions('train_probe')}
        changed=predictions('train_new_query')
        persistence[family]={'old_answer_predictions':sum(row['prediction']==original[row['id']] for row in changed),
                             'examples':len(changed)}
    report={'purpose':'exploratory diagnostics; single seed; unmatched parameter and compute budgets',
            'seed':args.seed,'gates':gates,'baselines':baselines,'old_answer_persistence':persistence,'runs':records}
    (args.results_root/'diagnostic_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Pointer diagnostic results','',
        'All six runs used the unchanged algorithmic model presets, seed 17, and 600 updates × 32 examples. Tiny fitting uses the final checkpoint; one-hop lookup uses best validation answer/EOS CE. Exact match requires a generated answer and EOS. Parameter and compute budgets are unmatched.','']
    for mode,title in [('tiny','Tiny-set fitting (32 fixed training examples)'),('onehop','One-hop lookup (2,048 training maps)')]:
        lines.extend(['## '+title,'',
            '| Model | Train | Paired train probe | Shuffled edges | New start | Held-out maps | Selected step | Training seconds |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for family in families:
            r=records[mode][family]
            accuracies=[values(mode,family,s)['exact_match'] for s in ('train','train_probe','train_reordered','train_new_query','test_id')]
            lines.append('| '+names[family]+' | '+' | '.join(f'{v:.1%}' for v in accuracies)+
                         f" | {r['checkpoint']['training_counters']['step']} | {r['training']['training_seconds']:.1f} |")
        lines.extend(['', 'Paired probes reuse training maps and are not held-out generalization scores. '+
                      ('All 32 maps are probed.' if mode=='tiny' else 'A fixed 256-map training subset is probed.'),''])
    lines.extend(['## Gates recorded before training','', '| Model | Tiny fitting ≥95% | Held-out one-hop ≥90% |','|---|---|---|'])
    for family in families:
        lines.append(f"| {names[family]} | {'Pass' if gates['tiny'][family] else 'Fail'} | {'Pass' if gates['onehop'][family] else 'Fail'} |")
    lines.extend(['', 'Eight-node uniform guessing is 12.5%; excluding the known-impossible start node gives 14.3% expected accuracy without reading edges.',''])
    for mode in ('tiny','onehop'):
        b=baselines[mode]
        lines.append(f"- {mode}: always emit training-majority label `{b['majority_answer']}` → {b['heldout_majority_accuracy']:.1%} on held-out maps.")
    lines.extend(['', 'After changing the start node on the tiny training maps, the prediction still equals the old (now incorrect) answer in '+
                  ', '.join(f"{persistence[f]['old_answer_predictions']}/32 cases for {names[f]}" for f in families)+'.'])
    lines.extend(['', 'Training timing includes batching, transfers, forward/backward, and optimizer updates, excluding validation and checkpoint writes. Both GPU queues ran concurrently.','',
                  '![Diagnostic learning curves](learning_curves.png)','',
                  'Per-example predictions and difficulty groups are in the adjacent `.eval.json` files. See the [protocol and interpretation](../../POINTER_DIAGNOSTICS.md) and the preserved [pre-run plan](PLAN_BEFORE_RUNS.md).'])
    (args.results_root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    colors={'transformer':'#2369bd','recurrent_depth':'#d87818','ctm':'#28944a'}
    for family in families:
        tiny=curves['tiny'][family]
        # Nonoverlapping 25-update averages reduce batch/numeric noise.
        groups=[tiny[i:i+25] for i in range(0,len(tiny),25)]
        axes[0].plot([g[-1]['step'] for g in groups],[sum(r['loss'] for r in g)/len(g) for g in groups],
                     color=colors[family],label=names[family])
        onehop=[r for r in curves['onehop'][family] if 'validation' in r]
        axes[1].plot([r['step'] for r in onehop],[r['validation']['loss'] for r in onehop],
                     marker='o',markersize=3,color=colors[family],label=names[family])
    axes[0].set(title='Tiny fitting: training loss',xlabel='Optimizer updates',ylabel='Answer/EOS CE (25-update mean)',yscale='log')
    axes[1].set(title='One-hop lookup: validation loss',xlabel='Optimizer updates',ylabel='Answer/EOS CE')
    for ax in axes: ax.grid(alpha=0.25); ax.legend()
    fig.suptitle('Pointer diagnostics, seed 17; parameters and compute unmatched')
    fig.savefig(args.results_root/'learning_curves.png',dpi=180)
    fig.savefig(args.results_root/'learning_curves.pdf')
    plt.close(fig)
    print('\n'.join(lines[:28]))


if __name__ == '__main__':
    main()
