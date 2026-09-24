"""Summarize ordered lookup, paired order transfer, and prior shuffled training."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root',type=Path,required=True)
    parser.add_argument('--run_root',type=Path,required=True)
    parser.add_argument('--previous_results',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    families=['transformer','recurrent_depth','ctm']
    names={'transformer':'Transformer','recurrent_depth':'Recurrent depth','ctm':'CTM'}
    records,previous,paired,curves={},{},{},{}
    def metric(record,split): return next(iter(record['metrics'][split].values()))
    for family in families:
        records[family]=json.loads((args.results_root/f'ordered_{family}_seed{args.seed}.summary.json').read_text())
        previous[family]=json.loads((args.previous_results/f'onehop_{family}_seed{args.seed}.summary.json').read_text())
        evaluation=json.loads((args.results_root/f'ordered_{family}_seed{args.seed}.eval.json').read_text())
        def predictions(split): return next(iter(evaluation['results'][split]['depths'].values()))['predictions']
        ordered={r['id']:r for r in predictions('test_id')}
        shuffled={r['id']:r for r in predictions('test_shuffled')}
        assert ordered.keys()==shuffled.keys()
        counts={'both_correct':0,'ordered_only_correct':0,'shuffled_only_correct':0,'neither_correct':0}
        for key,row in ordered.items():
            other=shuffled[key]
            assert row['answer']==other['answer']
            label=('both_correct' if row['correct'] and other['correct'] else
                   'ordered_only_correct' if row['correct'] else
                   'shuffled_only_correct' if other['correct'] else 'neither_correct')
            counts[label]+=1
        paired[family]=counts
        path=args.run_root/family/'metrics.jsonl'
        curves[family]=[json.loads(line) for line in path.read_text().splitlines()]
        assert records[family]['training']['complete'] and records[family]['training']['steps']==600
        for counter in ('examples_seen','tokens_seen','supervised_tokens_seen'):
            assert records[family]['training'][counter]==previous[family]['training'][counter]
    gates={family:metric(r,'test_id')['exact_match']>=0.9 for family,r in records.items()}
    summary={'purpose':'single-seed ordered-edge learning control; unmatched parameters and compute',
             'seed':args.seed,'ordered_lookup_gate':gates,'paired_heldout_counts':paired,
             'runs':records,'previous_shuffled_runs':previous}
    (args.results_root/'ordered_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Ordered-edge lookup results','',
           'Same one-hop maps, queries, answers, example order, seed 17, and 600-update presets as the earlier shuffled-edge control. Only edge presentation was sorted for training/validation/primary evaluation. Best validation answer/EOS CE selects the checkpoint.','',
           '| Model | Train | Held-out ordered | Same held-out maps shuffled | New start on training maps | Lookup ≥90% | Selected step | Training seconds |',
           '|---|---:|---:|---:|---:|---|---:|---:|']
    for family in families:
        r=records[family]
        values=[metric(r,s)['exact_match'] for s in ('train','test_id','test_shuffled','train_new_query')]
        lines.append('| '+names[family]+' | '+' | '.join(f'{v:.1%}' for v in values)+
                     f" | {'Pass' if gates[family] else 'Fail'} | {r['checkpoint']['training_counters']['step']} | {r['training']['training_seconds']:.1f} |")
    lines.extend(['','The changed-start probe deliberately reuses training maps. The two held-out columns use exactly the same 128 unseen maps and queries. All accuracies require unrestricted greedy answer generation followed by EOS.','',
                  '## Paired held-out outcomes','',
                  '| Model | Both correct | Ordered only | Shuffled only | Neither |',
                  '|---|---:|---:|---:|---:|'])
    for family in families:
        lines.append('| '+names[family]+' | '+' | '.join(str(v) for v in paired[family].values())+' |')
    lines.extend(['','## Earlier shuffled-edge training control','',
                  '| Model | Prior held-out accuracy | Current ordered held-out accuracy |', '|---|---:|---:|'])
    for family in families:
        lines.append(f"| {names[family]} | {metric(previous[family],'test_id')['exact_match']:.1%} | {metric(records[family],'test_id')['exact_match']:.1%} |")
    lines.extend(['','The two columns use separately trained models. Every run sees 19,200 examples, 1,036,800 real input tokens, and 38,400 supervised tokens; parameters and compute remain unmatched. CTM now runs on GPU 0 and the baselines on GPU 1; all devices are RTX 3090s. Timing is implementation-specific and excludes validation/checkpoint/evaluation work.','',
                  '![Validation learning curves](learning_curves.png)','',
                  'One seed and this fixed-position control do not establish graph reasoning or architectural superiority. See the [protocol](../../ORDERED_POINTER.md), [pre-run plan](PLAN_BEFORE_RUNS.md), and per-example `.eval.json` files.'])
    (args.results_root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'transformer':'#2369bd','recurrent_depth':'#d87818','ctm':'#28944a'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for family in families:
        rows=[r for r in curves[family] if 'validation' in r]
        axes[0].plot([r['step'] for r in rows],[r['validation']['loss'] for r in rows],marker='o',markersize=3,
                     color=colors[family],label=names[family])
    axes[0].set(title='Ordered lookup: validation loss',xlabel='Optimizer updates',ylabel='Answer/EOS cross-entropy')
    axes[0].grid(alpha=0.25);axes[0].legend()
    import numpy as np
    x=np.arange(len(families));width=0.34
    axes[1].bar(x-width/2,[100*metric(records[f],'test_id')['exact_match'] for f in families],width,label='Ordered')
    axes[1].bar(x+width/2,[100*metric(records[f],'test_shuffled')['exact_match'] for f in families],width,label='Same maps shuffled')
    axes[1].set(title='Paired held-out evaluation',ylabel='Exact match (%)',xticks=x,xticklabels=[names[f] for f in families],ylim=(0,105))
    axes[1].axhline(100/7,color='gray',linestyle=':',label='Exclude-start guessing')
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=0.2)
    fig.suptitle('Ordered-edge control, seed 17; parameters and compute unmatched')
    fig.savefig(args.results_root/'learning_curves.png',dpi=180)
    fig.savefig(args.results_root/'learning_curves.pdf')
    plt.close(fig)
    print('\n'.join(lines[:12]))


if __name__=='__main__':
    main()
