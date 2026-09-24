"""Report fixed-slot copying and paired order/query transfer probes."""
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
        records[family]=json.loads((args.results_root/f'fixedcopy_{family}_seed{args.seed}.summary.json').read_text())
        previous[family]=json.loads((args.previous_results/f'ordered_{family}_seed{args.seed}.summary.json').read_text())
        evaluation=json.loads((args.results_root/f'fixedcopy_{family}_seed{args.seed}.eval.json').read_text())
        def predictions(split): return next(iter(evaluation['results'][split]['depths'].values()))['predictions']
        paired[family]={}
        for source,probe in [('test_id','test_shuffled'),('test_id','test_new_query'),('train_probe','train_new_query')]:
            reference={r['id']:r for r in predictions(source)}
            changed={r['id']:r for r in predictions(probe)}
            assert reference.keys()==changed.keys()
            counts={'examples':len(reference),'both_correct':0,'original_only_correct':0,
                    'probe_only_correct':0,'neither_correct':0,'probe_outputs_original_answer':0,
                    'same_prediction':0}
            for key,row in reference.items():
                other=changed[key]
                if 'new_query' in probe: assert row['answer']!=other['answer']
                else: assert row['answer']==other['answer']
                label=('both_correct' if row['correct'] and other['correct'] else
                       'original_only_correct' if row['correct'] else
                       'probe_only_correct' if other['correct'] else 'neither_correct')
                counts[label]+=1
                counts['probe_outputs_original_answer']+=int(other['prediction']==row['answer'] and other['terminated'])
                counts['same_prediction']+=int(other['prediction']==row['prediction'])
            paired[family][probe]=counts
        curves[family]=[json.loads(line) for line in (args.run_root/family/'metrics.jsonl').read_text().splitlines()]
        assert records[family]['training']['complete'] and records[family]['training']['steps']==600
        for counter in ('examples_seen','tokens_seen','supervised_tokens_seen'):
            assert records[family]['training'][counter]==previous[family]['training'][counter]
    gates={f:metric(r,'test_id')['exact_match']>=0.9 for f,r in records.items()}
    summary={'purpose':'single-seed fixed-slot copying control; unmatched parameters and compute',
             'seed':args.seed,'fixed_copy_gate':gates,'paired_probe_counts':paired,
             'runs':records,'previous_variable_query_runs':previous}
    (args.results_root/'fixed_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Fixed-slot value-copy results','',
           'Same maps, splits, row order, prompt lengths, seed 17, and 600-update presets as the ordered-edge pilot. Training, validation, and primary evaluation now always query A; labels are its successors. Best validation answer/EOS CE selects each checkpoint.','',
           '| Model | Train | Held-out copy A | Same maps shuffled | Same maps query B | Copy ≥90% | Selected step | Training seconds |',
           '|---|---:|---:|---:|---:|---|---:|---:|']
    for family in families:
        r=records[family]
        values=[metric(r,s)['exact_match'] for s in ('train','test_id','test_shuffled','test_new_query')]
        lines.append('| '+names[family]+' | '+' | '.join(f'{v:.1%}' for v in values)+
                     f" | {'Pass' if gates[family] else 'Fail'} | {r['checkpoint']['training_counters']['step']} | {r['training']['training_seconds']:.1f} |")
    lines.extend(['','All held-out columns reuse the same 128 unseen maps. Query B is absent from training/validation and always changes the correct answer. Its score measures out-of-distribution transfer. All exact matches require unrestricted greedy answer generation followed by EOS.','',
                  '## Query intervention on held-out maps','',
                  '| Model | Correct B answer | Outputs original A answer with EOS | Identical prediction before/after |',
                  '|---|---:|---:|---:|'])
    for family in families:
        c=paired[family]['test_new_query']
        lines.append(f"| {names[family]} | {metric(records[family],'test_new_query')['correct']}/128 | {c['probe_outputs_original_answer']}/128 | {c['same_prediction']}/128 |")
    lines.extend(['','## Earlier variable-query training control','',
                  '| Model | Earlier ordered variable-query held-out | Current fixed-query held-out |', '|---|---:|---:|'])
    for family in families:
        lines.append(f"| {names[family]} | {metric(previous[family],'test_id')['exact_match']:.1%} | {metric(records[family],'test_id')['exact_match']:.1%} |")
    lines.extend(['','These columns use separately trained models with different query/label distributions. They are not paired correctness comparisons. Every run sees 19,200 examples, 1,036,800 real input tokens, and 38,400 supervised tokens. Parameters and compute remain unmatched. Training time excludes validation, checkpoint, and evaluation work.','',
                  '![Learning curves and transfer](learning_curves.png)','',
                  'See the [protocol and interpretation](../../FIXED_POINTER.md), [pre-run plan](PLAN_BEFORE_RUNS.md), raw learning curves under the run directories, and per-example `.eval.json` files. This exploratory control does not establish arbitrary lookup, multi-step reasoning, or an architecture ranking.'])
    (args.results_root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for family in families:
        rows=[r for r in curves[family] if 'validation' in r]
        axes[0].plot([r['step'] for r in rows],[r['validation']['loss'] for r in rows],marker='o',markersize=3,label=names[family])
    axes[0].set(title='Fixed-copy validation loss',xlabel='Optimizer updates',ylabel='Answer/EOS cross-entropy')
    axes[0].grid(alpha=0.25);axes[0].legend()
    x=np.arange(len(families));width=0.25
    for offset,split,label in [(-1,'test_id','Fixed query A'),(0,'test_shuffled','Shuffled edges'),(1,'test_new_query','Unseen query B')]:
        axes[1].bar(x+offset*width,[100*metric(records[f],split)['exact_match'] for f in families],width,label=label)
    axes[1].set(title='Same held-out maps: transfer probes',ylabel='Exact match (%)',xticks=x,xticklabels=[names[f] for f in families],ylim=(0,110))
    axes[1].axhline(90,color='gray',linestyle=':',label='Copy gate (A only)')
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=0.2)
    fig.suptitle('Fixed-slot copying, seed 17; parameters and compute unmatched')
    fig.savefig(args.results_root/'learning_curves.png',dpi=180)
    fig.savefig(args.results_root/'learning_curves.pdf')
    plt.close(fig)
    print('\n'.join(lines[:12]))


if __name__=='__main__':
    main()
