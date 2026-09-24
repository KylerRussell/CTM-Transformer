"""Summarize the 3,000-update ordered lookup experiment against its 600-update pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path

FAMILIES=('transformer','recurrent_depth','ctm')
NAMES={'transformer':'Transformer','recurrent_depth':'Recurrent depth','ctm':'CTM'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root',type=Path,default=Path('research/results/ordered_long_v1'))
    parser.add_argument('--run_root',type=Path,default=Path('research/runs/ordered_long_v1/seed17'))
    parser.add_argument('--previous_results',type=Path,default=Path('research/results/ordered_pointer_v1'))
    parser.add_argument('--previous_runs',type=Path,default=Path('research/runs/ordered_pointer_v1/seed17'))
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    def read(path):return json.loads(path.read_text())
    def metric(record,split):return next(iter(record['metrics'][split].values()))
    records,previous,curves,old_curves,paired,audits,by_start={},{},{},{},{},{},{}
    expected={'examples_seen':96000,'tokens_seen':5184000,'supervised_tokens_seen':192000}
    for family in FAMILIES:
        record=read(args.results_root/f'ordered_{family}_seed{args.seed}.summary.json')
        old=read(args.previous_results/f'ordered_{family}_seed{args.seed}.summary.json')
        run=read(args.run_root/family/'research_run.json')
        old_run=read(args.previous_runs/family/'research_run.json')
        assert record['training']['complete'] and record['training']['steps']==3000
        assert old['training']['complete'] and old['training']['steps']==600
        for counter,value in expected.items():
            assert record['training'][counter]==value==5*old['training'][counter]
        assert run['seed']==old_run['seed']==args.seed
        assert run['data_seed']==old_run['data_seed'] and run['depth_seed']==old_run['depth_seed']
        assert run['code_sha256']==old_run['code_sha256'], 'Training implementation changed'
        assert run['parameters']==old_run['parameters']
        assert record['dataset_manifest_sha256']==old['dataset_manifest_sha256']
        assert run['tokenizer_sha256']==old_run['tokenizer_sha256']
        changed={k:[v,run['effective_config'][k]] for k,v in old_run['effective_config'].items()
                 if v!=run['effective_config'][k]}
        assert changed['max_steps']==[600,3000]
        assert set(changed)<={'max_steps','device','checkpoint_dir'}
        rows=[json.loads(line) for line in (args.run_root/family/'metrics.jsonl').read_text().splitlines()]
        assert len(rows)==3000 and [r['step'] for r in rows]==list(range(1,3001))
        for row in rows:
            s=row['step']
            expected_lr=1e-3*(s/30 if s<=30 else 0.1+0.45*(1+math.cos(math.pi*(s-30)/2970)))
            assert math.isclose(row['lr'],expected_lr,rel_tol=1e-12)
            assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
        validation=[r for r in rows if 'validation' in r]
        assert len(validation)==30
        best=min(validation,key=lambda r:r['validation']['loss'])
        assert record['checkpoint']['training_counters']['step']==best['step']
        curves[family]=validation
        old_curves[family]=[json.loads(line) for line in (args.previous_runs/family/'metrics.jsonl').read_text().splitlines() if '"validation"' in line]
        evaluation=read(args.results_root/f'ordered_{family}_seed{args.seed}.eval.json')
        assert evaluation['complete']
        assert hashlib.sha256(Path(record['checkpoint']['path']).read_bytes()).hexdigest()==record['checkpoint']['sha256']
        def predictions(split):return next(iter(evaluation['results'][split]['depths'].values()))['predictions']
        by_start[family]={}
        for split in ('train','validation','test_id'):
            rows_by_id={r['id']:r for r in (json.loads(line) for line in
                        (Path(run['effective_config']['data_path']).parent/(split+'.jsonl')).read_text().splitlines())}
            counts={node:{'correct':0,'examples':0} for node in 'ABCDEFGH'}
            for pred in predictions(split):
                row=rows_by_id[pred['id']]
                assert pred['answer']==row['answer']
                counts[row['start']]['examples']+=1
                counts[row['start']]['correct']+=int(pred['correct'])
            for c in counts.values():
                c['exact_match']=c['correct']/c['examples'] if c['examples'] else None
            by_start[family][split]=counts
        paired[family]={}
        for source,probe in [('test_id','test_shuffled'),('train_probe','train_new_query')]:
            reference={r['id']:r for r in predictions(source)}
            changed_predictions={r['id']:r for r in predictions(probe)}
            assert reference.keys()==changed_predictions.keys()
            counts={'examples':len(reference),'both_correct':0,'original_only_correct':0,
                    'probe_only_correct':0,'neither_correct':0,'probe_outputs_original_answer':0}
            for key,row in reference.items():
                other=changed_predictions[key]
                assert (row['answer']!=other['answer'])==('new_query' in probe)
                label=('both_correct' if row['correct'] and other['correct'] else
                       'original_only_correct' if row['correct'] else
                       'probe_only_correct' if other['correct'] else 'neither_correct')
                counts[label]+=1
                counts['probe_outputs_original_answer']+=int(other['prediction']==row['answer'] and other['terminated'])
            paired[family][probe]=counts
        records[family],previous[family]=record,old
        audits[family]={'effective_config_changes':changed,'budget_counters':expected,
                       'all_learning_rates_verified':True,'training_source_identical_to_600_update_run':True,
                       'selected_validation_step':best['step'],'selected_validation_ce':best['validation']['loss'],
                       'last_validation_ce':validation[-1]['validation']['loss']}
    gates={f:metric(r,'test_id')['exact_match']>=0.9 for f,r in records.items()}
    summary={'purpose':'single-seed longer-budget/schedule ordered lookup development experiment; unmatched parameters and compute',
             'seed':args.seed,'ordered_lookup_gate':gates,'audits':audits,'paired_probe_counts':paired,
             'runs':records,'previous_600_update_runs':previous,'accuracy_by_start':by_start}
    (args.results_root/'long_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Longer-budget ordered lookup results','',
           'Fresh seed-17 runs on the unchanged ordered variable-query dataset. Each run uses 3,000 updates and an extended cosine decay horizon, with 30 warmup updates. Validation answer/EOS CE selects checkpoints. Parameters and compute remain unmatched.','',
           '| Model | Train | Held-out ordered | Same maps shuffled | New query on training maps | Lookup ≥90% | Selected step | Training seconds |',
           '|---|---:|---:|---:|---:|---|---:|---:|']
    for f in FAMILIES:
        r=records[f]
        values=[metric(r,s)['exact_match'] for s in ('train','test_id','test_shuffled','train_new_query')]
        lines.append('| '+NAMES[f]+' | '+' | '.join(f'{v:.1%}' for v in values)+
                     f" | {'Pass' if gates[f] else 'Fail'} | {r['checkpoint']['training_counters']['step']} | {r['training']['training_seconds']:.1f} |")
    lines.extend(['','Held-out ordered and shuffled evaluations use the same 128 unseen maps and queries. Changed-query probes reuse 256 training maps. All exact matches require unrestricted greedy answers followed by EOS.','',
                  '## Earlier 600-update comparison','',
                  '| Model | Earlier held-out ordered | Current held-out ordered | Best validation CE | Final validation CE |',
                  '|---|---:|---:|---:|---:|'])
    for f in FAMILIES:
        lines.append(f"| {NAMES[f]} | {metric(previous[f],'test_id')['exact_match']:.1%} | {metric(records[f],'test_id')['exact_match']:.1%} | {audits[f]['selected_validation_ce']:.6g} | {audits[f]['last_validation_ce']:.6g} |")
    lines.extend(['','These are separately initialized runs with the same seed and data. Changing the decay horizon changes learning rates before update 600; the experiment does not isolate the effect of additional updates. The 3,000-update runs see 96,000 examples, 5,184,000 real input tokens, and 192,000 supervised tokens each, exactly five times the previous exposure. Recorded training source hashes, tokenizer, model parameter counts, dataset hashes, and all learning rates were checked.','',
                  '## Paired held-out order intervention','',
                  '| Model | Both correct | Ordered only | Shuffled only | Neither |',
                  '|---|---:|---:|---:|---:|'])
    for f in FAMILIES:
        c=paired[f]['test_shuffled']
        lines.append('| '+NAMES[f]+' | '+' | '.join(str(c[k]) for k in ('both_correct','original_only_correct','probe_only_correct','neither_correct'))+' |')
    lines.extend(['','## Changed query on familiar maps','',
                  '| Model | Correct new answer | Outputs original answer with EOS |',
                  '|---|---:|---:|'])
    for f in FAMILIES:
        lines.append(f"| {NAMES[f]} | {metric(records[f],'train_new_query')['correct']}/256 | {paired[f]['train_new_query']['probe_outputs_original_answer']}/256 |")
    lines.extend(['','## Held-out accuracy by query start','',
                  '| Start | Transformer | Recurrent depth | CTM |','|---|---:|---:|---:|'])
    for node in 'ABCDEFGH':
        lines.append('| '+node+' | '+' | '.join(
            f"{by_start[f]['test_id'][node]['correct']}/{by_start[f]['test_id'][node]['examples']}"
            for f in FAMILIES)+' |')
    lines.extend(['','Small per-start counts are descriptive; they do not establish subgroup differences. Full train/validation/held-out breakdowns are in `long_summary.json`.'])
    lines.extend(['','![Validation curves and held-out transfer](learning_curves.png)','',
                  'The curves show validation loss at 100-update intervals. Dashed lines denote historical 600-update schedules; solid lines denote the new 3,000-update schedules. Timings exclude validation/checkpoint/evaluation work and are implementation-specific.','',
                  'See [protocol and interpretation](../../ORDERED_LONG.md), [pre-run plan](PLAN_BEFORE_RUNS.md), and per-example `.eval.json` records. These development results do not establish a model ranking or multi-step reasoning.'])
    (args.results_root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors={'transformer':'#2369bd','recurrent_depth':'#d87818','ctm':'#28944a'}
    for f in FAMILIES:
        for rows,style,label in [(curves[f],'-',NAMES[f]+' (3,000)'),(old_curves[f],'--',NAMES[f]+' (600)')]:
            axes[0].plot([r['step'] for r in rows],[max(r['validation']['loss'],1e-9) for r in rows],
                         linestyle=style,color=colors[f],label=label)
    axes[0].set(title='Validation loss: distinct training schedules',xlabel='Optimizer updates',ylabel='Answer/EOS cross-entropy',yscale='log')
    axes[0].set_yticks([0.1,0.2,0.5,1.0],labels=['0.1','0.2','0.5','1.0'])
    axes[0].grid(alpha=0.25);axes[0].legend(fontsize=8)
    x=np.arange(len(FAMILIES));width=0.25
    for offset,source,split,label in [(-1,previous,'test_id','Earlier ordered (600)'),(0,records,'test_id','Ordered (3,000)'),(1,records,'test_shuffled','Shuffled (3,000)')]:
        axes[1].bar(x+offset*width,[100*metric(source[f],split)['exact_match'] for f in FAMILIES],width,label=label)
    axes[1].set(title='Held-out accuracy and order transfer',ylabel='Exact match (%)',xticks=x,xticklabels=[NAMES[f] for f in FAMILIES],ylim=(0,110))
    axes[1].axhline(90,color='gray',linestyle=':',label='Ordered lookup gate')
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=0.2)
    fig.suptitle('Ordered variable-query lookup, seed 17; parameters and compute unmatched')
    fig.savefig(args.results_root/'learning_curves.png',dpi=180)
    fig.savefig(args.results_root/'learning_curves.pdf');plt.close(fig)
    print('\n'.join(lines[:12]))


if __name__=='__main__':
    main()
