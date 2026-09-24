"""Report the paired training-presentation intervention at fixed update 3000."""
import json,statistics
from pathlib import Path
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/presentation_control_v1')
FAMILIES=('ctm','transformer','recurrent_depth')
CONDITIONS=('ordered','shuffled')
SPLITS=('validation','validation_shuffled')


def read(path):return json.loads(Path(path).read_text())


def mean_sd(values):
    assert len(values)==3
    return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'values':values,'seeds':[23,29,31]}


def main():
    frozen=read(ROOT/'checkpoints.json');evaluations={}
    for t in frozen['models']:
        path=ROOT/f"{t['cell']}.evaluation.json";e=read(path)
        assert e['complete'] and e['no_test_evaluation'] and e['selected']==t
        assert e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
        assert e['checkpoint']['sha256']==t['checkpoint_sha256']==file_hash(t['checkpoint'])
        for p,h in e['source_sha256'].items():assert file_hash(p)==h,p
        for split in SPLITS:
            m=e['results'][split]['metrics'];pred=m['predictions']
            assert len(pred)==128 and len({p['id'] for p in pred})==128 and m['target_tokens']==256
            assert m['generation']['correct']==sum(p['correct'] for p in pred)
            assert m['generation']['exact_match']==m['generation']['correct']/128
        evaluations[t['cell']]=e
    assert len(evaluations)==18
    aggregate={};secondary={};contrasts={}
    for family in FAMILIES:
        aggregate[family]={};secondary[family]={}
        for condition in CONDITIONS:
            rows=[evaluations[f'{family}_{condition}_seed{s}'] for s in (23,29,31)]
            aggregate[family][condition]={split:{
                'accuracy':mean_sd([e['results'][split]['metrics']['generation']['exact_match'] for e in rows]),
                'ce':mean_sd([e['results'][split]['metrics']['loss'] for e in rows])} for split in SPLITS}
            aggregate[family][condition]['training_minutes']=mean_sd([e['selected']['training_seconds']/60 for e in rows])
            secondary[family][condition]={
                'ordered_validation_accuracy':mean_sd([e['selected']['secondary_best_ordered_validation']['generation']['exact_match'] for e in rows]),
                'ordered_validation_ce':mean_sd([e['selected']['secondary_best_ordered_validation']['ce'] for e in rows])}
        for split in SPLITS:
            accuracy=[];ce=[];map_counts=[]
            for seed in (23,29,31):
                a,b=[evaluations[f'{family}_{c}_seed{seed}']['results'][split]['metrics'] for c in ('shuffled','ordered')]
                pa,pb=a['predictions'],b['predictions']
                assert [(p['id'],p['answer']) for p in pa]==[(p['id'],p['answer']) for p in pb]
                accuracy.append(a['generation']['exact_match']-b['generation']['exact_match']);ce.append(a['loss']-b['loss'])
                map_counts.append({'seed':seed,'shuffled_training_only_correct':sum(x['correct'] and not y['correct'] for x,y in zip(pa,pb)),
                    'ordered_training_only_correct':sum(y['correct'] and not x['correct'] for x,y in zip(pa,pb))})
            contrasts[f'{family}_{split}']={'shuffled_minus_ordered_accuracy':mean_sd(accuracy),
                'shuffled_minus_ordered_ce':mean_sd(ce),'paired_map_counts':map_counts}
    report={'complete':True,'primary_checkpoint_step':3000,'aggregate':aggregate,'secondary_selected_ordered_validation':secondary,
        'paired_training_condition_differences':contrasts,'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),
        'evaluations':{cell:{'path':str(ROOT/f'{cell}.evaluation.json'),'sha256':file_hash(ROOT/f'{cell}.evaluation.json')} for cell in evaluations},
        'analysis_source_sha256':file_hash('scripts/summarize_presentation_control.py'),'no_test_evaluation':True,
        'scope':'Development at fixed update3000 on128 existing validation maps; three paired training seeds, one fixed presentation generator realization; unequal family compute.'}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Paired training-presentation control','',
        'All nine shuffled-training runs completed. Nine ordered-training runs are reused unchanged. Primary results use the checkpoint at exactly3,000 updates for every run, with family recipes/readouts fixed. Evaluation uses the same128 existing validation maps in ordered and shuffled presentations. No test set was evaluated.','',
        '## Primary: fixed-update validation accuracy','',
        '| Family | Training presentation | Ordered evaluation, mean ± SD | Shuffled evaluation, mean ± SD | Training min/run |',
        '|---|---|---:|---:|---:|']
    for family in FAMILIES:
        for condition in CONDITIONS:
            a=aggregate[family][condition];ordered,shuffled=[a[s]['accuracy'] for s in SPLITS]
            lines.append(f"| {family} | {condition} | {100*ordered['mean']:.2f}% ± {100*ordered['sample_sd']:.2f} points | {100*shuffled['mean']:.2f}% ± {100*shuffled['sample_sd']:.2f} points | {a['training_minutes']['mean']:.1f} |")
    lines+=['','SD is sample standard deviation across seeds23/29/31, not a confidence interval. These are development seeds on shared validation maps, not new confirmatory replications. Exact accuracy requires unrestricted correct answer+EOS generation.','',
        '## Primary: fixed-update answer/EOS CE','',
        '| Family | Training presentation | Ordered CE, mean ± SD | Shuffled CE, mean ± SD |','|---|---|---:|---:|']
    for family in FAMILIES:
        for condition in CONDITIONS:
            a,b=[aggregate[family][condition][s]['ce'] for s in SPLITS]
            lines.append(f"| {family} | {condition} | {a['mean']:.6g} ± {a['sample_sd']:.3g} | {b['mean']:.6g} ± {b['sample_sd']:.3g} |")
    lines+=['','CE is token-weighted and teacher-forced over answer/EOS.','',
        '## Paired effects of shuffled training','',
        '| Family and evaluation presentation | Accuracy difference, mean ± SD (points) | CE difference, mean ± SD |',
        '|---|---:|---:|']
    for key,d in contrasts.items():
        a,b=d['shuffled_minus_ordered_accuracy'],d['shuffled_minus_ordered_ce']
        lines.append(f"| {key} | {100*a['mean']:+.2f} ± {100*a['sample_sd']:.2f} | {b['mean']:+.6g} ± {b['sample_sd']:.3g} |")
    lines+=['','Each difference pairs the same family, initialization/data-order seed, semantic examples, final-update budget, readout and evaluation presentation. Positive accuracy differences favor shuffled training. No significance claim is implied by three seeds.','',
        '![Fixed-update paired presentation results](presentation_accuracy.png)','',
        '## Every primary endpoint','',
        '| Family | Training | Seed | Ordered accuracy | Shuffled accuracy | Ordered CE | Shuffled CE | Train min | Peak GiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        for condition in CONDITIONS:
            for seed in (23,29,31):
                e=evaluations[f'{family}_{condition}_seed{seed}'];t=e['selected'];a,b=[e['results'][s]['metrics'] for s in SPLITS]
                lines.append(f"| {family} | {condition} | {seed} | {a['generation']['exact_match']:.2%} | {b['generation']['exact_match']:.2%} | {a['loss']:.6g} | {b['loss']:.6g} | {t['training_seconds']/60:.1f} | {t['peak_allocated_bytes']/2**30:.3f} |")
    lines+=['','## Secondary: ordered-validation-selected checkpoints','',
        'Both training conditions retained the existing ordered-validation selection procedure. These secondary scores are selected on that same validation set and do not replace the fixed-update primary endpoints. Shuffled-validation outcomes were never used to select a checkpoint.','',
        '| Family | Training | Selected ordered accuracy, mean ± SD | Selected ordered CE, mean ± SD |','|---|---|---:|---:|']
    for family in FAMILIES:
        for condition in CONDITIONS:
            s=secondary[family][condition];a,b=s['ordered_validation_accuracy'],s['ordered_validation_ce']
            lines.append(f"| {family} | {condition} | {100*a['mean']:.2f}% ± {100*a['sample_sd']:.2f} points | {b['mean']:.6g} ± {b['sample_sd']:.3g} |")
    lines+=['','![Original ordered-validation trajectories, for context](ordered_validation_curves.png)','',
        '## Controls and interpretation limits','',
        '- Training file order, semantic maps, starts, queries, answers, masked targets, sequence lengths and example/token budgets are paired. Only training edge presentation changes. All2,048 training maps and128 validation maps come from the existing development splits; no semantic training/validation overlap is introduced.',
        '- One noncanonical presentation is fixed per map, shared across families/seeds and repeated each epoch. This is not online augmentation or an estimate over multiple shuffle-generator realizations. Ordered and shuffled evaluations reuse the same maps.',
        '- CTM keeps uniform supervision/LR0.0003/confidence readout; Transformer keeps final CE/LR0.0003/final readout; recurrent depth keeps final CE/LR0.001/confidence readout. T16 is fixed for CTM/recurrent. Recipes were selected under ordered training and are not retuned here.',
        '- Every run receives96,000 example presentations,5,184,000 input tokens and192,000 answer/EOS labels. Approximate parameter matching and equal exposure do not equalize family compute or prior tuning. Full-budget training-loop wall times are reported, not measured FLOPs.',
        '- Five new data/replay checks passed, including exact archived CTM/Transformer GPU replay; the previous recurrent control verified its archived final-CE replay. Both source versions are retained. All54,000 updates are audited, including27,000 newly trained updates.',
        '- Final checkpoint files retain legacy final-readout metadata; the frozen registry/evaluation explicitly applies the declared family readout. Final checkpoint identities were frozen before paired evaluation.',
        '- These results address one-hop lookup at existing recipes and budgets. Reliable retrieval, new confirmation, optimizer/schedule controls, and compute accounting remain prerequisites for broader claims. No completed test set was reopened.','',
        'See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json`, per-example evaluation files and source/checkpoint manifests.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharey=True,layout='constrained')
    for ax,family in zip(axes,FAMILIES):
        for i,condition in enumerate(CONDITIONS):
            for j,split in enumerate(SPLITS):
                x=j+(i-.5)*.3;a=aggregate[family][condition][split]['accuracy'];color=('tab:blue','tab:orange')[i]
                ax.errorbar(x,100*a['mean'],yerr=100*a['sample_sd'],fmt='o',color=color,capsize=4,
                            label=f'{condition} training' if j==0 else None)
                ax.scatter([x-.04,x,x+.04],[100*v for v in a['values']],color=color,s=18,alpha=.7)
        ax.set(title=family,xticks=[0,1],xticklabels=['Ordered eval','Shuffled eval'],ylabel='Validation answer + EOS (%)',ylim=(0,105))
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed update3000: paired development seeds23/29/31, mean ± sample SD')
    for ext in ('png','pdf'):fig.savefig(ROOT/f'presentation_accuracy.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for ax,family in zip(axes,FAMILIES):
        for i,condition in enumerate(CONDITIONS):
            for seed in (23,29,31):
                t=evaluations[f'{family}_{condition}_seed{seed}']['selected'];path=Path(t['run_directory'])/'metrics.jsonl'
                assert file_hash(path)==t['metrics_sha256']
                rows=[r for r in map(json.loads,path.read_text().splitlines()) if 'validation' in r]
                ax.plot([r['step'] for r in rows],[r['validation']['loss'] for r in rows],color=('tab:blue','tab:orange')[i],
                        linestyle={23:'-',29:'--',31:':'}[seed],label=f'{condition}/{seed}')
        ax.set(title=family,xlabel='Optimizer updates',ylabel='Ordered-validation answer/EOS CE');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Secondary diagnostic: unchanged ordered-validation checks during training')
    for ext in ('png','pdf'):fig.savefig(ROOT/f'ordered_validation_curves.{ext}',dpi=180)
    plt.close(fig)
    print('\n'.join(lines[:15]),flush=True)

if __name__=='__main__':main()
