"""Audit and summarize the validation-only within-CTM temporal-supervision ablation."""
import json
import math
from pathlib import Path
import numpy as np
from ctm_transformer.experiment import file_hash
from ctm_transformer.query_analysis import summarize_grid
from ctm_transformer.algorithmic import load_algorithmic_split

ROOT=Path('research/results/temporal_ablation_v1')
RUNS=Path('research/runs/temporal_ablation_v1/seed17')
RECIPES=('final','uniform','late')
NAMES={'final':'Final-only','uniform':'Uniform','late':'Later-weighted'}


def main():
    def read(path):return json.loads(path.read_text())
    def preds(evaluation,split):return next(iter(evaluation['results'][split]['depths'].values()))['predictions']
    records={};runs={};curves={};grids={};audits={}
    expected_counts={'examples_seen':96000,'tokens_seen':5184000,'supervised_tokens_seen':192000}
    expected_weights={'final':[0,0,0,1],'uniform':[.25]*4,'late':[0,1/6,1/3,1/2]}
    for recipe in RECIPES:
        record=read(ROOT/f'{recipe}.summary.json');records[recipe]=record
        run=read(RUNS/recipe/'research_run.json');runs[recipe]=run
        assert record['complete'] and record['training']['complete'] and record['training']['steps']==3000
        for k,v in expected_counts.items():assert record['training'][k]==v
        assert run['seed']==17 and run['data_seed']==18 and run['depth_seed']==19
        assert np.allclose(run['training_objective']['direct_ce_tick_weights_at_max_depth'],expected_weights[recipe])
        assert file_hash(run['config_identity']['path'])==run['config_identity']['sha256']
        rows=[json.loads(l) for l in (RUNS/recipe/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,3001))
        max_loss_error=0.
        for row in rows:
            step=row['step'];lr=1e-3*(step/30 if step<=30 else .1+.45*(1+math.cos(math.pi*(step-30)/2970)))
            assert math.isclose(row['lr'],lr,rel_tol=1e-12)
            assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
            assert len(row['per_tick_supervised_ce'])==4
            weighted=sum(w*c for w,c in zip(expected_weights[recipe],row['per_tick_supervised_ce']))
            error=abs(row['loss']-weighted);max_loss_error=max(error,max_loss_error)
            assert math.isclose(row['loss'],weighted,rel_tol=2e-5,abs_tol=2e-6)
        validation=[r for r in rows if 'validation' in r];assert len(validation)==30
        best=min(validation,key=lambda r:r['validation']['loss'])
        assert best['step']==record['checkpoint']['training_counters']['step']
        assert best['validation']['loss']==record['training']['best_validation_loss']
        assert file_hash(record['checkpoint']['path'])==record['checkpoint']['sha256']
        for suffix in ('eval','ticks.eval','all_queries.eval'):
            evaluation=read(ROOT/f'{recipe}.{suffix}.json')
            assert evaluation['complete'] and evaluation['checkpoint']['sha256']==record['checkpoint']['sha256']
            assert all('test' not in split for split in evaluation['results'])
            for p,h in evaluation['source_sha256'].items():assert file_hash(p)==h
        original=read(ROOT/f'{recipe}.eval.json')
        ticks=read(ROOT/f'{recipe}.ticks.eval.json')
        assert ticks['results']['validation']['depths']['4']['predictions']==preds(original,'validation')
        grid_eval=read(ROOT/f'{recipe}.all_queries.eval.json');expanded=[]
        for node in 'ABCDEFGH':
            split=f'validation_start_{node}'
            ds=load_algorithmic_split('research/data/query_diagnostic_v1','pointer',split,128)
            generated=preds(grid_eval,split)
            assert len(generated)==len(ds.records)
            for row,pred in zip(ds.records,generated):
                assert row['id']==pred['id'] and row['answer']==pred['answer']
                expanded.append({**row,**pred})
        prior={r['id']:r for r in preds(original,'validation')}
        for row in expanded:
            if row['start']==row['original_start']:
                assert all(row[k]==prior[row['id']][k] for k in ('answer','prediction','correct','terminated'))
        grids[recipe]=summarize_grid(expanded);curves[recipe]=rows
        audits[recipe]={'maximum_logged_weighted_loss_error':max_loss_error,'best_validation_step':best['step'],
                       'all_lr_and_budget_checks_passed':True,'all_query_and_tick4_original_outputs_identical':True}
    reference=runs['final']
    for recipe in RECIPES:
        run=runs[recipe]
        assert run['parameters']==reference['parameters'] and run['tokenizer_sha256']==reference['tokenizer_sha256']
        assert run['code_sha256']==reference['code_sha256']
        assert records[recipe]['dataset_manifest_sha256']==records['final']['dataset_manifest_sha256']
        changed={k:[v,run['effective_config'][k]] for k,v in reference['effective_config'].items() if v!=run['effective_config'][k]}
        assert set(changed)<={'temporal_loss_type','tick_ramp_start','tick_ramp_end','checkpoint_dir','device'}
        audits[recipe]['config_differences_from_final']=changed
    ranking=sorted(RECIPES,key=lambda r:(records[r]['training']['best_validation_loss'],RECIPES.index(r)))
    old=read(Path('research/results/ordered_long_v1/ordered_ctm_seed17.eval.json'))
    new=read(ROOT/'final.eval.json')
    original_agreement={}
    for split in ('train','validation','train_probe','train_new_query'):
        before={r['id']:r for r in preds(old,split)};after=preds(new,split)
        original_agreement[split]={'examples':len(after),'identical_predictions':sum(all(r[k]==before[r['id']][k] for k in ('prediction','correct','terminated','answer')) for r in after)}
    old_metrics=[json.loads(l) for l in Path('research/runs/ordered_long_v1/seed17/ctm/metrics.jsonl').read_text().splitlines()]
    max_prior_loss_delta=max(abs(a['loss']-b['loss']) for a,b in zip(curves['final'],old_metrics))
    selection={'rule':'minimum original-validation final-tick answer/EOS CE; ties final,uniform,late',
        'ranking':ranking,'selected_recipe':ranking[0],'scores':{r:records[r]['training']['best_validation_loss'] for r in RECIPES},
        'input_summary_sha256':{r:file_hash(ROOT/f'{r}.summary.json') for r in RECIPES},'no_heldout_test_evaluation':True}
    selection_path=ROOT/'selection.json'
    if selection_path.exists():assert read(selection_path)==selection
    else:selection_path.write_text(json.dumps(selection,indent=2)+'\n')
    report={'complete':True,'selection':selection,'runs':records,'all_start_validation':grids,'audits':audits,
        'historical_final_control':{'prediction_agreement':original_agreement,
            'maximum_training_loss_delta':max_prior_loss_delta,'previous_selected_step':old['checkpoint']['training_counters']['step']},
        'analysis_source_sha256':{p:file_hash(p) for p in ('scripts/summarize_temporal_ablation.py','ctm_transformer/query_analysis.py')}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# CTM temporal-supervision ablation results','',
        'Three fresh seed-17 runs, each with four ticks, history length eight, and 3,000 updates on identical ordered one-hop training data. Only temporal CE weights vary. No held-out test split was evaluated.','',
        '| Recipe | Best final validation CE | Original validation accuracy | All-start validation accuracy | Validation maps: all 8 correct | Selected step | Train seconds | Peak GPU GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in RECIPES:
        record=records[r];g=grids[r]
        lines.append(f"| {NAMES[r]} | {record['training']['best_validation_loss']:.6f} | {record['metrics']['validation']['exact_match']:.1%} | {g['overall']['exact_match']:.1%} | {g['all_queries_correct_maps']}/128 | {record['checkpoint']['training_counters']['step']} | {record['training']['training_seconds']:.1f} | {record['training']['peak_allocated_bytes']/2**30:.3f} |")
    lines.extend(['',f"**Selected by the recorded validation-CE rule: {NAMES[ranking[0]]}.** Ranking: "+' → '.join(NAMES[r] for r in ranking)+'.','',
        'Recipe selection uses original-validation final-tick CE, not auxiliary training loss, exact-match accuracy, or a preferred inference depth. Expanded validation queries share the same 128 maps and are secondary development measurements. Query outcomes within maps are correlated.','',
        '## Training and changed-query probes','',
        '| Recipe | Training accuracy | Training-map original query | Training-map changed query |',
        '|---|---:|---:|---:|'])
    for r in RECIPES:
        lines.append('| '+NAMES[r]+' | '+' | '.join(f"{records[r]['metrics'][s]['exact_match']:.1%}" for s in ('train','train_probe','train_new_query'))+' |')
    lines.extend(['','## Greedy validation accuracy at different thought depths','',
        '| Recipe | T=1 | T=2 | T=3 | T=4 |','|---|---:|---:|---:|---:|'])
    for r in RECIPES:
        lines.append('| '+NAMES[r]+' | '+' | '.join(f"{records[r]['validation_ticks'][str(t)]['exact_match']:.1%}" for t in range(1,5))+' |')
    lines.extend(['','These scores require the correct answer and EOS. They differ from the earlier attention audit’s answer-token-only readouts. The checkpoint is fixed by T=4 validation CE; shorter depths are diagnostic and are not selected.','',
        '## All-start validation accuracy','',
        '| Start | Final-only | Uniform | Later-weighted |','|---|---:|---:|---:|'])
    for node in 'ABCDEFGH':lines.append('| '+node+' | '+' | '.join(f"{grids[r]['by_start'][node]['correct']}/128" for r in RECIPES)+' |')
    lines.extend(['','## Reproduction and budgets','',
        f"The fresh final-only control selects update {records['final']['checkpoint']['training_counters']['step']}; the historical control selected {old['checkpoint']['training_counters']['step']}. Maximum absolute training-loss difference over 3,000 updates: {max_prior_loss_delta:.9g}.", '',
        '| Split | Identical predictions / examples |','|---|---:|'])
    for s,v in original_agreement.items():lines.append(f"| {s} | {v['identical_predictions']}/{v['examples']} |")
    lines.extend(['','Every recipe sees 96,000 example presentations, 5,184,000 real input tokens, and 192,000 supervised labels. Weighted objectives reuse those labels at several readouts, adding backward work. Timing excludes evaluation/checkpoint work and is an implementation measurement, not FLOP accounting. Raw logs retain objective loss and each tick’s CE; curves below compare the common final-tick validation metric.','',
        '![Validation and tick curves](learning_curves.png)','',
        'Eleven targeted runner checks passed before training. Post-run audits check all 9,000 updates, learning rates, logged weighted losses, budget counters, source/config/checkpoint hashes, checkpoint selection, and repeated original-query outputs. This is one-seed within-CTM development; it does not rank architectures or establish a general optimum. See the [protocol and interpretation](../../TEMPORAL_ABLATION.md), [pre-run plan](PLAN_BEFORE_RUNS.md), and immutable `selection.json`.'])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors={'final':'#2369bd','uniform':'#d87818','late':'#28944a'}
    for r in RECIPES:
        val=[row for row in curves[r] if 'validation' in row]
        axes[0].plot([row['step'] for row in val],[row['validation']['loss'] for row in val],label=NAMES[r],color=colors[r])
        axes[1].plot(range(1,5),[100*records[r]['validation_ticks'][str(t)]['exact_match'] for t in range(1,5)],marker='o',label=NAMES[r],color=colors[r])
    axes[0].set(title='Common final-tick validation loss',xlabel='Optimizer updates',ylabel='Answer/EOS cross-entropy')
    axes[1].set(title='Fixed-checkpoint validation depth diagnostic',xlabel='Thought depth',ylabel='Exact answer + EOS (%)',xticks=range(1,5),ylim=(0,105))
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('CTM temporal-supervision ablation; seed 17; T=4, H=8; validation-only selection')
    fig.savefig(ROOT/'learning_curves.png',dpi=180);fig.savefig(ROOT/'learning_curves.pdf');plt.close(fig)
    print('\n'.join(lines[:13]))

if __name__=='__main__':main()
