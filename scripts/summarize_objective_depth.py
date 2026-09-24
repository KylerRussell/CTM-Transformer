"""Report the uniform/dynamic by T4/T16 development interaction without test-set selection."""
import json
import math
from pathlib import Path
import numpy as np
from ctm_transformer.experiment import file_hash
from ctm_transformer.algorithmic import load_algorithmic_split
from ctm_transformer.query_analysis import summarize_grid

ROOT=Path('research/results/objective_depth_v1')
OLD=Path('research/results/temporal_ablation_v1')
CELLS=('uniform_t4','dynamic_t4','uniform_t16','dynamic_t16')


def main():
    def read(p):return json.loads(p.read_text())
    def result_root(cell):return OLD if cell=='uniform_t4' else ROOT
    def stem(cell):return 'uniform' if cell=='uniform_t4' else cell
    def run_root(cell):return Path('research/runs/temporal_ablation_v1/seed17/uniform') if cell=='uniform_t4' else Path('research/runs/objective_depth_v1/seed17')/cell
    def preds(evaluation,split):return next(iter(evaluation['results'][split]['depths'].values()))['predictions']
    records={};curves={};grids={};confidence={};audits={};runs={}
    expected={'examples_seen':96000,'tokens_seen':5184000,'supervised_tokens_seen':192000}
    for cell in CELLS:
        directory=result_root(cell);prefix=stem(cell)
        r=read(directory/f'{prefix}.summary.json');records[cell]=r
        run=read(run_root(cell)/'research_run.json');runs[cell]=run
        depth=int(cell.split('_t')[1]);mode='ramp_mono' if cell.startswith('uniform') else 'dynamic_aggregate'
        assert r['complete'] and r['training']['complete'] and r['training']['steps']==3000
        assert run['effective_config']['max_thought_steps']==depth and run['effective_config']['history_len']==8
        assert run['effective_config']['temporal_loss_type']==mode
        assert run['seed']==17 and run['data_seed']==18 and run['depth_seed']==19
        for k,v in expected.items():assert r['training'][k]==v
        assert r['training']['token_block_applications']==5184000*2*depth
        rows=[json.loads(l) for l in (run_root(cell)/'metrics.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows]==list(range(1,3001))
        for row in rows:
            step=row['step'];lr=1e-3*(step/30 if step<=30 else .1+.45*(1+math.cos(math.pi*(step-30)/2970)))
            assert math.isclose(row['lr'],lr,rel_tol=1e-12)
            assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
            losses=row['per_tick_supervised_ce'];assert len(losses)==depth and all(math.isfinite(v) for v in losses)
            if cell.startswith('uniform'):assert math.isclose(row['loss'],sum(losses)/depth,rel_tol=2e-5,abs_tol=2e-6)
        validation=[row for row in rows if 'validation' in row];assert len(validation)==30
        best=min(validation,key=lambda row:row['validation']['loss'])
        assert best['step']==r['checkpoint']['training_counters']['step']
        assert best['validation']['loss']==r['training']['best_validation_loss']
        assert file_hash(r['checkpoint']['path'])==r['checkpoint']['sha256']
        assert file_hash(run['config_identity']['path'])==run['config_identity']['sha256']
        evaluations={suffix:read(directory/f'{prefix}.{suffix}.json') for suffix in ('eval','ticks.eval','all_queries.eval')}
        for ev in evaluations.values():
            assert ev['complete'] and ev['checkpoint']['sha256']==r['checkpoint']['sha256']
            assert all('test' not in split for split in ev['results'])
        assert evaluations['ticks.eval']['results']['validation']['depths'][str(depth)]['predictions']==preds(evaluations['eval'],'validation')
        expanded=[]
        for node in 'ABCDEFGH':
            split=f'validation_start_{node}';ds=load_algorithmic_split('research/data/query_diagnostic_v1','pointer',split,128)
            generated=preds(evaluations['all_queries.eval'],split)
            assert len(generated)==len(ds.records)
            for row,pred in zip(ds.records,generated):
                assert row['id']==pred['id'] and row['answer']==pred['answer']
                expanded.append({**row,**pred})
        originals={row['id']:row for row in preds(evaluations['eval'],'validation')}
        for row in expanded:
            if row['start']==row['original_start']:assert all(row[k]==originals[row['id']][k] for k in ('answer','prediction','correct','terminated'))
        grids[cell]=summarize_grid(expanded)
        conf=read(ROOT/f'{cell}.confidence.json');assert conf['complete'] and conf['checkpoint']['sha256']==r['checkpoint']['sha256']
        assert conf['metrics']['examples']==128 and set(originals)=={p['id'] for p in conf['predictions']}
        for pred in conf['predictions']:assert pred['answer']==originals[pred['id']]['answer']
        confidence[cell]=conf
        curves[cell]=validation
        audits[cell]={'all_updates_and_budgets_verified':True,'selected_validation_step':best['step'],
                     'checkpoint_sha256':r['checkpoint']['sha256'],'reused_historical_cell':cell=='uniform_t4'}
    reference=runs['uniform_t4']
    for cell,run in runs.items():
        assert run['parameters']==reference['parameters'] and run['tokenizer_sha256']==reference['tokenizer_sha256']
        assert run['data']==reference['data']
        assert run['code_sha256']['ctm_transformer/model.py']==reference['code_sha256']['ctm_transformer/model.py']
        changed={k:[v,run['effective_config'][k]] for k,v in reference['effective_config'].items() if v!=run['effective_config'][k]}
        assert set(changed)<={'temporal_loss_type','max_thought_steps','checkpoint_dir','device'}
        audits[cell]['effective_config_changes_from_uniform_t4']=changed
    def ce(c):return records[c]['training']['best_validation_loss']
    def accuracy(c):return records[c]['metrics']['validation']['exact_match']
    contrasts={}
    for depth in (4,16):
        u,d=f'uniform_t{depth}',f'dynamic_t{depth}'
        contrasts[str(depth)]={'ce_dynamic_minus_uniform':ce(d)-ce(u),
            'final_accuracy_dynamic_minus_uniform':accuracy(d)-accuracy(u),
            'confidence_accuracy_dynamic_minus_uniform':confidence[d]['metrics']['exact_match']-confidence[u]['metrics']['exact_match'],
            'all_start_accuracy_dynamic_minus_uniform':grids[d]['overall']['exact_match']-grids[u]['overall']['exact_match']}
    interaction={k:contrasts['16'][k]-contrasts['4'][k] for k in contrasts['4']}
    report={'complete':True,'purpose':'single-seed descriptive objective-by-depth interaction; validation-only; equal exposure, unequal compute',
            'within_depth_contrasts':contrasts,'interaction_t16_minus_t4':interaction,'runs':records,
            'all_start_validation':grids,'confidence_readouts':confidence,'audits':audits,
            'analysis_source_sha256':{p:file_hash(p) for p in ('scripts/summarize_objective_depth.py','ctm_transformer/query_analysis.py')}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Uniform versus dynamic aggregation across depth','',
        'The subsequent user clarification and policy-matched final-checkpoint follow-up are documented in [READOUT_FOLLOWUP.md](READOUT_FOLLOWUP.md). Read these together: final-tick CE selected substantially worse dynamic checkpoints for confidence inference.','',
        'One seed, fixed H=8, 3,000 updates and equal example/token exposure per cell. Uniform T4 is reused from the archived temporal ablation; three other cells start fresh. Core model code is identical. The shared runner additionally admits corrected dynamic loss. No held-out test split was evaluated.','',
        '| Cell | Best final validation CE | Final-tick validation accuracy | Confidence readout accuracy | All-start accuracy | Maps: all 8 correct | Selected step | Train min | Peak GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in CELLS:
        r=records[c];g=grids[c]
        lines.append(f"| {c} | {ce(c):.6f} | {accuracy(c):.1%} | {confidence[c]['metrics']['exact_match']:.1%} | {g['overall']['exact_match']:.1%} | {g['all_queries_correct_maps']}/128 | {r['checkpoint']['training_counters']['step']} | {r['training']['training_seconds']/60:.1f} | {r['training']['peak_allocated_bytes']/2**30:.3f} |")
    lines.extend(['','Final-tick CE selects every checkpoint. Confidence readout chooses the minimum-FP32-entropy tick independently for each generated token using no labels, runs all trained ticks, and is not early stopping. Both accuracies require answer plus EOS. Confidence readout is secondary; checkpoints were not optimized for it.','',
        '## Objective-by-depth contrasts','',
        '| Contrast | T4: dynamic − uniform | T16: dynamic − uniform | Interaction: T16 contrast − T4 contrast |',
        '|---|---:|---:|---:|'])
    for key,label in [('ce_dynamic_minus_uniform','Final validation CE'),('final_accuracy_dynamic_minus_uniform','Final accuracy (percentage points)'),('confidence_accuracy_dynamic_minus_uniform','Confidence accuracy (percentage points)'),('all_start_accuracy_dynamic_minus_uniform','All-start accuracy (percentage points)')]:
        scale=1 if key.startswith('ce_') else 100
        lines.append(f"| {label} | {scale*contrasts['4'][key]:+.4f} | {scale*contrasts['16'][key]:+.4f} | {scale*interaction[key]:+.4f} |")
    lines.extend(['','Lower CE is better; higher accuracy is better. These differences are descriptive, without a seed-variance estimate or significance test. Repeated queries share maps, and validation influenced checkpoint selection.','',
        '## Fixed-checkpoint validation depth readouts','',
        '| Cell | T1 | T2 | T4 | T8 | T16 |','|---|---:|---:|---:|---:|---:|'])
    for c in CELLS:
        tick=read(result_root(c)/f'{stem(c)}.ticks.eval.json')['results']['validation']['depths']
        lines.append('| '+c+' | '+' | '.join(f"{tick[str(t)]['exact_match']:.1%}" if str(t) in tick else '—' for t in (1,2,4,8,16))+' |')
    lines.extend(['','Depth readouts do not reselect checkpoints or inference policies. T16 has four times the nominal layer applications of T4 and fills/evicts H=8 history; it is not an equal-compute intervention. Each run consumes 96,000 examples, 5,184,000 real input tokens, and 192,000 supervised labels. Reported training time excludes validation/evaluation/checkpoint work.','',
        '![Validation curves and objective-by-depth comparison](learning_curves.png)','',
        'See [protocol and interpretation](../../OBJECTIVE_DEPTH.md), [pre-run plan](PLAN_BEFORE_RUNS.md), profiles, per-example predictions, and `summary.json` audits. This study addresses the reported depth sensitivity on this task; it is not an architecture comparison or a complete replication of an unspecified historical recipe.'])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for c in CELLS:
        color='#2369bd' if c.startswith('uniform') else '#d87818';style='--' if c.endswith('t4') else '-'
        axes[0].plot([r['step'] for r in curves[c]],[r['validation']['loss'] for r in curves[c]],color=color,linestyle=style,label=c)
    axes[0].set(title='Common final-trained-tick validation CE',xlabel='Optimizer updates',ylabel='Answer/EOS CE')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    x=np.arange(4)
    axes[1].bar(x-.17,[100*accuracy(c) for c in CELLS],.34,label='Final tick')
    axes[1].bar(x+.17,[100*confidence[c]['metrics']['exact_match'] for c in CELLS],.34,label='Minimum-entropy readout')
    axes[1].set(title='Same selected checkpoints, two readouts',ylabel='Validation exact match (%)',xticks=x,xticklabels=CELLS,ylim=(0,105))
    axes[1].tick_params(axis='x',labelrotation=15);axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Uniform/dynamic × T4/T16; seed 17; H8 fixed; equal exposure, unequal compute')
    fig.savefig(ROOT/'learning_curves.png',dpi=180);fig.savefig(ROOT/'learning_curves.pdf');plt.close(fig)
    print('\n'.join(lines[:13]))

if __name__=='__main__':main()
