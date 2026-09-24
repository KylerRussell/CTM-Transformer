"""Audit and summarize paired recurrent objectives using validation only."""
import hashlib,json,math,statistics,zipfile
from pathlib import Path
import torch
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/recurrent_temporal_v1')
OBJECTIVES=('final_ce','uniform','dynamic_aggregate')


def read(path):return json.loads(Path(path).read_text())


def mean_sd(values):
    assert len(values)==3
    return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'values':values,'seeds':[23,29,31]}


def audit(entry,registry):
    directory=Path(entry['run_directory']);record=read(directory/'research_run.json');config=record['effective_config']
    summary=read(entry['summary_path']);objective=entry['objective']
    assert summary['complete'] and summary['steps']==3000 and summary['parameters']['total']==525984
    assert (record['seed'],record['data_seed'],record['depth_seed'])==(entry['seed'],entry['seed']+1,entry['seed']+2)
    assert file_hash(entry['config'])==entry['config_sha256']==record['config_identity']['sha256']
    declared=read(entry['config'])['config']
    assert {k for k,v in declared.items() if config[k]!=v}<={'device','data_path','eval_data_path','checkpoint_dir'}
    assert config['learning_rate']==.001 and config['max_thought_steps']==config['train_depth_min']==config['train_depth_max']==16
    assert record['checkpoint_selection']['readouts']==['confidence'] and record['checkpoint_selection']['save_every_validation']
    assert record['training_objective']['type']==objective
    assert record['data']['training']['sha256']==registry['train_sha256']==file_hash(registry['train_path'])
    assert record['data']['evaluation']['sha256']==registry['validation_sha256']==file_hash(registry['validation_path'])
    if entry['reused']:
        reference=entry['reference_control']
        assert file_hash(directory/'research_run.json')==entry['control_run_manifest_sha256']
        assert file_hash(entry['summary_path'])==reference['summary_sha256']
        assert file_hash(directory/'metrics.jsonl')==reference['metrics_sha256']
        assert file_hash(directory/'best.pt')==reference['checkpoint_sha256']
        with zipfile.ZipFile(registry['control_source_archive']) as archive:
            for p,digest in record['code_sha256'].items():assert hashlib.sha256(archive.read(p)).hexdigest()==digest,p
    else:
        assert summary['objective']==objective and summary['worker_source_sha256']==file_hash('scripts/run_recurrent_temporal.py')
        for p,digest in record['code_sha256'].items():assert file_hash(p)==digest,p
    rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,3001))
    blocks=34 if objective=='final_ce' else 49
    validations=[]
    for r in rows:
        step=r['step'];factor=step/30 if step<=30 else .1+.45*(1+math.cos(math.pi*(step-30)/2970))
        assert math.isclose(r['lr'],.001*factor,rel_tol=1e-12)
        assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
        assert r['examples_seen']==32*step and r['tokens_seen']==1728*step and r['supervised_tokens_seen']==64*step
        assert r['thought_steps']==16 and r['block_applications_per_sequence']==blocks
        if objective!='final_ce':assert len(r['per_tick_supervised_ce'])==16 and all(math.isfinite(v) for v in r['per_tick_supervised_ce'])
        if objective=='uniform':assert math.isclose(r['loss'],statistics.mean(r['per_tick_supervised_ce']),abs_tol=1e-6,rel_tol=1e-6)
        if 'validation_readouts' in r:
            assert step%100==0 and (directory/f'step_{step:06d}.pt').exists()
            assert set(r['validation_readouts'])=={'confidence','diagnostics'}
            v=r['validation_readouts']['confidence']
            assert math.isfinite(v['loss']) and v['target_tokens']==256 and len(v['predictions'])==128
            assert v['generation']['correct']==sum(p['correct'] for p in v['predictions'])
            assert v['generation']['exact_match']==v['generation']['correct']/128
            assert r['validation']['readout_policy']=='confidence' and r['validation']['loss']==v['loss']
            validations.append(r)
    assert [r['step'] for r in validations]==list(range(100,3001,100))
    best=min(validations,key=lambda r:(r['validation']['loss'],r['step']))
    assert summary['best_validation_loss']==best['validation']['loss'] and summary['best_readout_policy']=='confidence'
    assert summary['tokens_seen']==5184000 and summary['examples_seen']==96000 and summary['supervised_tokens_seen']==192000
    assert summary['token_block_applications']==summary['padded_token_positions']*blocks
    assert summary['depth_counts']=={'16':3000}
    payload=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    assert payload['step']==best['step'] and payload['readout_policy']=='confidence' and payload['config']==config
    if not entry['reused']:assert payload['training_objective']==record['training_objective']
    assert payload['tokens_seen']==best['step']*1728
    return {'cell':entry['cell'],'objective':objective,'seed':entry['seed'],'reused':entry['reused'],
        'step':best['step'],'readout':'confidence','validation_ce':best['validation']['loss'],
        'validation_generation':best['validation']['generation'],'predictions':best['validation']['predictions'],
        'parameters':summary['parameters'],'training_seconds':summary['training_seconds'],'wall_seconds':summary['wall_seconds'],
        'peak_allocated_bytes':summary['peak_allocated_bytes'],'block_applications_per_sequence':blocks,
        'checkpoint':str(directory/'best.pt'),'checkpoint_sha256':file_hash(directory/'best.pt'),
        'metrics_sha256':file_hash(directory/'metrics.jsonl'),'summary_sha256':file_hash(entry['summary_path']),
        'run_directory':str(directory),'config':entry['config'],'config_sha256':entry['config_sha256']},validations


def main():
    torch.set_num_threads(4)
    registry=read(ROOT/'registry.json');pre=read(ROOT/'pre_run_source.json')
    assert file_hash(ROOT/'pre_run_source.zip')==pre['archive_sha256']
    assert file_hash(registry['control_source_archive'])==registry['control_source_archive_sha256']
    assert file_hash(registry['control_checkpoint_freeze'])==registry['control_checkpoint_freeze_sha256']
    for p,digest in pre['files'].items():assert file_hash(p)==digest,p
    failures=list(ROOT.glob('*.failure.json'))
    if failures:raise RuntimeError(f'Declared failures require explicit reporting before aggregation: {failures}')
    trials=[];curves={}
    for entry in registry['entries']:
        t,curve=audit(entry,registry);trials.append(t);curves[t['cell']]=curve
    # Every matched seed changes only objective/run paths, never architecture or optimizer.
    for seed in registry['seeds']:
        configs=[read(e['config'])['config'] for e in registry['entries'] if e['seed']==seed]
        for other in configs[1:]:
            assert {k for k,v in configs[0].items() if other[k]!=v}<={'checkpoint_dir'}
    aggregate={}
    for objective in OBJECTIVES:
        subset=sorted([t for t in trials if t['objective']==objective],key=lambda t:t['seed'])
        assert [t['seed'] for t in subset]==[23,29,31]
        aggregate[objective]={'validation_accuracy':mean_sd([t['validation_generation']['exact_match'] for t in subset]),
            'validation_ce':mean_sd([t['validation_ce'] for t in subset]),
            'training_minutes':mean_sd([t['training_seconds']/60 for t in subset])}
    contrasts={}
    bycell={t['cell']:t for t in trials}
    for objective in OBJECTIVES[1:]:
        deltas=[];ce_deltas=[]
        for seed in registry['seeds']:
            a,b=bycell[f'{objective}_seed{seed}'],bycell[f'final_ce_seed{seed}']
            assert [(p['id'],p['answer']) for p in a['predictions']]==[(p['id'],p['answer']) for p in b['predictions']]
            deltas.append(a['validation_generation']['exact_match']-b['validation_generation']['exact_match'])
            ce_deltas.append(a['validation_ce']-b['validation_ce'])
        contrasts[objective+'_minus_final_ce']={'accuracy':mean_sd(deltas),'ce':mean_sd(ce_deltas)}
    result={'complete':True,'registry_sha256':file_hash(ROOT/'registry.json'),'trials':trials,'aggregate':aggregate,
        'paired_seed_differences':contrasts,'analysis_source_sha256':file_hash('scripts/analyze_recurrent_temporal.py'),
        'no_test_evaluation':True,'new_training_runs':6,'reused_controls':3,'new_training_updates':18000,
        'scope':'Validation-selected development results at three previously examined seeds; equal exposure and recurrent architecture; unequal compute; fixed LR and confidence policy.'}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Recurrent temporal-supervision control','',
        'All six new auxiliary-objective runs completed. Three final-CE runs are reused unchanged. Each objective uses the same recurrent architecture, LR0.001, T16, data exposure, paired seeds23/29/31 and confidence readout. Results below use original validation data only. No test forward was performed.','',
        '## Selected validation results','',
        '| Objective | Accuracy, mean ± SD | CE, mean ± SD | Training min/run, mean ± SD |','|---|---:|---:|---:|']
    for objective in OBJECTIVES:
        a=aggregate[objective];acc,ce,time=[a[k] for k in ('validation_accuracy','validation_ce','training_minutes')]
        lines.append(f"| {objective} | {100*acc['mean']:.2f}% ± {100*acc['sample_sd']:.2f} points | {ce['mean']:.6g} ± {ce['sample_sd']:.3g} | {time['mean']:.1f} ± {time['sample_sd']:.1f} |")
    lines+=['','SD is the sample standard deviation over three development seeds, not a confidence interval. Each seed selects its checkpoint by minimum confidence-readout validation CE; accuracy requires unrestricted correct answer+EOS generation. The same128 maps are used across seeds and objectives. Selection on these maps makes these development scores, not held-out confirmation.','',
        '## Every declared cell','',
        '| Objective | Seed | Update | Validation CE | Validation accuracy | Train min | Run min | Peak GiB | Reused |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for t in trials:
        lines.append(f"| {t['objective']} | {t['seed']} | {t['step']} | {t['validation_ce']:.6g} | {t['validation_generation']['exact_match']:.2%} | {t['training_seconds']/60:.1f} | {t['wall_seconds']/60:.1f} | {t['peak_allocated_bytes']/2**30:.3f} | {'yes' if t['reused'] else 'no'} |")
    lines+=['','## Paired objective differences','',
        '| Auxiliary minus final CE | Accuracy difference, mean ± SD (points) | CE difference, mean ± SD |',
        '|---|---:|---:|']
    for objective,delta in contrasts.items():
        a,c=delta['accuracy'],delta['ce'];lines.append(f"| {objective} | {100*a['mean']:+.2f} ± {100*a['sample_sd']:.2f} | {c['mean']:+.6g} ± {c['sample_sd']:.3g} |")
    lines+=['','The JSON retains per-seed paired differences and per-map predictions. No statistical-significance claim is made from three seeds.','',
        '![Policy-matched validation trajectories](validation_curves.png)','',
        '![Seed variability and added training cost](objective_comparison.png)','',
        '## Correctness and cost','',
        '- Eleven GPU checks passed: exact CTM objective/gradient parity at T16 under FP32/BF16, masking and earliest ties, independent recurrent truncations/gradients, unchanged initialization and inference, checkpoint metadata and exact archived final-CE training replay.',
        '- Every run receives96,000 example presentations,5,184,000 input tokens and192,000 supervised answer/EOS labels. Parameters remain525,984.',
        '- Uniform and dynamic auxiliary training decode each recurrent state using the same coda/head without feeding decoded states back. Both execute49 decoder blocks per sequence versus34 for final CE. Counts are not FLOPs. Training-loop time includes the full3,000 updates.',
        '- Dynamic training uses CTM native-logits-dtype certainty; inference confidence uses FP32 entropy. The minimum-CE branch uses labels during training only. No monotonic penalty is added.',
        '- Reused controls are checked against their archived source bytes and checkpoint/metric hashes. The current runner extension preserves the default final-CE path; original archives remain unchanged.','',
        '## Interpretation limits','',
        'This isolates the temporal objective within one recurrent architecture at a fixed learning rate and fixed data exposure. Extra auxiliary decoding increases compute. Objectives may have different optimizer preferences; the fixed-LR result is not an exhaustive comparison of their attainable performance. The seeds were already examined in the preceding confirmation and are explicitly development seeds here.','',
        'These results do not establish a broad architecture ranking or reproduce an unidentified historical CTM recipe. Any revised recipe requires a separately declared confirmation; the preceding fresh test set remains closed. Shuffled-training and multistep-task controls remain separate work.','',
        'See [frozen plan](PLAN_BEFORE_RUNS.md), `registry.json`, `summary.json`, all saved validation checkpoints and source snapshots.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(OBJECTIVES,('tab:gray','tab:blue','tab:orange')))
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True,layout='constrained')
    for ax,seed in zip(axes,registry['seeds']):
        for objective in OBJECTIVES:
            values=curves[f'{objective}_seed{seed}']
            ax.plot([r['step'] for r in values],[r['validation']['loss'] for r in values],label=objective,color=colors[objective])
        ax.set(title=f'Seed {seed}',xlabel='Optimizer updates',ylabel='Validation answer/EOS CE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Recurrent T16: fixed LR0.001 and confidence readout, paired objective control')
    for ext in ('png','pdf'):fig.savefig(ROOT/f'validation_curves.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for x,objective in enumerate(OBJECTIVES):
        a=aggregate[objective]['validation_accuracy'];times=aggregate[objective]['training_minutes']
        axes[0].scatter([x-.08,x,x+.08],[100*v for v in a['values']],color=colors[objective],zorder=3)
        axes[0].errorbar(x,100*a['mean'],yerr=100*a['sample_sd'],fmt='_',markersize=22,color='black',capsize=5)
        axes[1].bar(x,times['mean'],color=colors[objective],yerr=times['sample_sd'],capsize=5)
    for ax in axes:ax.set(xticks=range(3),xticklabels=('Final CE','Uniform','Dynamic'));ax.grid(axis='y',alpha=.2)
    axes[0].set(ylabel='Selected validation answer + EOS (%)',ylim=(0,105))
    axes[1].set(ylabel='Training-loop minutes / full-budget run')
    fig.suptitle('Development seeds23/29/31: mean ± sample SD; unequal compute')
    for ext in ('png','pdf'):fig.savefig(ROOT/f'objective_comparison.{ext}',dpi=180)
    plt.close(fig)
    print('\n'.join(lines[:12]),flush=True)

if __name__=='__main__':main()
