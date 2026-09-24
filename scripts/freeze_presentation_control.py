"""Audit all paired runs and freeze the fixed-update development endpoints."""
import hashlib,json,math,zipfile
from pathlib import Path
import torch
from ctm_transformer.experiment import file_hash
from ctm_transformer.presentation_control import validate_presentation_control

ROOT=Path('research/results/presentation_control_v1')


def read(path):return json.loads(Path(path).read_text())


def audit(entry,registry):
    directory=Path(entry['run_directory']);run=read(directory/'research_run.json');summary=read(entry['summary_path'])
    config=run['effective_config'];policy=entry['selection_readouts'][0];family=entry['model_family']
    assert summary['complete'] and summary['steps']==3000 and summary['model_family']==family
    assert (run['seed'],run['data_seed'],run['depth_seed'])==(entry['seed'],entry['seed']+1,entry['seed']+2)
    assert file_hash(entry['config'])==entry['config_sha256']==run['config_identity']['sha256']
    declared=read(entry['config'])['config']
    assert {k for k,v in declared.items() if config[k]!=v}<={'device','data_path','eval_data_path','checkpoint_dir'}
    train_key='ordered_train_sha256' if entry['reused'] else 'train_sha256'
    assert run['data']['training']['sha256']==registry[train_key]
    assert run['data']['evaluation']['sha256']==registry['validation_sha256']
    assert run['checkpoint_selection']['readouts']==[policy] and run['checkpoint_selection']['save_every_validation']
    assert summary['worker_source_sha256']==file_hash('scripts/run_registry_trials.py')
    if entry['reused']:
        ref=entry['reference_control']
        assert file_hash(directory/'research_run.json')==entry['control_run_manifest_sha256']
        assert file_hash(entry['summary_path'])==ref['summary_sha256']
        assert file_hash(directory/'metrics.jsonl')==ref['metrics_sha256']
        assert file_hash(directory/'best.pt')==ref['checkpoint_sha256']
        assert file_hash(directory/'final.pt')==entry['control_final_sha256']
        with zipfile.ZipFile(registry['control_source_archive']) as z:
            for p,h in run['code_sha256'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h,p
    else:
        for p,h in run['code_sha256'].items():assert file_hash(p)==h==registry['training_source_sha256'][p],p
    rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,3001))
    validations=[];blocks={'ctm':32,'transformer':2,'recurrent_depth':34}[family]
    for r in rows:
        step=r['step'];factor=step/30 if step<=30 else .1+.45*(1+math.cos(math.pi*(step-30)/2970))
        assert math.isclose(r['lr'],config['learning_rate']*factor,rel_tol=1e-12)
        assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
        assert r['tokens_seen']==step*1728 and r['examples_seen']==step*32 and r['supervised_tokens_seen']==step*64
        assert r['block_applications_per_sequence']==blocks and r['thought_steps']==config['max_thought_steps']
        if 'validation_readouts' in r:
            assert (directory/f'step_{step:06d}.pt').exists()
            assert set(r['validation_readouts'])=={policy,'diagnostics'}
            v=r['validation_readouts'][policy]
            assert v['target_tokens']==256 and len(v['predictions'])==128 and math.isfinite(v['loss'])
            assert v['generation']['correct']==sum(p['correct'] for p in v['predictions'])
            assert v['generation']['exact_match']==v['generation']['correct']/128
            assert r['validation']['readout_policy']==policy and r['validation']['loss']==v['loss']
            validations.append(r)
    assert [r['step'] for r in validations]==list(range(100,3001,100))
    best=min(validations,key=lambda r:(r['validation']['loss'],r['step']))
    assert summary['best_validation_loss']==best['validation']['loss'] and summary['best_readout_policy']==policy
    assert summary['examples_seen']==96000 and summary['tokens_seen']==5184000 and summary['supervised_tokens_seen']==192000
    assert summary['token_block_applications']==summary['padded_token_positions']*blocks
    final=directory/'final.pt';payload=torch.load(final,map_location='cpu',weights_only=False)
    assert payload['step']==3000 and payload['config']==config and payload['tokens_seen']==5184000
    last=rows[-1]['validation_readouts'][policy]
    return {'cell':entry['cell'],'model_family':family,'seed':entry['seed'],'training_presentation':entry['training_presentation'],
        'readout':policy,'step':3000,'checkpoint':str(final),'checkpoint_sha256':file_hash(final),
        'checkpoint_file_readout_metadata':payload.get('readout_policy'),
        'ordered_validation_ce':last['loss'],'ordered_validation_generation':last['generation'],
        'secondary_best_ordered_validation':{'step':best['step'],'ce':best['validation']['loss'],'generation':best['validation']['generation']},
        'parameters':summary['parameters'],'training_seconds':summary['training_seconds'],'wall_seconds':summary['wall_seconds'],
        'peak_allocated_bytes':summary['peak_allocated_bytes'],'reused':entry['reused'],
        'run_directory':str(directory),'summary_path':entry['summary_path'],'summary_sha256':file_hash(entry['summary_path']),
        'metrics_sha256':file_hash(directory/'metrics.jsonl'),'config':entry['config'],'config_sha256':entry['config_sha256']}


def main():
    torch.set_num_threads(4)
    assert not (ROOT/'checkpoints.json').exists()
    registry=read(ROOT/'registry.json');pre=read(ROOT/'pre_run_source.json')
    assert file_hash(ROOT/'pre_run_source.zip')==pre['archive_sha256']
    assert file_hash(registry['control_source_archive'])==registry['control_source_archive_sha256']
    for p,h in pre['files'].items():assert file_hash(p)==h,p
    for k in ('train','validation','ordered_train'):assert file_hash(registry[k+'_path'])==registry[k+'_sha256']
    validate_presentation_control(registry['dataset'])
    assert not list(ROOT.glob('*.failure.json'))
    records=[audit(e,registry) for e in registry['entries']]
    for seed in registry['seeds']:
        for family in ('ctm','transformer','recurrent_depth'):
            a,b=[read(e['config'])['config'] for e in registry['entries'] if e['seed']==seed and e['model_family']==family]
            assert {k for k,v in a.items() if b[k]!=v}<={'checkpoint_dir'}
    sources={p:file_hash(p) for p in registry['training_source_sha256']}
    for p in ('scripts/freeze_presentation_control.py','scripts/evaluate_presentation_control.py','scripts/eval_harness.py',
              'ctm_transformer/presentation_control.py','ctm_transformer/pointer_diagnostics.py'):sources[p]=file_hash(p)
    with zipfile.ZipFile(ROOT/'pre_evaluation_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources):z.write(p,p)
    result={'complete':True,'registry_sha256':file_hash(ROOT/'registry.json'),'dataset':registry['dataset'],
        'dataset_manifest_sha256':registry['dataset_manifest_sha256'],'models':records,'source_sha256':sources,
        'source_archive_sha256':file_hash(ROOT/'pre_evaluation_source.zip'),'primary_checkpoint_rule':'fixed update 3000',
        'all_primary_checkpoints_frozen_before_paired_evaluation':True,'no_test_evaluation':True}
    with (ROOT/'checkpoints.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Audited 54,000 updates and froze all 18 final checkpoints before paired validation evaluation.',flush=True)

if __name__=='__main__':main()
