"""Audit every dense-calibration run and freeze its fixed-update endpoint."""
import json,math,zipfile
from pathlib import Path
import torch
from ctm_transformer.dense_pointer import QUERIES
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/dense_calibration_v1')
BLOCKS={'ctm':32,'transformer':2,'recurrent_depth':34}


def read(path):return json.loads(Path(path).read_text())


def audit(entry,registry):
    directory=Path(entry['run_directory']);run=read(directory/'research_run.json');summary=read(entry['summary_path'])
    config=run['effective_config'];policy=entry['readout'];family=entry['model_family']
    steps,warmup,interval,batch=config['max_steps'],config['warmup_steps'],config['eval_interval'],config['batch_size']
    assert steps==registry['primary_checkpoint_step'] and warmup==registry['warmup_steps'] and interval==registry['eval_interval']
    assert run['runner']==summary['runner']==registry['runner'] and run['evaluator']=='ctm_transformer.dense_pointer.evaluate_dense'
    assert summary['complete'] and summary['steps']==steps and summary['model_family']==family
    assert (run['seed'],run['data_seed'],run['depth_seed'])==(entry['seed'],entry['seed']+1,entry['seed']+2)
    assert file_hash(entry['config'])==entry['config_sha256']==run['config_identity']['sha256']
    declared=read(entry['config'])['config']
    assert {k for k,v in declared.items() if config[k]!=v}<={'device','data_path','eval_data_path','checkpoint_dir'}
    assert math.isclose(config['learning_rate'],entry['learning_rate'])
    assert run['data']['training']['sha256']==entry['train_sha256'] and run['data']['evaluation']['sha256']==registry['validation_sha256']
    assert run['checkpoint_selection']['readouts']==[policy] and run['checkpoint_selection']['save_every_validation']
    assert summary['worker_source_sha256']==file_hash('scripts/run_dense_calibration.py')
    for p,h in run['code_sha256'].items():assert file_hash(p)==h==registry['training_source_sha256'][p],p
    with Path(entry['train_path']).open() as f:first=json.loads(f.readline())
    width=1+len(first['prompt'])+len(first['answer']);labels=QUERIES+1
    rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,steps+1))
    validations=[]
    for r in rows:
        step=r['step']
        factor=step/warmup if step<=warmup else .1+.45*(1+math.cos(math.pi*(step-warmup)/(steps-warmup)))
        assert math.isclose(r['lr'],config['learning_rate']*factor,rel_tol=1e-12)
        assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
        assert r['examples_seen']==step*batch and r['supervised_tokens_seen']==step*batch*labels and r['tokens_seen']==step*batch*width
        assert r['block_applications_per_sequence']==BLOCKS[family] and r['thought_steps']==config['max_thought_steps']
        if 'validation_readouts' in r:
            assert (directory/f'step_{step:06d}.pt').exists() and set(r['validation_readouts'])=={policy,'diagnostics'}
            v=r['validation_readouts'][policy]
            assert v['target_tokens']==384*labels and len(v['predictions'])==384 and math.isfinite(v['loss'])
            assert math.isclose(v['answer_accuracy'],sum(map(sum,(p['correct'] for p in v['predictions'])))/(384*QUERIES))
            validations.append(r)
    assert [r['step'] for r in validations]==list(range(interval,steps+1,interval))
    # Each map is presented exactly once: the training file holds exactly steps x batch unique maps.
    assert summary['examples_seen']==steps*batch==entry['train_examples']
    best=min(validations,key=lambda r:(r['validation']['loss'],r['step']))
    assert summary['best_validation_loss']==best['validation']['loss']
    final=directory/'final.pt';payload=torch.load(final,map_location='cpu',weights_only=False)
    assert payload['step']==steps and payload['config']==config and payload['runner']==registry['runner']
    curve=[{'step':r['step'],'ce':r['validation']['loss'],'answer_accuracy':r['validation']['answer_accuracy'],
            'by_hop':{h:v['answer_accuracy'] for h,v in r['validation']['by_hop'].items()}} for r in validations]
    return {'cell':entry['cell'],'model_family':family,'seed':entry['seed'],'task':entry['task'],'learning_rate':entry['learning_rate'],
        'readout':policy,'trained_thought_steps':entry['trained_thought_steps'],'evaluation_thought_steps':entry['evaluation_thought_steps'],
        'step':steps,'checkpoint':str(final),'checkpoint_sha256':file_hash(final),'unique_training_maps':entry['train_examples'],
        'final_training_loss_last100':sum(r['loss'] for r in rows[-100:])/100,'validation_curve':curve,
        'secondary_best_validation':{'step':best['step'],'ce':best['validation']['loss']},
        'parameters':summary['parameters'],'training_seconds':summary['training_seconds'],'wall_seconds':summary['wall_seconds'],
        'peak_allocated_bytes':summary['peak_allocated_bytes'],'run_directory':str(directory),
        'summary_path':entry['summary_path'],'summary_sha256':file_hash(entry['summary_path']),
        'metrics_sha256':file_hash(directory/'metrics.jsonl'),'config':entry['config'],'config_sha256':entry['config_sha256']}


def main():
    torch.set_num_threads(4);path=ROOT/'checkpoints.json'
    if path.exists():
        try:
            if read(path)['complete']:print('Checkpoints already frozen',flush=True);return
        except (json.JSONDecodeError,KeyError):pass
        path.rename(ROOT/'checkpoints.incomplete.json')
    registry=read(ROOT/'registry.json');pre=read(ROOT/'pre_run_source.json')
    for p,h in pre['files'].items():assert file_hash(p)==h,p
    assert not list(ROOT.glob('*.failure.json'))
    records=[audit(e,registry) for e in registry['entries']]
    sources={p:file_hash(p) for p in registry['training_source_sha256']}
    for p in ('scripts/freeze_dense_calibration.py','scripts/evaluate_dense_calibration.py','scripts/eval_harness.py'):sources[p]=file_hash(p)
    with zipfile.ZipFile(ROOT/'pre_evaluation_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources):z.write(p,p)
    result={'complete':True,'registry_sha256':file_hash(ROOT/'registry.json'),'dataset':registry['dataset'],
        'dataset_manifest_sha256':registry['dataset_manifest_sha256'],'models':records,'source_sha256':sources,
        'source_archive_sha256':file_hash(ROOT/'pre_evaluation_source.zip'),'primary_checkpoint_rule':f"fixed update {registry['primary_checkpoint_step']}",
        'all_primary_checkpoints_frozen_before_evaluation':True,'no_test_evaluation':True}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    print(f"Audited {sum(r['step'] for r in records):,} updates and froze {len(records)} final checkpoints before evaluation.",flush=True)

if __name__=='__main__':main()
