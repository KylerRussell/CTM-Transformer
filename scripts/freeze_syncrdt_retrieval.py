"""Audit every Sync-RDT retrieval run and freeze its fixed-update endpoint."""
import json,math,zipfile
from pathlib import Path
import torch
from ctm_transformer.experiment import file_hash
from ctm_transformer.mqar_suite import QUERIES

ROOT=Path('research/results/syncrdt_retrieval_v1')


def read(path):return json.loads(Path(path).read_text())


def audit(entry,registry):
    directory=Path(entry['run_directory']);run=read(directory/'research_run.json');summary=read(entry['summary_path'])
    config=run['effective_config'];policy=entry['readout']
    steps,warmup,interval,batch=config['max_steps'],config['warmup_steps'],config['eval_interval'],config['batch_size']
    assert (steps,warmup,interval)==(registry['primary_checkpoint_step'],registry['warmup_steps'],registry['eval_interval'])
    assert math.isclose(config['learning_rate'],registry['learning_rate'])
    assert run['runner']==summary['runner']==registry['runner'] and run['evaluator']=='ctm_transformer.dense_pointer.evaluate_dense'
    assert run['model_factory']==f"ctm_transformer.sync_rdt.cell_factory[{entry['factory_cell']}]"
    assert summary['complete'] and summary['steps']==steps and summary['model_family']=='recurrent_depth'
    assert (run['seed'],run['data_seed'],run['depth_seed'])==(entry['seed'],entry['seed']+1,entry['seed']+2)
    assert file_hash(entry['config'])==entry['config_sha256']==run['config_identity']['sha256']
    declared=read(entry['config'])['config']
    assert {k for k,v in declared.items() if config[k]!=v}<={'device','data_path','eval_data_path','checkpoint_dir'}
    assert all(config[k]==v for k,v in entry['config_overrides'].items())
    assert run['data']['training']['sha256']==entry['train_sha256'] and run['data']['evaluation']['sha256']==entry['validation_sha256']
    assert run['checkpoint_selection']['readouts']==[policy]
    assert summary['worker_source_sha256']==file_hash('scripts/run_syncrdt_retrieval.py')
    for p,h in run['code_sha256'].items():
        if p in registry['training_source_sha256']:assert file_hash(p)==h==registry['training_source_sha256'][p],p
    assert summary['parameters']['total']==run['parameters']['total']
    rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,steps+1))
    labels=QUERIES+1;first=rows[0];width=first['tokens_seen']//batch
    curve=[]
    for r in rows:
        step=r['step']
        factor=step/warmup if step<=warmup else .1+.45*(1+math.cos(math.pi*(step-warmup)/(steps-warmup)))
        assert math.isclose(r['lr'],config['learning_rate']*factor,rel_tol=1e-12)
        assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
        assert r['examples_seen']==step*batch and r['supervised_tokens_seen']==step*batch*labels and r['tokens_seen']==step*batch*width
        assert r['thought_steps']==16
        if 'validation_readouts' in r:
            v=r['validation_readouts'][policy]
            assert v['target_tokens']==256*labels and len(v['predictions'])==256 and math.isfinite(v['loss'])
            curve.append({'step':step,'ce':v['loss'],'answer_accuracy':v['answer_accuracy'],
                          'position1':v['answer_accuracy_by_position'][0],'sequence_exact':v['sequence_exact']})
    assert [c['step'] for c in curve]==list(range(interval,steps+1,interval))
    assert summary['examples_seen']==steps*batch==entry['train_maps']  # Every training map is presented once.
    final=directory/'final.pt';payload=torch.load(final,map_location='cpu',weights_only=False)
    assert payload['step']==steps and payload['config']==config and payload['runner']==registry['runner']
    return {'cell':entry['cell'],'study_cell':entry['study_cell'],'factory_cell':entry['factory_cell'],'seed':entry['seed'],
        'readout':policy,'step':steps,'checkpoint':str(final),'checkpoint_sha256':file_hash(final),'validation_curve':curve,
        'final_training_loss_last100':sum(r['loss'] for r in rows[-100:])/100,'parameters':summary['parameters'],
        'training_seconds':summary['training_seconds'],'wall_seconds':summary['wall_seconds'],'peak_allocated_bytes':summary['peak_allocated_bytes'],
        'run_directory':str(directory),'summary_path':entry['summary_path'],'summary_sha256':file_hash(entry['summary_path']),
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
    sources=dict(registry['training_source_sha256'])
    for p in ('scripts/freeze_syncrdt_retrieval.py','scripts/evaluate_syncrdt_retrieval.py'):sources[p]=file_hash(p)
    with zipfile.ZipFile(ROOT/'pre_evaluation_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources):z.write(p,p)
    result={'complete':True,'registry_sha256':file_hash(ROOT/'registry.json'),'dataset':registry['dataset'],
        'dataset_manifest_sha256':registry['dataset_manifest_sha256'],'models':records,'source_sha256':sources,
        'source_archive_sha256':file_hash(ROOT/'pre_evaluation_source.zip'),'primary_checkpoint_rule':f"fixed update {registry['primary_checkpoint_step']}",
        'all_primary_checkpoints_frozen_before_evaluation':True,'no_test_evaluation':True}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    print(f"Audited {sum(r['step'] for r in records):,} updates and froze {len(records)} final checkpoints before evaluation.",flush=True)

if __name__=='__main__':main()
