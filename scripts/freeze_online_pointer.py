"""Audit every fresh-map pointer run and freeze its fixed-update endpoint."""
import json,math,zipfile
from pathlib import Path
import torch
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/online_pointer_v1')
BLOCKS={'ctm':32,'transformer':2,'recurrent_depth':34}


def read(path):return json.loads(Path(path).read_text())


def audit(entry,registry):
    directory=Path(entry['run_directory']);run=read(directory/'research_run.json');summary=read(entry['summary_path'])
    config=run['effective_config'];policy=entry['readout'];family=entry['model_family']
    assert summary['complete'] and summary['steps']==3000 and summary['model_family']==family
    assert (run['seed'],run['data_seed'],run['depth_seed'])==(entry['seed'],entry['seed']+1,entry['seed']+2)
    assert file_hash(entry['config'])==entry['config_sha256']==run['config_identity']['sha256']
    declared=read(entry['config'])['config']
    assert {k for k,v in declared.items() if config[k]!=v}<={'device','data_path','eval_data_path','checkpoint_dir'}
    assert run['data']['training']['sha256']==entry['train_sha256']==file_hash(entry['train_path'])
    assert run['data']['evaluation']['sha256']==registry['validation_sha256']
    assert run['checkpoint_selection']['readouts']==[policy] and run['checkpoint_selection']['save_every_validation']
    assert summary['worker_source_sha256']==file_hash('scripts/run_registry_trials.py')
    for p,h in run['code_sha256'].items():assert file_hash(p)==h==registry['training_source_sha256'][p],p
    first=json.loads(Path(entry['train_path']).open().readline());width=1+len(first['prompt'])+len(first['answer'])
    rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,3001))
    validations=[]
    for r in rows:
        step=r['step'];factor=step/30 if step<=30 else .1+.45*(1+math.cos(math.pi*(step-30)/2970))
        assert math.isclose(r['lr'],config['learning_rate']*factor,rel_tol=1e-12)
        assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
        assert r['examples_seen']==step*32 and r['supervised_tokens_seen']==step*64 and r['tokens_seen']==step*32*width
        assert r['block_applications_per_sequence']==BLOCKS[family] and r['thought_steps']==config['max_thought_steps']
        if 'validation_readouts' in r:
            assert (directory/f'step_{step:06d}.pt').exists() and set(r['validation_readouts'])=={policy,'diagnostics'}
            v=r['validation_readouts'][policy]
            assert v['target_tokens']==768 and len(v['predictions'])==384 and math.isfinite(v['loss'])
            assert v['generation']['correct']==sum(p['correct'] for p in v['predictions'])
            validations.append(r)
    assert [r['step'] for r in validations]==list(range(100,3001,100))
    best=min(validations,key=lambda r:(r['validation']['loss'],r['step']))
    assert summary['best_validation_loss']==best['validation']['loss'] and summary['examples_seen']==96000
    final=directory/'final.pt';payload=torch.load(final,map_location='cpu',weights_only=False)
    assert payload['step']==3000 and payload['config']==config
    last=rows[-1]['validation_readouts'][policy]
    unique=len({json.loads(x)['id'] for x in Path(entry['train_path']).read_text().splitlines()})
    return {'cell':entry['cell'],'model_family':family,'seed':entry['seed'],'condition':entry['condition'],
        'readout':policy,'trained_thought_steps':entry['trained_thought_steps'],'evaluation_thought_steps':entry['evaluation_thought_steps'],
        'step':3000,'checkpoint':str(final),'checkpoint_sha256':file_hash(final),
        'unique_training_maps':unique,'mean_presentations_per_map':96000/unique,
        'final_training_loss_last100':sum(r['loss'] for r in rows[-100:])/100,
        'validation_ce_at_3000':last['loss'],'validation_generation_at_3000':last['generation'],
        'secondary_best_validation':{'step':best['step'],'ce':best['validation']['loss'],'generation':best['validation']['generation']},
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
    for p in ('scripts/freeze_online_pointer.py','scripts/evaluate_online_pointer.py','scripts/eval_harness.py',
              'ctm_transformer/online_pointer.py','ctm_transformer/readout_selection.py','ctm_transformer/confidence_readout.py',
              'ctm_transformer/algorithmic_eval.py'):sources[p]=file_hash(p)
    with zipfile.ZipFile(ROOT/'pre_evaluation_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources):z.write(p,p)
    result={'complete':True,'registry_sha256':file_hash(ROOT/'registry.json'),'dataset':registry['dataset'],
        'dataset_manifest_sha256':registry['dataset_manifest_sha256'],'models':records,'source_sha256':sources,
        'source_archive_sha256':file_hash(ROOT/'pre_evaluation_source.zip'),'primary_checkpoint_rule':'fixed update 3000',
        'all_primary_checkpoints_frozen_before_evaluation':True,'no_test_evaluation':True}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    print(f'Audited {len(records)*3000:,} updates and froze {len(records)} final checkpoints before evaluation.',flush=True)

if __name__=='__main__':main()
