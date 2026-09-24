"""Shared integrity audit for fixed-budget optimizer and seed studies."""
import json,math
from pathlib import Path
import torch
from ctm_transformer.experiment import file_hash


def read(path):return json.loads(Path(path).read_text())


def audit_trial(entry,registry):
    directory=Path(entry['run_directory']);s=read(entry['summary_path']);r=read(directory/'research_run.json');cfg=r['effective_config']
    assert s['complete'] and s['steps']==3000
    assert r['seed']==entry['seed'] and r['data_seed']==entry['seed']+1 and r['depth_seed']==entry['seed']+2
    assert file_hash(entry['config'])==entry['config_sha256']==r['config_identity']['sha256']
    declared=read(entry['config'])['config']
    assert {k for k,v in declared.items() if cfg[k]!=v} <= {'data_path','eval_data_path','checkpoint_dir','device'}
    assert r['model_family']==entry['model_family'] and r['parameters']==s['parameters']
    assert r['data']['training']['sha256']==file_hash(registry['train_path'])
    assert r['data']['evaluation']['sha256']==file_hash(registry['validation_path'])
    for p,h in r['code_sha256'].items():assert file_hash(p)==h,(entry['cell'],p)
    for p,h in registry['training_source_sha256'].items():assert file_hash(p)==h,p
    if entry.get('reused'):
        assert file_hash(entry['summary_path'])==entry['reused_summary_sha256']
        assert s['worker_source_sha256']==file_hash('scripts/run_readout_comparison.py')
    else:assert s['worker_source_sha256']==file_hash('scripts/run_registry_trials.py')
    policies=entry['selection_readouts'];assert r['checkpoint_selection']['readouts']==policies
    assert r['checkpoint_selection']['save_every_validation']
    for key,value in {'examples_seen':96000,'tokens_seen':5184000,'supervised_tokens_seen':192000}.items():assert s[key]==value
    rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [x['step'] for x in rows]==list(range(1,3001))
    choices=[];values={}
    for x in rows:
        step=x['step'];factor=step/30 if step<=30 else .1+.45*(1+math.cos(math.pi*(step-30)/2970))
        assert math.isclose(x['lr'],cfg['learning_rate']*factor,rel_tol=1e-12)
        assert math.isfinite(x['loss']) and math.isfinite(x['gradient_norm'])
        assert x['examples_seen']==step*32 and x['tokens_seen']==step*1728 and x['supervised_tokens_seen']==step*64
        if 'validation_readouts' not in x:continue
        assert step%100==0 and (directory/f'step_{step:06d}.pt').exists()
        for index,policy in enumerate(policies):
            v=x['validation_readouts'][policy];assert v['target_tokens']==256 and len(v['predictions'])==128
            assert v['generation']['correct']==sum(p['correct'] for p in v['predictions'])
            assert math.isfinite(v['loss'])
            choices.append((v['loss'],step,index,policy));values[step,policy]=v
        selected=min(policies,key=lambda p:x['validation_readouts'][p]['loss'])
        assert x['validation']['readout_policy']==selected and x['validation']['loss']==x['validation_readouts'][selected]['loss']
    assert len(choices)==30*len(policies)
    best=min(choices);assert best[0]==s['best_validation_loss'] and best[3]==s['best_readout_policy']
    checkpoint=directory/'best.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['step']==best[1] and payload['readout_policy']==best[3] and payload['config']==cfg
    assert payload['tokens_seen']==best[1]*1728
    return {'cell':entry['cell'],'seed':entry['seed'],'model_family':entry['model_family'],
        'objective':entry.get('objective',cfg.get('temporal_loss_type','final_ce')),'learning_rate':cfg['learning_rate'],
        'readout':best[3],'step':best[1],'validation_ce':best[0],'validation_generation':values[best[1],best[3]]['generation'],
        'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':file_hash(checkpoint),'config':entry['config'],
        'config_sha256':entry['config_sha256'],'run_directory':str(directory),'summary_path':entry['summary_path'],
        'summary_sha256':file_hash(entry['summary_path']),'metrics_sha256':file_hash(directory/'metrics.jsonl'),
        'parameters':s['parameters'],'training_seconds':s['training_seconds'],'wall_seconds':s['wall_seconds'],
        'peak_allocated_bytes':s['peak_allocated_bytes'],'reused':entry.get('reused',False)}
