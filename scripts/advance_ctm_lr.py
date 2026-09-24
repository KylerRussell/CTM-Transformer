"""Audit the CTM LR grid, freeze recipes, and declare independent-seed trials."""
from dataclasses import asdict,replace
import json,zipfile
from pathlib import Path
from ctm_transformer.experiment import file_hash
from ctm_transformer.fresh_pointer import validate_fresh_suite
from ctm_transformer.research import load_research_config,model_family
from scripts.confirmation_common import read,audit_trial

ROOT=Path('research/results/ctm_lr_v1');CONF=Path('research/results/fresh_confirmation_v1')


def main():
    if (CONF/'recipes.json').exists():raise ValueError('Recipe freeze already exists')
    r=read(ROOT/'registry.json');pre=read(ROOT/'pre_run_source.json')
    for p,h in pre['files'].items():assert file_hash(p)==h,('pre-run file changed',p)
    trials=[audit_trial(e,r) for e in r['entries']]
    winner=min(trials,key=lambda t:t['validation_ce'])
    baseline=read(r['baseline_selection_path']);assert file_hash(r['baseline_selection_path'])==r['baseline_selection_sha256']
    recipes={'ctm':winner}
    for family in ('transformer','recurrent_depth'):
        old=baseline['winners'][family];entry=next(e for e in read('research/results/readout_comparison_v1/registry.json')['entries'] if e['cell']==old['cell'])
        entry={**entry,'seed':17,'reused':True,'run_directory':f"research/runs/readout_comparison_v1/seed17/{old['cell']}",
            'summary_path':f"research/results/readout_comparison_v1/{old['cell']}.summary.json",'reused_summary_sha256':old['summary_sha256']}
        value=audit_trial(entry,r)
        assert value['checkpoint_sha256']==old['checkpoint_sha256'] and value['readout']==old['readout']
        recipes[family]=value
    fresh=Path('research/data/fresh_pointer_confirmation_v1');manifest=validate_fresh_suite(fresh)
    record={'complete':True,'selection_rule':'minimum original-validation CE across frozen CTM objective/LR grid; previous baseline choices retained; no test outcomes used',
        'ctm_trials':trials,'recipes':recipes,'primary_seeds':r['confirmation_seeds'],'development_reference_seed':17,
        'fresh_dataset':str(fresh),'fresh_manifest_sha256':file_hash(fresh/'manifest.json'),
        'fresh_maps':manifest['tasks']['pointer']['test_id']['examples'],
        'lr_registry_sha256':file_hash(ROOT/'registry.json'),'baseline_selection_sha256':file_hash(r['baseline_selection_path']),
        'training_source_sha256':r['training_source_sha256'],
        'selection_source_sha256':{p:file_hash(p) for p in ('scripts/advance_ctm_lr.py','scripts/confirmation_common.py','ctm_transformer/fresh_pointer.py')}}
    CONF.mkdir(parents=True,exist_ok=True)
    with (CONF/'recipes.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    entries=[];cfgdir=Path('research/configs/fresh_confirmation_v1');cfgdir.mkdir(parents=True,exist_ok=True)
    for seed in r['confirmation_seeds']:
        for family,recipe in recipes.items():
            c,_=load_research_config(recipe['config']);cell=f'{family}_seed{seed}'
            directory=Path(f'research/runs/fresh_confirmation_v1/seed{seed}/{family}')
            c=replace(c,checkpoint_dir=str(directory));path=cfgdir/(cell+'.json')
            path.write_text(json.dumps({'schema_version':1,'name':cell+'_v1','model_family':family,'config':asdict(c)},indent=2)+'\n')
            entries.append({'cell':cell,'seed':seed,'model_family':family,'config':str(path),'config_sha256':file_hash(path),
                'run_directory':str(directory),'summary_path':str(CONF/(cell+'.summary.json')),
                'selection_readouts':[recipe['readout']],'reused':False})
    registry={'schema_version':1,'name':'fresh_confirmation_v1','entries':entries,
        'train_path':r['train_path'],'validation_path':r['validation_path'],
        'training_source_sha256':r['training_source_sha256'],'recipes_sha256':file_hash(CONF/'recipes.json'),
        'fresh_manifest_sha256':record['fresh_manifest_sha256']}
    (CONF/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    # Record all execution dependencies before new-seed training begins.
    files={p:file_hash(p) for p in r['training_source_sha256']}
    for p in ('scripts/run_registry_trials.py','scripts/advance_ctm_lr.py','scripts/confirmation_common.py',str(CONF/'registry.json'),str(CONF/'recipes.json'),str(fresh/'manifest.json')):files[p]=file_hash(p)
    for e in entries:files[e['config']]=file_hash(e['config'])
    with zipfile.ZipFile(CONF/'pre_run_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p)
    (CONF/'pre_run_source.json').write_text(json.dumps({'files':files,'archive_sha256':file_hash(CONF/'pre_run_source.zip')},indent=2)+'\n')
    # A stable LR-stage record also makes development plots possible before confirmation.
    (ROOT/'selection.json').write_text(json.dumps({'complete':True,'trials':trials,'winner':winner,'confirmation_recipes_sha256':file_hash(CONF/'recipes.json')},indent=2)+'\n')
    print(json.dumps({family:{k:v[k] for k in ('cell','readout','validation_ce','learning_rate')} for family,v in recipes.items()},indent=2))

if __name__=='__main__':main()
