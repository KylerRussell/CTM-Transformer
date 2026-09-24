"""Lock all independent-seed checkpoints before any fresh-test forward pass."""
import json,zipfile
from pathlib import Path
from ctm_transformer.experiment import file_hash
from ctm_transformer.fresh_pointer import validate_fresh_suite
from scripts.confirmation_common import read,audit_trial

ROOT=Path('research/results/fresh_confirmation_v1')


def main():
    target=ROOT/'checkpoints.json'
    if target.exists():raise ValueError('Checkpoint freeze already exists')
    registry=read(ROOT/'registry.json');recipes=read(ROOT/'recipes.json');pre=read(ROOT/'pre_run_source.json')
    assert file_hash(ROOT/'recipes.json')==registry['recipes_sha256']
    for p,h in pre['files'].items():assert file_hash(p)==h,p
    validate_fresh_suite(recipes['fresh_dataset'])
    assert file_hash(Path(recipes['fresh_dataset'])/'manifest.json')==recipes['fresh_manifest_sha256']
    primary=[]
    for entry in registry['entries']:
        t=audit_trial(entry,registry);recipe=recipes['recipes'][t['model_family']]
        assert t['readout']==recipe['readout'] and t['parameters']==recipe['parameters']
        t['role']='independent_confirmation';primary.append(t)
    assert len(primary)==9 and {t['seed'] for t in primary}==set(recipes['primary_seeds'])
    references=[]
    for family,t in recipes['recipes'].items():
        assert file_hash(t['checkpoint'])==t['checkpoint_sha256']
        references.append({**t,'cell':f'{family}_seed17_reference','role':'development_reference'})
    sources={p:file_hash(p) for p in registry['training_source_sha256']}
    for p in ('scripts/freeze_fresh_confirmation.py','scripts/confirmation_common.py','scripts/evaluate_fresh_confirmation.py','scripts/eval_harness.py','ctm_transformer/fresh_pointer.py'):
        sources[p]=file_hash(p)
    with zipfile.ZipFile(ROOT/'pre_evaluation_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources):z.write(p,p)
    result={'complete':True,'recipes_sha256':file_hash(ROOT/'recipes.json'),'registry_sha256':file_hash(ROOT/'registry.json'),
        'source_sha256':sources,'source_archive_sha256':file_hash(ROOT/'pre_evaluation_source.zip'),
        'fresh_dataset':recipes['fresh_dataset'],'fresh_manifest_sha256':recipes['fresh_manifest_sha256'],
        'no_fresh_test_forward_before_freeze':True,'models':primary+references}
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Frozen nine independent-seed checkpoints and three separate development references.')

if __name__=='__main__':main()
