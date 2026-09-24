"""Freeze the training-presentation intervention and its ordered controls."""
from dataclasses import asdict,replace
import hashlib,json,shutil,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
from ctm_transformer.experiment import file_hash
from ctm_transformer.presentation_control import create_presentation_control,validate_presentation_control
from ctm_transformer.research import load_research_config

ROOT=Path('research/results/presentation_control_v1');DATA=Path('research/data/presentation_control_v1')


def read(p):return json.loads(Path(p).read_text())


def main():
    assert not (ROOT/'registry.json').exists()
    suites=list(ET.parse('/tmp/presentation-checks.xml').getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)==5
    assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==int(s.attrib.get('skipped',0))==0 for s in suites)
    ROOT.mkdir(parents=True,exist_ok=True)
    create_presentation_control('research/data/ordered_pointer_v1',DATA)
    data=validate_presentation_control(DATA)
    assert len(data['train'])==2048 and len(data['validation'])==len(data['validation_shuffled'])==128
    shutil.copyfile('/tmp/presentation-checks.xml',ROOT/'checks.xml')
    shutil.copyfile('/tmp/presentation-checks.log',ROOT/'checks.log')
    shutil.copyfile('research/PRESENTATION_CONTROL.md',ROOT/'PLAN_BEFORE_RUNS.md')
    oldroot=Path('research/results/fresh_confirmation_v1');old=read(oldroot/'checkpoints.json');pre=read(oldroot/'pre_run_source.json')
    assert file_hash(oldroot/'pre_run_source.zip')==pre['archive_sha256']
    cfgroot=Path('research/configs/presentation_control_v1');cfgroot.mkdir(parents=True,exist_ok=True)
    sources={p:file_hash(p) for p in read(oldroot/'registry.json')['training_source_sha256']}
    entries=[]
    for seed in (23,29,31):
        for family in ('ctm','transformer','recurrent_depth'):
            t=next(t for t in old['models'] if t['cell']==f'{family}_seed{seed}')
            assert t['role']=='independent_confirmation'
            record=read(Path(t['run_directory'])/'research_run.json')
            assert file_hash(t['summary_path'])==t['summary_sha256'] and file_hash(Path(t['run_directory'])/'metrics.jsonl')==t['metrics_sha256']
            with zipfile.ZipFile(oldroot/'pre_run_source.zip') as z:
                for p,h in record['code_sha256'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h,p
            for condition in ('ordered','shuffled'):
                cell=f'{family}_{condition}_seed{seed}';reused=condition=='ordered'
                if reused:
                    path=Path(t['config']);directory=t['run_directory'];summary=t['summary_path']
                else:
                    c,_=load_research_config(t['config']);directory=f'research/runs/presentation_control_v1/seed{seed}/{family}'
                    c=replace(c,checkpoint_dir=directory);path=cfgroot/f'{cell}.json'
                    path.write_text(json.dumps({'schema_version':1,'name':cell+'_v1','model_family':family,'config':asdict(c)},indent=2)+'\n')
                    summary=str(ROOT/f'{cell}.summary.json')
                final=Path(t['run_directory'])/'final.pt'
                entries.append({'cell':cell,'seed':seed,'model_family':family,'training_presentation':condition,'reused':reused,
                    'config':str(path),'config_sha256':file_hash(path),'run_directory':str(directory),'summary_path':summary,
                    'selection_readouts':[t['readout']],'reference_control':t if reused else None,
                    'control_run_manifest_sha256':file_hash(Path(t['run_directory'])/'research_run.json') if reused else None,
                    'control_final_sha256':file_hash(final) if reused else None})
    registry={'schema_version':1,'name':'presentation_control_v1','entries':entries,'seeds':[23,29,31],
        'role':'paired development training-presentation control; fixed-update primary endpoint; no test evaluation',
        'train_path':str(DATA/'pointer/train.jsonl'),'validation_path':str(DATA/'pointer/validation.jsonl'),
        'ordered_train_path':'research/data/ordered_pointer_v1/pointer/train.jsonl',
        'dataset':str(DATA),'dataset_manifest_sha256':file_hash(DATA/'manifest.json'),'training_source_sha256':sources,
        'control_source_archive':str(oldroot/'pre_run_source.zip'),'control_source_archive_sha256':file_hash(oldroot/'pre_run_source.zip'),
        'primary_checkpoint_step':3000,'gpu_queues':{
            'cuda:0':['ctm_shuffled_seed23','ctm_shuffled_seed31','transformer_shuffled_seed23','transformer_shuffled_seed29','transformer_shuffled_seed31'],
            'cuda:1':['ctm_shuffled_seed29','recurrent_depth_shuffled_seed23','recurrent_depth_shuffled_seed29','recurrent_depth_shuffled_seed31']}}
    for key in ('train','validation','ordered_train'):registry[key+'_sha256']=file_hash(registry[key+'_path'])
    (ROOT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    files=dict(sources)
    for p in ('ctm_transformer/presentation_control.py','ctm_transformer/pointer_diagnostics.py','scripts/prepare_presentation_control.py',
              'scripts/run_registry_trials.py','scripts/run_presentation_control.py','tests/test_presentation_control.py',
              str(ROOT/'registry.json'),str(ROOT/'PLAN_BEFORE_RUNS.md'),str(ROOT/'checks.xml'),str(DATA/'manifest.json'),
              'research/results/recurrent_temporal_v1/checks.xml'):
        files[p]=file_hash(p)
    for e in entries:files[e['config']]=file_hash(e['config'])
    for p in DATA.rglob('*.jsonl'):files[str(p)]=file_hash(p)
    with zipfile.ZipFile(ROOT/'pre_run_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p)
    (ROOT/'pre_run_source.json').write_text(json.dumps({'files':files,'archive_sha256':file_hash(ROOT/'pre_run_source.zip')},indent=2)+'\n')
    print('Frozen nine shuffled runs and nine ordered controls, paired at step 3000.')

if __name__=='__main__':main()
