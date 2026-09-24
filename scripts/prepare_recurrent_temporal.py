"""Freeze the paired objective study, including reused controls and source bytes."""
from dataclasses import asdict,replace
import hashlib,json,shutil,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
from ctm_transformer.experiment import file_hash
from ctm_transformer.research import load_research_config

ROOT=Path('research/results/recurrent_temporal_v1')


def read(path):return json.loads(Path(path).read_text())


def main():
    assert not (ROOT/'registry.json').exists()
    test=ET.parse('/tmp/recurrent-temporal-checks.xml').getroot()
    suites=list(test.iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)==11
    assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==int(s.attrib.get('skipped',0))==0 for s in suites)
    ROOT.mkdir(parents=True,exist_ok=True)
    shutil.copyfile('/tmp/recurrent-temporal-checks.xml',ROOT/'checks.xml')
    shutil.copyfile('/tmp/recurrent-temporal-checks.log',ROOT/'checks.log')
    shutil.copyfile('research/RECURRENT_TEMPORAL_CONTROL.md',ROOT/'PLAN_BEFORE_RUNS.md')
    previous=Path('research/results/fresh_confirmation_v1')
    old=read(previous/'checkpoints.json');pre=read(previous/'pre_run_source.json')
    assert file_hash(previous/'pre_run_source.zip')==pre['archive_sha256']
    sources={p:file_hash(p) for p in read(previous/'registry.json')['training_source_sha256']}
    sources['ctm_transformer/recurrent_temporal.py']=file_hash('ctm_transformer/recurrent_temporal.py')
    entries=[];configs=Path('research/configs/recurrent_temporal_v1');configs.mkdir(parents=True,exist_ok=True)
    for seed in (23,29,31):
        control=next(t for t in old['models'] if t['cell']==f'recurrent_depth_seed{seed}')
        assert control['readout']=='confidence' and control['role']=='independent_confirmation'
        assert file_hash(control['checkpoint'])==control['checkpoint_sha256']
        assert file_hash(control['summary_path'])==control['summary_sha256']
        record=read(Path(control['run_directory'])/'research_run.json')
        with zipfile.ZipFile(previous/'pre_run_source.zip') as archive:
            for p,h in record['code_sha256'].items():assert hashlib.sha256(archive.read(p)).hexdigest()==h,p
        for objective in ('final_ce','uniform','dynamic_aggregate'):
            cell=f'{objective}_seed{seed}';reused=objective=='final_ce'
            if reused:
                path=Path(control['config']);directory=control['run_directory'];summary=control['summary_path']
            else:
                c,_=load_research_config(control['config']);directory=f'research/runs/recurrent_temporal_v1/seed{seed}/{objective}'
                c=replace(c,checkpoint_dir=directory);path=configs/(cell+'.json')
                path.write_text(json.dumps({'schema_version':1,'name':cell+'_v1','model_family':'recurrent_depth','config':asdict(c)},indent=2)+'\n')
                summary=str(ROOT/(cell+'.summary.json'))
            entries.append({'cell':cell,'objective':objective,'seed':seed,'model_family':'recurrent_depth','reused':reused,
                'config':str(path),'config_sha256':file_hash(path),'run_directory':str(directory),'summary_path':summary,
                'selection_readouts':['confidence'],'reference_control':control if reused else None,
                'control_run_manifest_sha256':file_hash(Path(control['run_directory'])/'research_run.json') if reused else None})
    registry={'schema_version':1,'name':'recurrent_temporal_v1','role':'paired development objective control; no test evaluation',
        'entries':entries,'seeds':[23,29,31],'train_path':'research/data/ordered_pointer_v1/pointer/train.jsonl',
        'validation_path':'research/data/ordered_pointer_v1/pointer/validation.jsonl','training_source_sha256':sources,
        'control_source_archive':str(previous/'pre_run_source.zip'),'control_source_archive_sha256':file_hash(previous/'pre_run_source.zip'),
        'control_checkpoint_freeze':str(previous/'checkpoints.json'),'control_checkpoint_freeze_sha256':file_hash(previous/'checkpoints.json'),
        'gpu_queues':{'cuda:0':['uniform_seed23','dynamic_aggregate_seed29','uniform_seed31'],
                      'cuda:1':['dynamic_aggregate_seed23','uniform_seed29','dynamic_aggregate_seed31']}}
    for key in ('train','validation'):registry[key+'_sha256']=file_hash(registry[key+'_path'])
    (ROOT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    files=dict(sources)
    for p in ('scripts/run_recurrent_temporal.py','scripts/prepare_recurrent_temporal.py','scripts/profile_recurrent_temporal.py',
              'tests/test_recurrent_temporal.py',str(ROOT/'registry.json'),str(ROOT/'PLAN_BEFORE_RUNS.md'),str(ROOT/'checks.xml'),str(ROOT/'profile.json')):
        files[p]=file_hash(p)
    for e in entries:files[e['config']]=file_hash(e['config'])
    with zipfile.ZipFile(ROOT/'pre_run_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):archive.write(p,p)
    (ROOT/'pre_run_source.json').write_text(json.dumps({'files':files,'archive_sha256':file_hash(ROOT/'pre_run_source.zip')},indent=2)+'\n')
    print('Frozen six new training cells and three reused final-CE controls; no test forwards.')

if __name__=='__main__':main()
