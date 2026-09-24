"""Freeze the Transformer optimizer/budget calibration on fresh 12-node pointer maps."""
import argparse,json,shutil,zipfile
from dataclasses import asdict,replace
from pathlib import Path
import xml.etree.ElementTree as ET
from ctm_transformer.experiment import file_hash
from ctm_transformer.pointer_calibration import CALIBRATION_DESIGN,validate_pointer_suite
from ctm_transformer.research import load_research_config

ROOT=Path('research/results/pointer_calibration_v1');DATA=Path('research/data/pointer_calibration_v1')
RUNS=Path('research/runs/pointer_calibration_v1');CONFIGS=Path('research/configs/pointer_calibration_v1')
RECIPE='research/configs/presentation_control_v1/transformer_shuffled_seed23.json'
SEEDS=(41,43);LRS=(0.0003,0.001,0.003);TASKS={'onehop':'train_fresh_onehop','multihop':'train_fresh_multihop'}
STEPS,WARMUP,EVAL_INTERVAL=30000,300,1000


def lr_name(lr):return f'lr{lr:.0e}'.replace('e-0','e-')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checks',type=Path,required=True,help='JUnit XML of the passed study checks')
    a=p.parse_args()
    assert not (ROOT/'registry.json').exists()
    suites=list(ET.parse(a.checks).getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)>0
    assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==int(s.attrib.get('skipped',0))==0 for s in suites)
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert manifest['seed']==20260926 and manifest['design']=={s:[n,h] for s,(n,h) in CALIBRATION_DESIGN.items()}
    validate_pointer_suite(DATA)
    ROOT.mkdir(parents=True,exist_ok=True);CONFIGS.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.checks,ROOT/'checks.xml');shutil.copyfile('research/POINTER_CALIBRATION.md',ROOT/'PLAN_BEFORE_RUNS.md')
    sources=json.loads(Path('research/results/presentation_control_v1/registry.json').read_text())['training_source_sha256']
    for path,h in sources.items():assert file_hash(path)==h,path
    splits=manifest['tasks']['pointer'];entries=[]
    for seed in SEEDS:
        for task,split in TASKS.items():
            for lr in LRS:
                cell=f'transformer_{task}_{lr_name(lr)}_seed{seed}';directory=RUNS/task/lr_name(lr)/f'seed{seed}'
                c,_=load_research_config(RECIPE)
                c=replace(c,learning_rate=lr,max_steps=STEPS,warmup_steps=WARMUP,eval_interval=EVAL_INTERVAL,checkpoint_dir=str(directory))
                path=CONFIGS/f'{cell}.json'
                path.write_text(json.dumps({'schema_version':1,'name':cell+'_v1','model_family':'transformer','config':asdict(c)},indent=2)+'\n')
                train=DATA/splits[split]['path']
                entries.append({'cell':cell,'seed':seed,'model_family':'transformer','task':task,'learning_rate':lr,'reused':False,
                    'train_split':split,'train_path':str(train),'train_sha256':splits[split]['sha256'],'train_examples':splits[split]['examples'],
                    'config':str(path),'config_sha256':file_hash(path),'recipe_source':RECIPE,
                    'run_directory':str(directory),'summary_path':str(ROOT/f'{cell}.summary.json'),
                    'selection_readouts':['final'],'readout':'final','trained_thought_steps':1,'evaluation_thought_steps':[1]})
    assert all(e['train_examples']==STEPS*32 for e in entries)  # Every map is presented exactly once.
    by_seed={s:[e['cell'] for e in entries if e['seed']==s] for s in SEEDS}
    registry={'schema_version':1,'name':'pointer_calibration_v1','entries':entries,'seeds':list(SEEDS),
        'role':'Stage 1a Transformer optimizer/budget calibration; development only; no test set',
        'dataset':str(DATA),'dataset_manifest_sha256':file_hash(DATA/'manifest.json'),
        'validation_path':str(DATA/splits['validation']['path']),'validation_sha256':splits['validation']['sha256'],
        'evaluation_splits':['eval_id','eval_depth'],'training_source_sha256':sources,'primary_checkpoint_step':STEPS,
        'warmup_steps':WARMUP,'eval_interval':EVAL_INTERVAL,'chance_accuracy':1/11,
        'gpu_queues':{'cuda:0':by_seed[41],'cuda:1':by_seed[43]},'evaluation_queues':{'cuda:0':by_seed[41],'cuda:1':by_seed[43]}}
    (ROOT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    files=dict(sources)
    for p in ('ctm_transformer/pointer_calibration.py','ctm_transformer/online_pointer.py','scripts/prepare_pointer_calibration.py',
              'scripts/run_pointer_calibration.py','scripts/run_registry_trials.py','scripts/freeze_pointer_calibration.py',
              'scripts/evaluate_pointer_calibration.py','scripts/summarize_pointer_calibration.py','scripts/study_supervisor.py',
              'scripts/supervise_pointer_calibration.py','tests/test_pointer_calibration.py','scripts/eval_harness.py',
              str(ROOT/'registry.json'),str(ROOT/'PLAN_BEFORE_RUNS.md'),str(ROOT/'checks.xml'),str(DATA/'manifest.json')):
        files[p]=file_hash(p)
    for e in entries:files[e['config']]=file_hash(e['config'])
    large={str(DATA/splits[s]['path']) for s in TASKS.values()}
    for path in sorted(DATA.rglob('*.jsonl')):files[str(path)]=file_hash(path)
    # The two 960,000-example training files are regenerable from the manifest seed; hash them but keep them out of the archive.
    with zipfile.ZipFile(ROOT/'pre_run_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            if path not in large:z.write(path,path)
    (ROOT/'pre_run_source.json').write_text(json.dumps({'files':files,'hashed_not_archived':sorted(large),
        'archive_sha256':file_hash(ROOT/'pre_run_source.zip')},indent=2)+'\n')
    print(f'Frozen {len(entries)} Transformer calibration runs.')

if __name__=='__main__':main()
