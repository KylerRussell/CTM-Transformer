"""Freeze the fresh-map pointer study: data, configs, registry and pre-run sources."""
import argparse,json,shutil,zipfile
from dataclasses import asdict,replace
from pathlib import Path
import xml.etree.ElementTree as ET
from ctm_transformer.experiment import file_hash
from ctm_transformer.online_pointer import create_online_pointer_suite,validate_online_pointer_suite
from ctm_transformer.research import load_research_config

ROOT=Path('research/results/online_pointer_v1');DATA=Path('research/data/online_pointer_v1')
RUNS=Path('research/runs/online_pointer_v1');CONFIGS=Path('research/configs/online_pointer_v1')
SOURCE=Path('research/results/presentation_control_v1')
SEEDS=(23,29,31);FAMILIES=('ctm','transformer','recurrent_depth')
CONDITIONS={'repeated':'train_repeated_onehop','fresh':'train_fresh_onehop','multihop':'train_fresh_multihop'}
TICK_SWEEP=[4,8,16,32]
# Balanced by measured per-update cost; one-hop cells first so the data-regime contrast finishes earliest.
QUEUES={
    'cuda:0':['transformer_fresh_seed23','transformer_fresh_seed29','transformer_fresh_seed31',
              'transformer_repeated_seed23','transformer_repeated_seed29','transformer_repeated_seed31',
              'ctm_fresh_seed23','ctm_repeated_seed23','ctm_fresh_seed31','ctm_repeated_seed31',
              'recurrent_depth_fresh_seed31','recurrent_depth_repeated_seed31',
              'transformer_multihop_seed23','transformer_multihop_seed29','transformer_multihop_seed31',
              'ctm_multihop_seed23','recurrent_depth_multihop_seed31'],
    'cuda:1':['ctm_fresh_seed29','ctm_repeated_seed29',
              'recurrent_depth_fresh_seed23','recurrent_depth_repeated_seed23',
              'recurrent_depth_fresh_seed29','recurrent_depth_repeated_seed29',
              'ctm_multihop_seed29','ctm_multihop_seed31','recurrent_depth_multihop_seed23','recurrent_depth_multihop_seed29'],
}


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checks',type=Path,required=True,help='JUnit XML of the passed study checks')
    a=p.parse_args()
    assert not (ROOT/'registry.json').exists()
    suites=list(ET.parse(a.checks).getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)>0
    assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==int(s.attrib.get('skipped',0))==0 for s in suites)
    ROOT.mkdir(parents=True,exist_ok=True)
    create_online_pointer_suite(DATA);validate_online_pointer_suite(DATA)
    shutil.copyfile(a.checks,ROOT/'checks.xml');shutil.copyfile('research/ONLINE_POINTER.md',ROOT/'PLAN_BEFORE_RUNS.md')
    source=read(SOURCE/'registry.json');sources=source['training_source_sha256']
    for path,h in sources.items():assert file_hash(path)==h,path
    CONFIGS.mkdir(parents=True,exist_ok=True);manifest=read(DATA/'manifest.json')['tasks']['pointer'];entries=[]
    for seed in SEEDS:
        for family in FAMILIES:
            recipe=next(e for e in source['entries'] if e['cell']==f'{family}_shuffled_seed{seed}')
            assert file_hash(recipe['config'])==recipe['config_sha256']
            for condition,split in CONDITIONS.items():
                cell=f'{family}_{condition}_seed{seed}';directory=RUNS/condition/f'seed{seed}'/family
                c,_=load_research_config(recipe['config']);c=replace(c,checkpoint_dir=str(directory))
                path=CONFIGS/f'{cell}.json'
                path.write_text(json.dumps({'schema_version':1,'name':cell+'_v1','model_family':family,'config':asdict(c)},indent=2)+'\n')
                train=DATA/manifest[split]['path']
                entries.append({'cell':cell,'seed':seed,'model_family':family,'condition':condition,'reused':False,
                    'train_split':split,'train_path':str(train),'train_sha256':file_hash(train),
                    'config':str(path),'config_sha256':file_hash(path),'recipe_source':recipe['config'],
                    'run_directory':str(directory),'summary_path':str(ROOT/f'{cell}.summary.json'),
                    'selection_readouts':recipe['selection_readouts'],'readout':recipe['selection_readouts'][0],
                    'trained_thought_steps':c.max_thought_steps,
                    'evaluation_thought_steps':[1] if family=='transformer' else TICK_SWEEP})
    cells=[e['cell'] for e in entries];queued=[c for q in QUEUES.values() for c in q]
    assert sorted(queued)==sorted(cells) and len(set(queued))==len(cells)==27
    registry={'schema_version':1,'name':'online_pointer_v1','entries':entries,'seeds':list(SEEDS),
        'role':'development study of training-data regime and multi-hop depth; fixed-update primary endpoint; no test set',
        'dataset':str(DATA),'dataset_manifest_sha256':file_hash(DATA/'manifest.json'),
        'validation_path':str(DATA/manifest['validation']['path']),'validation_sha256':file_hash(DATA/manifest['validation']['path']),
        'evaluation_splits':['eval_id','eval_depth'],'training_source_sha256':sources,'primary_checkpoint_step':3000,
        'chance_accuracy':1/11,'gpu_queues':QUEUES,
        # CTM evaluation (four tick budgets) is the slowest; split it across both GPUs.
        'evaluation_queues':{'cuda:0':[c for c in cells if c.startswith(('ctm',)) and c.endswith(('seed23','seed31'))]+[c for c in cells if c.startswith('transformer')],
                             'cuda:1':[c for c in cells if c.startswith('ctm') and c.endswith('seed29')]+[c for c in cells if c.startswith('recurrent_depth')]}}
    assert sorted(c for q in registry['evaluation_queues'].values() for c in q)==sorted(cells)
    (ROOT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    files=dict(sources)
    for p in ('ctm_transformer/online_pointer.py','ctm_transformer/pointer_diagnostics.py','scripts/prepare_online_pointer.py',
              'scripts/run_online_pointer.py','scripts/run_registry_trials.py','scripts/freeze_online_pointer.py',
              'scripts/evaluate_online_pointer.py','scripts/summarize_online_pointer.py','scripts/study_supervisor.py',
              'scripts/supervise_online_pointer.py','tests/test_online_pointer.py','scripts/eval_harness.py',
              str(ROOT/'registry.json'),str(ROOT/'PLAN_BEFORE_RUNS.md'),str(ROOT/'checks.xml'),str(DATA/'manifest.json')):
        files[p]=file_hash(p)
    for e in entries:files[e['config']]=file_hash(e['config'])
    large=set(e['train_path'] for e in entries if e['condition']!='repeated')
    for path in sorted(DATA.rglob('*.jsonl')):files[str(path)]=file_hash(path)
    # The two 96,000-example training files are regenerable from the manifest seed; hash them but keep them out of the archive.
    with zipfile.ZipFile(ROOT/'pre_run_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            if path not in large:z.write(path,path)
    (ROOT/'pre_run_source.json').write_text(json.dumps({'files':files,'hashed_not_archived':sorted(large),
        'archive_sha256':file_hash(ROOT/'pre_run_source.zip')},indent=2)+'\n')
    print(f'Frozen {len(entries)} runs across {len(CONDITIONS)} conditions.')

if __name__=='__main__':main()
