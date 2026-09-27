"""Freeze Sync-RDT ablation A7 (self-pairs only; five seeds) on the S3 study's data."""
import argparse,json,shutil,zipfile
from dataclasses import asdict,replace
from pathlib import Path
import xml.etree.ElementTree as ET
from ctm_transformer.dense_experiment import SOURCES
from ctm_transformer.experiment import file_hash
from ctm_transformer.group_suite import validate_group_suite
from ctm_transformer.research import load_research_config

ROOT=Path('research/results/sync_ablation_v2');DATA=Path('research/data/group_s3_v1');CONTROL=Path('research/results/group_s3_v1')
RUNS=Path('research/runs/sync_ablation_v2');CONFIGS=Path('research/configs/sync_ablation_v2')
RECIPE='research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'
SEEDS=(47,53,59,61,67);ABLATIONS=('self',);CONTROLS=('sync','rdt','rdt_wide')
STEPS,WARMUP,EVAL_INTERVAL,LR=10000,100,250,0.0003
SLOTS=['cuda:0#0','cuda:0#1','cuda:0#2','cuda:1#3','cuda:1#4','cuda:1#5']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checks',type=Path,required=True,help='JUnit XML of the passed study checks')
    a=p.parse_args()
    assert not (ROOT/'registry.json').exists()
    suites=list(ET.parse(a.checks).getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)>0
    assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==int(s.attrib.get('skipped',0))==0 for s in suites)
    manifest=validate_group_suite(DATA);assert [int(s) for s in manifest['seeds']]==list(SEEDS)
    control_registry=json.loads((CONTROL/'registry.json').read_text());control_summary=json.loads((CONTROL/'summary.json').read_text())
    assert control_registry['dataset_manifest_sha256']==file_hash(DATA/'manifest.json')
    controls={}
    for cell in CONTROLS:
        for seed in SEEDS:
            name=f'{cell}_seed{seed}';ev=control_summary['evaluations'][name]
            assert file_hash(ev['path'])==ev['sha256'];controls[name]={'evaluation':ev['path'],'sha256':ev['sha256']}
    ROOT.mkdir(parents=True,exist_ok=True);CONFIGS.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.checks,ROOT/'checks.xml');shutil.copyfile('research/SYNC_ABLATION_A7.md',ROOT/'PLAN_BEFORE_RUNS.md')
    sources={p:file_hash(p) for p in SOURCES+['ctm_transformer/sync_rdt.py','ctm_transformer/sync_ablations.py','ctm_transformer/sync_ablation_self.py','ctm_transformer/group_word.py','ctm_transformer/group_suite.py']}
    entries=[]
    for seed in SEEDS:
        for kind in ABLATIONS:
            name=f'{kind}_seed{seed}';directory=RUNS/kind/f'seed{seed}'
            c,_=load_research_config(RECIPE)
            c=replace(c,learning_rate=LR,max_steps=STEPS,warmup_steps=WARMUP,eval_interval=EVAL_INTERVAL,log_interval=500,checkpoint_dir=str(directory))
            path=CONFIGS/f'{name}.json'
            path.write_text(json.dumps({'schema_version':1,'name':name+'_v1','model_family':'recurrent_depth','config':asdict(c)},indent=2)+'\n')
            split=manifest['seeds'][str(seed)]
            entries.append({'cell':name,'study_cell':kind,'model_family':'recurrent_depth','factory_kind':'self_pair','factory_arg':kind,
                'config_overrides':{},'seed':seed,'config':str(path),'config_sha256':file_hash(path),'recipe_source':RECIPE,
                'train_sha256':split['train']['sha256'],'validation_sha256':split['validation']['sha256'],'evaluation_sha256':split['evaluation']['sha256'],
                'train_examples':split['train']['examples'],'run_directory':str(directory),'summary_path':str(ROOT/f'{name}.summary.json'),
                'selection_readouts':['confidence'],'readout':'confidence','trained_thought_steps':16,'evaluation_thought_steps':[4,8,16,32]})
    # Seed j runs ablation k on slot (k + j) mod 6: every slot mixes ablations and seeds.
    queues={s:[] for s in SLOTS}
    for i,e in enumerate(entries):queues[SLOTS[i%len(SLOTS)]].append(e['cell'])
    registry={'schema_version':1,'name':'sync_ablation_v2','runner':'shared_research_v3','entries':entries,'seeds':list(SEEDS),
        'ablations':list(ABLATIONS),'controls':controls,'control_study':str(CONTROL),
        'control_checkpoints_sha256':file_hash(CONTROL/'checkpoints.json'),'control_summary_sha256':file_hash(CONTROL/'summary.json'),
        'role':'development ablation A7 (self-pairs) of the sync cell on the S3 study data; paired with frozen controls; no test set',
        'dataset':str(DATA),'dataset_manifest_sha256':file_hash(DATA/'manifest.json'),'training_source_sha256':sources,
        'primary_checkpoint_step':STEPS,'warmup_steps':WARMUP,'eval_interval':EVAL_INTERVAL,'learning_rate':LR,
        'train_max_len':16,'evaluation_length':32,'chance':1/6,'gpu_queues':queues,
        'evaluation_queues':{'cuda:0':[e['cell'] for e in entries if e['seed'] in (47,53,59)],'cuda:1':[e['cell'] for e in entries if e['seed'] in (61,67)]}}
    (ROOT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    files=dict(sources)
    for p in ('scripts/prepare_sync_ablation_v2.py','scripts/run_sync_ablation_v2.py','scripts/freeze_sync_ablation_v2.py','scripts/evaluate_sync_ablation_v2.py',
              'scripts/summarize_sync_ablation_v2.py','scripts/study_supervisor.py','scripts/supervise_sync_ablation_v2.py',
              'tests/test_sync_ablations.py','tests/test_sync_ablation_self.py','tests/test_sync_rdt.py','tests/test_group_s3.py',
              str(ROOT/'registry.json'),str(ROOT/'PLAN_BEFORE_RUNS.md'),str(ROOT/'checks.xml'),str(DATA/'manifest.json'),
              str(CONTROL/'checkpoints.json'),str(CONTROL/'summary.json')):
        files[p]=file_hash(p)
    for e in entries:files[e['config']]=file_hash(e['config'])
    with zipfile.ZipFile(ROOT/'pre_run_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            if not path.endswith('train.jsonl'):z.write(path,path)
    (ROOT/'pre_run_source.json').write_text(json.dumps({'files':files,'archive_sha256':file_hash(ROOT/'pre_run_source.zip')},indent=2)+'\n')
    print(f'Frozen {len(entries)} A7 runs paired with {len(controls)} frozen control runs.')

if __name__=='__main__':main()
