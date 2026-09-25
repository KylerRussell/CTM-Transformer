"""Freeze the Sync-RDT retrieval sample-efficiency study (six cells x five seeds)."""
import argparse,json,shutil,zipfile
from dataclasses import asdict,replace
from pathlib import Path
import xml.etree.ElementTree as ET
from ctm_transformer.dense_experiment import SOURCES
from ctm_transformer.experiment import file_hash
from ctm_transformer.mqar_suite import validate_mqar_suite
from ctm_transformer.research import load_research_config

ROOT=Path('research/results/syncrdt_retrieval_v1');DATA=Path('research/data/syncrdt_retrieval_v1')
RUNS=Path('research/runs/syncrdt_retrieval_v1');CONFIGS=Path('research/configs/syncrdt_retrieval_v1')
RECIPE='research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'
SEEDS=(71,73,79,83,89)
# cell name -> (Sync-RDT factory cell, config overrides)
CELLS={'rdt':('rdt',{}),'history':('history',{}),'sync':('sync',{}),'sync_rdt':('sync_rdt',{}),
       'rdt_wide':('rdt',{'d_model':100,'ffn_hidden_dim':297}),'history_lowgate':('history_lowgate',{})}
STEPS,WARMUP,EVAL_INTERVAL,LR=10000,100,250,0.0003
SLOTS=['cuda:0#0','cuda:0#1','cuda:0#2','cuda:1#3','cuda:1#4','cuda:1#5']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checks',type=Path,required=True,help='JUnit XML of the passed study checks')
    a=p.parse_args()
    assert not (ROOT/'registry.json').exists()
    suites=list(ET.parse(a.checks).getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)>0
    assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==int(s.attrib.get('skipped',0))==0 for s in suites)
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert [int(s) for s in manifest['seeds']]==list(SEEDS)
    validate_mqar_suite(DATA)
    ROOT.mkdir(parents=True,exist_ok=True);CONFIGS.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.checks,ROOT/'checks.xml');shutil.copyfile('research/SYNC_RDT_RETRIEVAL.md',ROOT/'PLAN_BEFORE_RUNS.md')
    sources={p:file_hash(p) for p in SOURCES+['ctm_transformer/sync_rdt.py','ctm_transformer/pointer_probes.py','ctm_transformer/mqar_suite.py']}
    entries=[]
    for seed in SEEDS:
        for cell,(factory_cell,overrides) in CELLS.items():
            name=f'{cell}_seed{seed}';directory=RUNS/cell/f'seed{seed}'
            c,_=load_research_config(RECIPE)
            c=replace(c,learning_rate=LR,max_steps=STEPS,warmup_steps=WARMUP,eval_interval=EVAL_INTERVAL,log_interval=500,
                      checkpoint_dir=str(directory),**overrides)
            path=CONFIGS/f'{name}.json'
            path.write_text(json.dumps({'schema_version':1,'name':name+'_v1','model_family':'recurrent_depth','config':asdict(c)},indent=2)+'\n')
            split=manifest['seeds'][str(seed)]
            entries.append({'cell':name,'study_cell':cell,'factory_cell':factory_cell,'config_overrides':overrides,'seed':seed,
                'model_family':'recurrent_depth','config':str(path),'config_sha256':file_hash(path),'recipe_source':RECIPE,
                'train_sha256':split['train']['sha256'],'validation_sha256':split['validation']['sha256'],
                'evaluation_sha256':split['evaluation']['sha256'],'train_maps':split['train']['maps'],
                'run_directory':str(directory),'summary_path':str(ROOT/f'{name}.summary.json'),
                'selection_readouts':['confidence'],'readout':'confidence','trained_thought_steps':16})
    assert all(e['train_maps']==STEPS*32 for e in entries)  # Every training map is presented once.
    # Latin-square rotation: seed j runs cell c on slot (c + j) mod 6, so every slot runs every cell
    # once and co-scheduling is balanced across cells.
    cells=list(CELLS);queues={slot:[] for slot in SLOTS}
    for j,seed in enumerate(SEEDS):
        for c,cell in enumerate(cells):queues[SLOTS[(c+j)%len(SLOTS)]].append(f'{cell}_seed{seed}')
    registry={'schema_version':1,'name':'syncrdt_retrieval_v1','runner':'shared_research_v3','entries':entries,'seeds':list(SEEDS),
        'cells':{c:{'factory_cell':f,'config_overrides':o} for c,(f,o) in CELLS.items()},
        'role':'development sample-efficiency study; fixed-update endpoints; no test set',
        'dataset':str(DATA),'dataset_manifest_sha256':file_hash(DATA/'manifest.json'),'training_source_sha256':sources,
        'primary_checkpoint_step':STEPS,'warmup_steps':WARMUP,'eval_interval':EVAL_INTERVAL,'learning_rate':LR,
        'chance_position1':1/11,'thresholds':[0.5,0.9],'gpu_queues':queues,
        'evaluation_queues':{'cuda:0':[e['cell'] for e in entries[:15]],'cuda:1':[e['cell'] for e in entries[15:]]}}
    (ROOT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    files=dict(sources)
    for p in ('scripts/prepare_syncrdt_retrieval.py','scripts/run_syncrdt_retrieval.py','scripts/freeze_syncrdt_retrieval.py',
              'scripts/evaluate_syncrdt_retrieval.py','scripts/summarize_syncrdt_retrieval.py','scripts/study_supervisor.py',
              'scripts/supervise_syncrdt_retrieval.py','tests/test_sync_rdt.py','tests/test_syncrdt_retrieval.py',
              str(ROOT/'registry.json'),str(ROOT/'PLAN_BEFORE_RUNS.md'),str(ROOT/'checks.xml'),str(DATA/'manifest.json')):
        files[p]=file_hash(p)
    for e in entries:files[e['config']]=file_hash(e['config'])
    large=set()
    for seed,splits in manifest['seeds'].items():
        for split,entry in splits.items():
            files[str(DATA/entry['path'])]=entry['sha256']
            if split=='train':large.add(str(DATA/entry['path']))
    # The five 320,000-map training files are regenerable from the manifest; hash them but keep them out of the archive.
    with zipfile.ZipFile(ROOT/'pre_run_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            if path not in large:z.write(path,path)
    (ROOT/'pre_run_source.json').write_text(json.dumps({'files':files,'hashed_not_archived':sorted(large),
        'archive_sha256':file_hash(ROOT/'pre_run_source.zip')},indent=2)+'\n')
    print(f'Frozen {len(entries)} runs: {len(CELLS)} cells x {len(SEEDS)} seeds.')

if __name__=='__main__':main()
