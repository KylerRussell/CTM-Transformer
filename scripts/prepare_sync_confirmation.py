"""Freeze the locked confirmation of second-order query features (five cells x seven new seeds)."""
import argparse,json,shutil,zipfile
from dataclasses import asdict,replace
from pathlib import Path
import xml.etree.ElementTree as ET
from ctm_transformer.dense_experiment import SOURCES
from ctm_transformer.experiment import file_hash
from ctm_transformer.group_suite import validate_group_suite
from ctm_transformer.research import load_research_config

ROOT=Path('research/results/sync_confirmation_v1');DATA=Path('research/data/sync_confirmation_v1')
RUNS=Path('research/runs/sync_confirmation_v1');CONFIGS=Path('research/configs/sync_confirmation_v1')
SEEDS=(97,101,103,107,109,113,127)
RECIPES={'transformer':'research/configs/presentation_control_v1/transformer_shuffled_seed23.json',
         'ctm':'research/configs/presentation_control_v1/ctm_shuffled_seed23.json',
         'recurrent_depth':'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'}
# cell -> (family, factory kind, factory argument, config overrides, readout, evaluation tick budgets, estimated minutes)
RECURRENT_TICKS=[4,8,16,32]
CELLS={'transformer':('transformer',None,None,{},'final',[1],5),
       'rdt':('recurrent_depth','sync_rdt','rdt',{},'confidence',RECURRENT_TICKS,55),
       'rdt_wide':('recurrent_depth','sync_rdt','rdt',{'d_model':100,'ffn_hidden_dim':297},'confidence',RECURRENT_TICKS,57),
       'sync':('recurrent_depth','sync_rdt','sync',{},'confidence',RECURRENT_TICKS,62),
       'current':('recurrent_depth','sync_ablation','current',{},'confidence',RECURRENT_TICKS,62)}
STEPS,WARMUP,EVAL_INTERVAL,LR=10000,100,250,0.0003
SLOTS=['cuda:0#0','cuda:0#1','cuda:0#2','cuda:1#3','cuda:1#4','cuda:1#5']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checks',type=Path,required=True,help='JUnit XML of the passed study checks')
    a=p.parse_args()
    assert not (ROOT/'registry.json').exists()
    suites=list(ET.parse(a.checks).getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)>0
    assert all(int(s.attrib['failures'])==int(s.attrib['errors'])==int(s.attrib.get('skipped',0))==0 for s in suites)
    manifest=validate_group_suite(DATA)
    assert [int(s) for s in manifest['seeds']]==list(SEEDS) and manifest['group']=='S3'
    ROOT.mkdir(parents=True,exist_ok=True);CONFIGS.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.checks,ROOT/'checks.xml');shutil.copyfile('research/SYNC_CONFIRMATION.md',ROOT/'PLAN_BEFORE_RUNS.md')
    sources={p:file_hash(p) for p in SOURCES+['ctm_transformer/sync_rdt.py','ctm_transformer/sync_ablations.py','ctm_transformer/group_word.py','ctm_transformer/group_suite.py']}
    entries=[]
    for seed in SEEDS:
        for cell,(family,kind,arg,overrides,readout,ticks,minutes) in CELLS.items():
            name=f'{cell}_seed{seed}';directory=RUNS/cell/f'seed{seed}'
            c,_=load_research_config(RECIPES[family])
            c=replace(c,learning_rate=LR,max_steps=STEPS,warmup_steps=WARMUP,eval_interval=EVAL_INTERVAL,log_interval=500,
                      checkpoint_dir=str(directory),**overrides)
            path=CONFIGS/f'{name}.json'
            path.write_text(json.dumps({'schema_version':1,'name':name+'_v1','model_family':family,'config':asdict(c)},indent=2)+'\n')
            split=manifest['seeds'][str(seed)]
            entries.append({'cell':name,'study_cell':cell,'model_family':family,'factory_kind':kind,'factory_arg':arg,
                'config_overrides':overrides,'seed':seed,'config':str(path),'config_sha256':file_hash(path),'recipe_source':RECIPES[family],
                'train_sha256':split['train']['sha256'],'validation_sha256':split['validation']['sha256'],'evaluation_sha256':split['evaluation']['sha256'],
                'train_examples':split['train']['examples'],'run_directory':str(directory),'summary_path':str(ROOT/f'{name}.summary.json'),
                'selection_readouts':[readout],'readout':readout,'trained_thought_steps':c.max_thought_steps,'evaluation_thought_steps':ticks,
                'estimated_minutes':minutes})
    assert all(e['train_examples']==STEPS*32 for e in entries)  # Fresh training words, one presentation each.
    # Longest-first assignment to the least-loaded slot balances very unequal run costs (5 to 115 minutes).
    queues={s:[] for s in SLOTS};load={s:0 for s in SLOTS}
    for e in sorted(entries,key=lambda e:(-e['estimated_minutes'],e['seed'],e['cell'])):
        slot=min(SLOTS,key=lambda s:(load[s],SLOTS.index(s)));queues[slot].append(e['cell']);load[slot]+=e['estimated_minutes']
    registry={'schema_version':1,'name':'sync_confirmation_v1','runner':'shared_research_v3','entries':entries,'seeds':list(SEEDS),
        'cells':{c:{'family':v[0],'factory_kind':v[1],'factory_arg':v[2],'config_overrides':v[3],'readout':v[4],'evaluation_thought_steps':v[5]}
                 for c,v in CELLS.items()},
        'role':'locked confirmation; the evaluation split is the locked test set, evaluated once after the checkpoint freeze',
        'dataset':str(DATA),'dataset_manifest_sha256':file_hash(DATA/'manifest.json'),'training_source_sha256':sources,
        'primary_checkpoint_step':STEPS,'warmup_steps':WARMUP,'eval_interval':EVAL_INTERVAL,'learning_rate':LR,
        'train_max_len':16,'evaluation_length':32,'chance':1/6,'gpu_queues':queues,'estimated_slot_minutes':load,
        'evaluation_queues':{'cuda:0':[e['cell'] for e in entries if e['seed'] in (97,101,103,107)],'cuda:1':[e['cell'] for e in entries if e['seed'] in (109,113,127)]}}
    (ROOT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    files=dict(sources)
    for p in ('scripts/prepare_sync_confirmation.py','scripts/run_sync_confirmation.py','scripts/freeze_sync_confirmation.py','scripts/evaluate_sync_confirmation.py',
              'scripts/summarize_sync_confirmation.py','scripts/study_supervisor.py','scripts/supervise_sync_confirmation.py','scripts/summarize_group_s3.py',
              'tests/test_group_word.py','tests/test_group_s3.py','tests/test_sync_rdt.py','tests/test_sync_ablations.py','tests/test_sync_confirmation.py',
              str(ROOT/'registry.json'),str(ROOT/'PLAN_BEFORE_RUNS.md'),str(ROOT/'checks.xml'),str(DATA/'manifest.json')):
        files[p]=file_hash(p)
    for e in entries:files[e['config']]=file_hash(e['config'])
    large=set()
    for seed,splits in manifest['seeds'].items():
        for split,entry in splits.items():
            files[str(DATA/entry['path'])]=entry['sha256']
            if split=='train':large.add(str(DATA/entry['path']))
    with zipfile.ZipFile(ROOT/'pre_run_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            if path not in large:z.write(path,path)
    (ROOT/'pre_run_source.json').write_text(json.dumps({'files':files,'hashed_not_archived':sorted(large),
        'archive_sha256':file_hash(ROOT/'pre_run_source.zip')},indent=2)+'\n')
    print(f'Frozen {len(entries)} runs; estimated slot minutes {load}.')

if __name__=='__main__':main()
