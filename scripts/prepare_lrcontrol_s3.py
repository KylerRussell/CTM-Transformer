"""Freeze the S3 learning-rate control: fixed-depth RoPE RDT at the randomized-depth cell's learning rate (1e-3), on the recipe confirmation's seeds and data."""
import argparse,json,shutil,zipfile
from dataclasses import asdict,replace
from pathlib import Path
import xml.etree.ElementTree as ET
from ctm_transformer.dense_experiment_v5 import SOURCES
from ctm_transformer.experiment import file_hash
from ctm_transformer.group_suite import validate_group_suite
from ctm_transformer.research import load_research_config
from ctm_transformer.ctm_lm import lm_config
from scripts.create_recipe_s3_data import ROOT as DATA_ROOT,SEEDS

ROOT=Path('research/results/lrcontrol_s3_v1');DATA=Path(DATA_ROOT)  # the recipe confirmation's seeds and data
CONTROL=Path('research/results/recipe_s3_v1')
RUNS=Path('research/runs/lrcontrol_s3_v1');CONFIGS=Path('research/configs/lrcontrol_s3_v1')
RECIPES={'transformer':'research/configs/presentation_control_v1/transformer_shuffled_seed23.json',
         'ctm':'research/configs/presentation_control_v1/ctm_shuffled_seed23.json',
         'recurrent_depth':'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'}
RECURRENT_TICKS=[4,8,16,32,48,64]
SAMPLER={'kind':'lognormal_poisson','mean':15,'sigma':0.5,'maximum':48}  # expected depth about 16, as fixed T=16
NO_TABLE={'use_positional_encoding':False}  # RoPE replaces the learned absolute table
# cell -> (family, factory argument, learning rate, depth sampler, readout, evaluation tick budgets, estimated minutes)
# The one new cell: fixed depth at the learning rate of the recipe confirmation's rdt_rope_rand cell.
CELLS={'rdt_rope_fixed_lr1e3':('recurrent_depth','rdt,rope',1e-3,None,'confidence',RECURRENT_TICKS,75)}
STEPS,WARMUP,EVAL_INTERVAL=10000,100,250
SLOTS=['cuda:0#0','cuda:0#1','cuda:1#3','cuda:1#4']  # the other slots run development probes
STUDY_FILES=('scripts/create_recipe_s3_data.py','scripts/prepare_lrcontrol_s3.py','scripts/run_lrcontrol_s3.py','scripts/freeze_lrcontrol_s3.py',
    'scripts/evaluate_lrcontrol_s3.py','scripts/summarize_lrcontrol_s3.py','scripts/study_supervisor.py','scripts/supervise_lrcontrol_s3.py','scripts/summarize_recipe_s3.py',
    'research/results/recipe_s3_v1/registry.json','research/results/recipe_s3_v1/summary.json','tests/test_lrcontrol_s3.py',
    'scripts/summarize_group_s3.py','scripts/summarize_reliability_s3.py','ctm_transformer/ctm_variants.py','ctm_transformer/sync_rdt.py',
    'tests/test_positions.py','tests/test_curriculum.py','tests/test_depth_sampling.py','tests/test_ctm_lm.py','tests/test_group_word.py',
    'tests/test_group_s3.py','tests/test_recipe_s3.py')


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
    shutil.copyfile(a.checks,ROOT/'checks.xml');shutil.copyfile('research/LRCONTROL_S3.md',ROOT/'PLAN_BEFORE_RUNS.md')
    sources={p:file_hash(p) for p in SOURCES+['ctm_transformer/ctm_lm.py','ctm_transformer/group_word.py','ctm_transformer/group_suite.py']}
    entries=[]
    for seed in SEEDS:
        for cell,(family,arg,lr,sampler,readout,ticks,minutes) in CELLS.items():
            name=f'{cell}_seed{seed}';directory=RUNS/cell/f'seed{seed}'
            c,_=load_research_config(RECIPES[family])
            if family=='ctm':c=lm_config(c)  # CTM's min-loss plus max-certainty loss, recorded in the config
            c=replace(c,learning_rate=lr,max_steps=STEPS,warmup_steps=WARMUP,eval_interval=EVAL_INTERVAL,log_interval=500,
                      checkpoint_dir=str(directory),**NO_TABLE)
            path=CONFIGS/f'{name}.json'
            path.write_text(json.dumps({'schema_version':1,'name':name+'_v1','model_family':family,'config':asdict(c)},indent=2)+'\n')
            split=manifest['seeds'][str(seed)]
            entries.append({'cell':name,'study_cell':cell,'model_family':family,'factory_kind':'position','factory_arg':arg,
                'config_overrides':NO_TABLE,'seed':seed,'config':str(path),'config_sha256':file_hash(path),'recipe_source':RECIPES[family],
                'train_sha256':split['train']['sha256'],'validation_sha256':split['validation']['sha256'],'evaluation_sha256':split['evaluation']['sha256'],
                'train_examples':split['train']['examples'],'run_directory':str(directory),'summary_path':str(ROOT/f'{name}.summary.json'),
                'selection_readouts':[readout],'readout':readout,'trained_thought_steps':c.max_thought_steps,'evaluation_thought_steps':ticks,
                'learning_rate':lr,'estimated_minutes':minutes,'depth_sampler':sampler})
    assert all(e['train_examples']==STEPS*32 for e in entries)  # Fresh training words, one presentation each.
    # Longest-first assignment to the least-loaded slot balances unequal run costs.
    queues={s:[] for s in SLOTS};load={s:0 for s in SLOTS}
    for e in sorted(entries,key=lambda e:(-e['estimated_minutes'],e['seed'],e['cell'])):
        slot=min(SLOTS,key=lambda s:(load[s],SLOTS.index(s)));queues[slot].append(e['cell']);load[slot]+=e['estimated_minutes']
    half=len(SEEDS)//2
    registry={'schema_version':1,'name':'lrcontrol_s3_v1','control_study':str(CONTROL),'control_registry_sha256':file_hash(CONTROL/'registry.json'),'control_summary_sha256':file_hash(CONTROL/'summary.json'),'runner':'shared_research_v5','entries':entries,'seeds':list(SEEDS),
        'cells':{c:{'family':v[0],'factory_kind':'position','factory_arg':v[1],'learning_rate':v[2],'depth_sampler':v[3],'readout':v[4],
                    'evaluation_thought_steps':v[5]} for c,v in CELLS.items()},
        'role':'learning-rate-matched fixed-depth control for the S3 recipe confirmation; evaluation split read once after the freeze',
        'dataset':str(DATA),'dataset_manifest_sha256':file_hash(DATA/'manifest.json'),'training_source_sha256':sources,
        'primary_checkpoint_step':STEPS,'warmup_steps':WARMUP,'eval_interval':EVAL_INTERVAL,
        'train_max_len':16,'evaluation_length':32,'chance':1/6,'gpu_queues':queues,'estimated_slot_minutes':load,
        'evaluation_queues':{'cuda:0':[e['cell'] for e in entries if e['seed'] in SEEDS[:half]],'cuda:1':[e['cell'] for e in entries if e['seed'] in SEEDS[half:]]}}
    (ROOT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    files=dict(sources)
    for p in (*STUDY_FILES,str(ROOT/'registry.json'),str(ROOT/'PLAN_BEFORE_RUNS.md'),str(ROOT/'checks.xml'),str(DATA/'manifest.json')):
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
