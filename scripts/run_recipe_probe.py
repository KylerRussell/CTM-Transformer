"""Run one S3 training-recipe development probe (development only; not a paper endpoint).

Varies the position scheme (learned table, RoPE, NoPE), the training depth
(fixed T=16 or log-normal Poisson with mean about 16), the learning rate and
the length curriculum, on the development suites research/data/recipe_dev_s3
and research/data/recipe_dev_a5 (seeds disjoint from every study). The frozen evaluator scores held-out
length-32 words at several inference depths; positions 17-32 measure length
extrapolation.
"""
import argparse,json,statistics
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.dense_experiment_v5 import train_dense_experiment
from ctm_transformer.dense_pointer import evaluate_dense
from ctm_transformer.depth_sampling import lognormal_poisson
from ctm_transformer.experiment import file_hash
from ctm_transformer.group_suite import create_group_suite,load_group_split
from ctm_transformer.research import build_model,load_research_config

RECIPES={'transformer':'research/configs/presentation_control_v1/transformer_shuffled_seed23.json',
         'ctm_lm':'research/configs/presentation_control_v1/ctm_shuffled_seed23.json',
         'rdt':'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'}
DATA={'S3':Path('research/data/recipe_dev_s3'),'A5':Path('research/data/recipe_dev_a5')};DEV_SEEDS=(307,311,313,317,331,337)
OUT=Path('research/results/recipe_probes');RUNS=Path('research/runs/recipe_probes')


def factory_for(model,positions):
    if positions!='learned':
        from ctm_transformer.positions import position_factory;return position_factory(model,positions)
    if model=='ctm_lm':
        from ctm_transformer.ctm_lm import ctm_lm_factory;return ctm_lm_factory()
    if model=='rdt':
        from ctm_transformer.sync_rdt import cell_factory;return cell_factory('rdt')
    return None


def curriculum_for(name,steps,batch):
    if name=='none':return None
    from ctm_transformer.curriculum import length_curriculum;return length_curriculum(name,steps,batch)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--model',choices=RECIPES,required=True);p.add_argument('--group',choices=DATA,default='S3')
    p.add_argument('--positions',choices=('learned','rope','nope'),required=True);p.add_argument('--depth',choices=('fixed','rand'),default='fixed')
    p.add_argument('--curriculum',default='none');p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--train-depth',type=int,help='fixed training depth (default: the recipe T=16)');p.add_argument('--width',type=int,help='d_model (FFN width 3x)');p.add_argument('--ticks',type=int,nargs='+',default=[4,8,16,32,48,64]);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.seed not in DEV_SEEDS:p.error(f'Development seeds only: {DEV_SEEDS}')
    data=DATA[a.group]
    if not (data/'manifest.json').exists():
        create_group_suite(data,a.group,DEV_SEEDS if a.group=='S3' else DEV_SEEDS[:3],320000,16,512,1024,32,data.name)
    directory=RUNS/a.name
    if directory.exists():p.error('Probe directory exists')
    config,identity=load_research_config(RECIPES[a.model])
    if a.model=='ctm_lm':
        from ctm_transformer.ctm_lm import lm_config;config=lm_config(config)
    config=replace(config,learning_rate=a.lr,max_steps=a.steps,warmup_steps=max(1,a.steps//100),eval_interval=max(1,a.steps//40),
                   log_interval=max(1,a.steps//20),device=a.device,checkpoint_dir=str(directory),
                   use_positional_encoding=a.positions=='learned')
    if a.train_depth:config=replace(config,max_thought_steps=a.train_depth,train_depth_min=a.train_depth,train_depth_max=a.train_depth)
    if a.width:config=replace(config,d_model=a.width,ffn_hidden_dim=3*a.width)
    train,valid,evaluation=(load_group_split(data,a.seed,s,config.seq_len) for s in ('train','validation','evaluation'))
    factory=factory_for(a.model,a.positions);sampler=lognormal_poisson(15,0.5,48) if a.depth=='rand' else None
    if a.depth=='rand' and a.model=='transformer':p.error('The Transformer has no recurrence depth')
    order=curriculum_for(a.curriculum,a.steps,config.batch_size)
    policy='final' if a.model=='transformer' else 'confidence'
    summary=train_dense_experiment(config,identity,a.seed,train,valid,evaluate_dense,[policy],save_validation_checkpoints=False,
        data_policy=f'{a.group} running products; development suite; lengths 1-16; curriculum {a.curriculum}',model_factory=factory,depth_sampler=sampler,order_sampler=order)
    model=(factory or build_model)(config).to(a.device).eval()
    model.load_state_dict(torch.load(directory/'final.pt',map_location=a.device,weights_only=False)['model_state_dict'])
    ticks=[1] if a.model=='transformer' else a.ticks;sweep={}
    for T in ticks:
        m=evaluate_dense(model,evaluation,replace(config,max_thought_steps=T),a.device,(policy,))[policy]
        by=m['answer_accuracy_by_position']
        sweep[str(T)]={'answer_accuracy':m['answer_accuracy'],'by_position':by,'positions_1_16':statistics.mean(by[:16]),
                       'positions_9_16':statistics.mean(by[8:16]),'positions_17_32':statistics.mean(by[16:32])}
    rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    record={'role':'exploratory development probe; not a paper endpoint','name':a.name,'group':a.group,'model':a.model,'positions':a.positions,'depth':a.depth,
        'curriculum':a.curriculum,'learning_rate':a.lr,'train_depth':config.max_thought_steps,'d_model':config.d_model,'steps':a.steps,'seed':a.seed,'parameters':summary['parameters'],
        'training_minutes':summary['training_seconds']/60,'final_train_loss_last100':sum(r['loss'] for r in rows[-100:])/100,
        'validation_curve':[{'step':r['step'],'answer_accuracy':r['validation']['answer_accuracy']} for r in rows if 'validation' in r],
        'eval_by_ticks':sweep,'data_manifest_sha256':file_hash(data/'manifest.json'),
        'source_sha256':{s:file_hash(s) for s in ('scripts/run_recipe_probe.py','ctm_transformer/positions.py','ctm_transformer/dense_experiment_v5.py','ctm_transformer/curriculum.py',
                                                  'ctm_transformer/ctm_lm.py','ctm_transformer/baselines.py','ctm_transformer/dense_pointer.py')}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'{a.name}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'name':a.name,**{T:{k:round(v[k],3) for k in ('positions_1_16','positions_9_16','positions_17_32')} for T,v in sweep.items()}}),flush=True)

if __name__=='__main__':main()
