"""Run one exploratory group-word probe (development only; not a paper endpoint).

Trains on running products for lengths 1..L (balanced; fresh random sequences),
validates on lengths 1..L, and evaluates held-out sequences at longer lengths.
Recurrent families are also evaluated at several inference tick budgets.
"""
import argparse,hashlib,json
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.algorithmic import AlgorithmicTokenizer
from ctm_transformer.dense_experiment import train_dense_experiment
from ctm_transformer.dense_pointer import evaluate_dense
from ctm_transformer.experiment import file_hash
from ctm_transformer.group_word import Group,GroupWordDataset,write_group_split
from ctm_transformer.research import build_model,load_research_config

RECIPES={'transformer':'research/configs/presentation_control_v1/transformer_shuffled_seed23.json',
         'ctm':'research/configs/presentation_control_v1/ctm_shuffled_seed23.json',
         'recurrent_depth':'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'}
OUT=Path('research/results/group_probes');RUNS=Path('research/runs/group_probes')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--group',choices=('Z2','S3','A5'),required=True)
    p.add_argument('--family',choices=RECIPES,required=True);p.add_argument('--variant',default='reference');p.add_argument('--cell')
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--steps',type=int,default=10000);p.add_argument('--seed',type=int,default=41)
    p.add_argument('--train-max-len',type=int,default=16);p.add_argument('--eval-lengths',type=int,nargs='+',default=[4,8,12,16,24,32,48,64])
    p.add_argument('--ticks',type=int,nargs='+',default=[4,8,16,32]);p.add_argument('--data-name');p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    directory=RUNS/a.name;data=directory/'data';data.mkdir(parents=True,exist_ok=False)
    config,identity=load_research_config(RECIPES[a.family])
    config=replace(config,learning_rate=a.lr,max_steps=a.steps,warmup_steps=max(1,a.steps//100),eval_interval=max(1,a.steps//20),
                   log_interval=max(1,a.steps//20),device=a.device,checkpoint_dir=str(directory/'run'))
    from ctm_transformer.ctm_variants import variant_config,variant_factory
    from ctm_transformer.sync_rdt import cell_factory
    from ctm_transformer.ctm_lm import ctm_lm_factory,lm_config
    if a.family=='ctm' and a.variant=='ctm_lm':config=lm_config(config)  # CTM-LM (research/CTM_LM_DESIGN.md)
    elif a.family=='ctm':config=variant_config(config,a.variant)
    factory=ctm_lm_factory() if a.variant=='ctm_lm' else variant_factory(a.variant) if a.family=='ctm' else cell_factory(a.cell) if a.cell else None
    group=Group(a.group);base=int(hashlib.sha256((a.data_name or a.name).encode()).hexdigest()[:8],16)
    train_lengths=list(range(1,a.train_max_len+1))
    ev=write_group_split(data/'eval.jsonl',group,a.eval_lengths,128*len(a.eval_lengths),base+2)
    val=write_group_split(data/'validation.jsonl',group,train_lengths,32*len(train_lengths),base+1,exclude=ev)
    write_group_split(data/'train.jsonl',group,train_lengths,a.steps*config.batch_size,base+3,exclude=ev|val)
    tokenizer=AlgorithmicTokenizer()
    train,valid,evaluation=(GroupWordDataset(data/f'{s}.jsonl',tokenizer,config.seq_len) for s in ('train','validation','eval'))
    policy='final' if a.family=='transformer' else 'confidence'
    summary=train_dense_experiment(config,identity,a.seed,train,valid,evaluate_dense,[policy],save_validation_checkpoints=False,
        data_policy=f'group word {a.group}; fresh random sequences, lengths 1..{a.train_max_len}',model_factory=factory)
    model=(factory or build_model)(config).to(a.device)
    model.load_state_dict(torch.load(directory/'run/final.pt',map_location=a.device,weights_only=False)['model_state_dict'])
    ticks=[1] if a.family=='transformer' else a.ticks;sweep={}
    for T in ticks:
        m=evaluate_dense(model,evaluation,replace(config,max_thought_steps=T),a.device,(policy,))[policy]
        sweep[str(T)]={'answer_accuracy':m['answer_accuracy'],'sequence_exact':m['sequence_exact'],
            'by_length':{h:{'answer_accuracy':v['answer_accuracy'],'sequence_exact':v['sequence_exact']} for h,v in m['by_hop'].items()},
            'by_position':m['answer_accuracy_by_position']}
    rows=[json.loads(x) for x in (directory/'run/metrics.jsonl').read_text().splitlines()]
    curve=[{'step':r['step'],'answer_accuracy':r['validation']['answer_accuracy'],'sequence_exact':r['validation']['sequence_exact']} for r in rows if 'validation' in r]
    record={'role':'exploratory development probe; not a paper endpoint','name':a.name,'group':a.group,'group_size':len(group.elements),
        'family':a.family,'variant':a.variant,'cell':a.cell,'learning_rate':a.lr,'steps':a.steps,'seed':a.seed,'train_max_len':a.train_max_len,
        'eval_lengths':a.eval_lengths,'trained_ticks':config.max_thought_steps,'parameters':summary['parameters'],
        'training_minutes':summary['training_seconds']/60,'final_train_ce_last100':sum(r['loss'] for r in rows[-100:])/100,
        'validation_curve':curve,'eval_by_ticks':sweep,
        'source_sha256':{s:file_hash(s) for s in ('ctm_transformer/group_word.py','scripts/run_group_probe.py','ctm_transformer/dense_experiment.py',
                                                  'ctm_transformer/dense_pointer.py','ctm_transformer/sync_rdt.py','ctm_transformer/ctm_variants.py','ctm_transformer/ctm_lm.py')}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'{a.name}.json').write_text(json.dumps(record,indent=2)+'\n')
    trained=str(config.max_thought_steps if a.family!='transformer' else 1)
    primary=sweep[trained] if trained in sweep else sweep[str(ticks[-1])]
    print(json.dumps({'name':a.name,'by_length':{h:round(v['answer_accuracy'],3) for h,v in primary['by_length'].items()}}),flush=True)

if __name__=='__main__':main()
