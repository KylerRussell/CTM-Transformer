"""Run one exploratory pointer-format probe (development only; not a paper endpoint).

Generates fresh maps (training maps are disjoint from validation/evaluation
maps; every training map is presented once), trains the declared Transformer
recipe with runner v3, then scores held-out maps teacher-forced per answer.
"""
import argparse,hashlib,json
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.algorithmic import AlgorithmicTokenizer
from ctm_transformer.dense_experiment import train_dense_experiment
from ctm_transformer.dense_pointer import evaluate_dense
from ctm_transformer.experiment import file_hash
from ctm_transformer.pointer_probes import FORMATS,InOrderDataset,ProbeDataset,write_curriculum_split,write_probe_split
from ctm_transformer.research import load_research_config

RECIPES={'transformer':'research/configs/presentation_control_v1/transformer_shuffled_seed23.json',
         'ctm':'research/configs/presentation_control_v1/ctm_shuffled_seed23.json',
         'recurrent_depth':'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'}
OUT=Path('research/results/pointer_probes');RUNS=Path('research/runs/pointer_probes')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--format',choices=FORMATS,required=True)
    p.add_argument('--hops',type=int,nargs='+',default=[1]);p.add_argument('--eval-hops',type=int,nargs='+')
    p.add_argument('--family',choices=RECIPES,default='transformer');p.add_argument('--variant',default='reference',help='CTM architecture variant (ctm_transformer.ctm_variants)');p.add_argument('--lr',type=float,required=True)
    p.add_argument('--layers',type=int);p.add_argument('--d-model',type=int)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch-size',type=int);p.add_argument('--queries',type=int,default=6);p.add_argument('--curriculum',nargs='+',help='phases HOPS:UPDATES in order, e.g. 1:5000 1,2:5000; overrides --hops/--steps');p.add_argument('--train-nodes',type=int,nargs=2,default=[12,12],help='inclusive map-size range for training only');p.add_argument('--seed',type=int,default=41);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    phases=[([int(h) for h in x.split(':')[0].split(',')],int(x.split(':')[1])) for x in a.curriculum] if a.curriculum else None
    if phases:a.hops=sorted({h for hs,_ in phases for h in hs});a.steps=sum(n for _,n in phases)
    directory=RUNS/a.name;data=directory/'data';data.mkdir(parents=True,exist_ok=False)
    eval_hops=a.eval_hops or sorted(set(a.hops)|{max(a.hops)+1,max(a.hops)+2})
    config,identity=load_research_config(RECIPES[a.family])
    overrides={k:v for k,v in (('n_layers',a.layers),('d_model',a.d_model),('batch_size',a.batch_size)) if v is not None}
    if a.family!='ctm' and a.variant!='reference':p.error('Variants apply to CTM only')
    config=replace(config,learning_rate=a.lr,max_steps=a.steps,warmup_steps=max(1,a.steps//100),eval_interval=max(1,a.steps//20),
                   log_interval=max(1,a.steps//20),device=a.device,checkpoint_dir=str(directory/'run'),**overrides)
    # Seeds per split keep training, validation and evaluation maps independent and disjoint.
    base=int(hashlib.sha256(a.name.encode()).hexdigest()[:8],16)
    val=write_probe_split(data/'validation.jsonl',a.format,a.hops,96*len(a.hops),base+1,queries=a.queries)
    ev=write_probe_split(data/'eval.jsonl',a.format,eval_hops,256*len(eval_hops),base+2,exclude=val,queries=a.queries)
    if phases:write_curriculum_split(data/'train.jsonl',a.format,phases,config.batch_size,base+3,exclude=val|ev,queries=a.queries)
    else:write_probe_split(data/'train.jsonl',a.format,a.hops,a.steps*config.batch_size,base+3,exclude=val|ev,nodes=tuple(a.train_nodes),queries=a.queries)
    from ctm_transformer.ctm_variants import build_variant,variant_config,variant_factory
    if a.family=='ctm':config=variant_config(config,a.variant)
    factory=variant_factory(a.variant) if a.family=='ctm' else None
    tokenizer=AlgorithmicTokenizer()
    train=(InOrderDataset if phases else ProbeDataset)(data/'train.jsonl',tokenizer,config.seq_len)
    valid,evaluation=(ProbeDataset(data/f'{s}.jsonl',tokenizer,config.seq_len) for s in ('validation','eval'))
    policy='final' if a.family=='transformer' else 'confidence'
    summary=train_dense_experiment(config,identity,a.seed,train,valid,evaluate_dense,[policy],
        save_validation_checkpoints=False,data_policy=f'exploratory probe {a.format}; each training map presented once',model_factory=factory)
    from ctm_transformer.research import build_model
    model=(factory or build_model)(config).to(a.device)
    model.load_state_dict(torch.load(directory/'run/final.pt',map_location=a.device,weights_only=False)['model_state_dict'])
    final=evaluate_dense(model,evaluation,config,a.device,(policy,))[policy]
    rows=[json.loads(x) for x in (directory/'run/metrics.jsonl').read_text().splitlines()]
    curve=[{'step':r['step'],'answer_accuracy':r['validation']['answer_accuracy'],
            'by_hop':{h:v['answer_accuracy'] for h,v in r['validation']['by_hop'].items()}} for r in rows if 'validation' in r]
    record={'role':'exploratory development probe; not a paper endpoint','name':a.name,'format':a.format,'family':a.family,'variant':a.variant,
        'train_hops':a.hops,'batch_size':config.batch_size,'train_nodes':a.train_nodes,'queries':a.queries,'curriculum':phases,'eval_hops':eval_hops,'learning_rate':a.lr,'steps':a.steps,'seed':a.seed,'model_overrides':overrides,
        'parameters':summary['parameters'],'training_minutes':summary['training_seconds']/60,
        'final_train_ce_last100':sum(r['loss'] for r in rows[-100:])/min(100,len(rows)),'validation_curve':curve,
        'eval_answer_accuracy':final['answer_accuracy'],'eval_sequence_exact':final['sequence_exact'],
        'eval_by_hop':{h:v['answer_accuracy'] for h,v in final['by_hop'].items()},'eval_by_position':final['answer_accuracy_by_position'],
        'source_sha256':{s:file_hash(s) for s in ('ctm_transformer/pointer_probes.py','scripts/run_pointer_probe.py','ctm_transformer/dense_experiment.py','ctm_transformer/dense_pointer.py','ctm_transformer/ctm_variants.py')}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'{a.name}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('name','eval_answer_accuracy','eval_sequence_exact','eval_by_hop','final_train_ce_last100')}),flush=True)

if __name__=='__main__':main()
