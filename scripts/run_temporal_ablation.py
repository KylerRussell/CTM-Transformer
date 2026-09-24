"""Train fixed-budget CTM temporal CE recipes; evaluate training/validation only."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import torch
from ctm_transformer.algorithmic_eval import evaluate_checkpoint
from ctm_transformer.experiment import train_experiment,file_hash,objective_metadata
from ctm_transformer.ordered_pointer import validate_ordered_suite
from ctm_transformer.research import load_research_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipes',nargs='+',choices=['final','uniform','late'],required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--run_root',type=Path,default=Path('research/runs/temporal_ablation_v1/seed17'))
    p.add_argument('--results_root',type=Path,default=Path('research/results/temporal_ablation_v1'))
    args=p.parse_args()
    if len(set(args.recipes))!=len(args.recipes):p.error('Recipes must be unique')
    root=Path('research/data/ordered_pointer_v1');validate_ordered_suite(root)
    args.results_root.mkdir(parents=True,exist_ok=True)
    for recipe in args.recipes:
        run=args.run_root/recipe;output=args.results_root/f'{recipe}.eval.json'
        if run.exists() or output.exists():p.error('Fresh recipe paths required')
        config,identity=load_research_config(f'research/configs/ctm_temporal_{recipe}_v1.json')
        config=replace(config,data_path=str((root/'pointer/train.jsonl').resolve()),
            eval_data_path=str((root/'pointer/validation.jsonl').resolve()),checkpoint_dir=str(run.resolve()),device=args.device)
        training=train_experiment(config,identity,args.seed,data_format='algorithmic')
        gc.collect();torch.cuda.empty_cache()
        evaluation=evaluate_checkpoint(run/'best.pt',root,'pointer',args.device,output,
            splits=['train','validation','train_probe','train_new_query'],
            selection='Minimum original-validation final-tick answer/EOS CE at 100-update checkpoints; independent of training objective')
        # Descriptive depth readouts on original validation only; never used for selection.
        ticks=evaluate_checkpoint(run/'best.pt',root,'pointer',args.device,
            args.results_root/f'{recipe}.ticks.eval.json',depths=[1,2,3,4],splits=['validation'],
            selection='Same frozen best final-tick validation-CE checkpoint; T1-3 are diagnostic readouts, not selected depths')
        grid=evaluate_checkpoint(run/'best.pt','research/data/query_diagnostic_v1','pointer',args.device,
            args.results_root/f'{recipe}.all_queries.eval.json',splits=[f'validation_start_{n}' for n in 'ABCDEFGH'],
            selection='Same frozen best original-validation final-tick CE checkpoint; all-start scores are secondary')
        summary={'complete':True,'recipe':recipe,'seed':args.seed,'objective':objective_metadata(config),
            'training':training,'checkpoint':evaluation['checkpoint'],
            'validation_ticks':{d:{k:v for k,v in row.items() if k!='predictions'} for d,row in ticks['results']['validation']['depths'].items()},
            'all_query_dataset_manifest_sha256':grid['dataset_manifest_sha256'],
            'dataset_manifest_sha256':evaluation['dataset_manifest_sha256'],
            'source_sha256':{p:file_hash(p) for p in ('scripts/run_temporal_ablation.py','ctm_transformer/experiment.py','ctm_transformer/model.py')},
            'metrics':{s:next(iter(v['depths'].values())) for s,v in evaluation['results'].items()}}
        for v in summary['metrics'].values():v.pop('predictions')
        (args.results_root/f'{recipe}.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f'Completed {recipe}',flush=True)
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
