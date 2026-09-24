"""Run new cells of the uniform/dynamic by T4/T16 CTM comparison."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import torch
from ctm_transformer.algorithmic import load_algorithmic_split
from ctm_transformer.algorithmic_eval import evaluate_checkpoint,greedy_answers,summarize_predictions
from ctm_transformer.confidence_readout import ConfidenceReadout
from ctm_transformer.experiment import train_experiment,file_hash,objective_metadata
from ctm_transformer.ordered_pointer import validate_ordered_suite
from ctm_transformer.research import load_research_config
from scripts.eval_harness import _load_checkpoint


def confidence_evaluation(checkpoint,device,output):
    if Path(output).exists():raise ValueError('Fresh confidence output required')
    model,c=_load_checkpoint(checkpoint);model.to(device).eval()
    data=load_algorithmic_split('research/data/ordered_pointer_v1','pointer','validation',c.seq_len)
    wrapped=ConfidenceReadout(model).eval();preds=[]
    for i in range(0,len(data),c.batch_size):
        preds.extend(greedy_answers(wrapped,data.records[i:i+c.batch_size],c,device,c.max_thought_steps))
    result={'complete':True,'checkpoint':model._checkpoint_metadata,'policy':'per-token minimum FP32 entropy, no labels; full unroll; no early exit',
        'metrics':summarize_predictions(preds),'predictions':preds,
        'source_sha256':{p:file_hash(p) for p in ('ctm_transformer/confidence_readout.py','scripts/run_objective_depth.py','ctm_transformer/algorithmic_eval.py','ctm_transformer/model.py')}}
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    del wrapped,model;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells',nargs='+',required=True,choices=['dynamic_t4','uniform_t16','dynamic_t16'])
    p.add_argument('--device',required=True)
    a=p.parse_args();root=Path('research/results/objective_depth_v1');root.mkdir(parents=True,exist_ok=True)
    if len(set(a.cells))!=len(a.cells):p.error('Cells must be unique')
    dataset=Path('research/data/ordered_pointer_v1');validate_ordered_suite(dataset)
    for cell in a.cells:
        run=Path('research/runs/objective_depth_v1/seed17')/cell
        if run.exists() or (root/f'{cell}.summary.json').exists():p.error('Fresh cell required')
        c,identity=load_research_config(f'research/configs/ctm_{cell}_v1.json')
        c=replace(c,data_path=str((dataset/'pointer/train.jsonl').resolve()),eval_data_path=str((dataset/'pointer/validation.jsonl').resolve()),
            checkpoint_dir=str(run.resolve()),device=a.device)
        training=train_experiment(c,identity,17,data_format='algorithmic');gc.collect();torch.cuda.empty_cache()
        ev=evaluate_checkpoint(run/'best.pt',dataset,'pointer',a.device,root/f'{cell}.eval.json',
            splits=['train','validation','train_probe','train_new_query'],selection='Minimum original-validation final-trained-tick CE; no confidence/readout reselection')
        ticks=evaluate_checkpoint(run/'best.pt',dataset,'pointer',a.device,root/f'{cell}.ticks.eval.json',
            depths=[t for t in (1,2,4,8,16) if t<=c.max_thought_steps],splits=['validation'],selection='Frozen checkpoint selected at trained T; other T values are diagnostic')
        grid=evaluate_checkpoint(run/'best.pt','research/data/query_diagnostic_v1','pointer',a.device,root/f'{cell}.all_queries.eval.json',
            splits=[f'validation_start_{n}' for n in 'ABCDEFGH'],selection='Same original-validation final-trained-tick checkpoint; secondary query coverage')
        confidence=confidence_evaluation(run/'best.pt',a.device,root/f'{cell}.confidence.json')
        record={'complete':True,'cell':cell,'training':training,'checkpoint':ev['checkpoint'],'objective':objective_metadata(c),
            'metrics':{s:{k:v for k,v in next(iter(x['depths'].values())).items() if k!='predictions'} for s,x in ev['results'].items()},
            'confidence_metrics':confidence['metrics'],'source_sha256':{p:file_hash(p) for p in ('scripts/run_objective_depth.py','ctm_transformer/experiment.py','ctm_transformer/model.py')}}
        (root/f'{cell}.summary.json').write_text(json.dumps(record,indent=2)+'\n');print('Completed '+cell,flush=True)
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
