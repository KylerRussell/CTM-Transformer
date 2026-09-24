"""Evaluate frozen longer-budget checkpoints on the paired all-start query grid."""
import argparse
import gc
import json
from pathlib import Path
import torch
from ctm_transformer.algorithmic_eval import evaluate_checkpoint
from ctm_transformer.experiment import file_hash
from ctm_transformer.query_diagnostic import validate_query_suite


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',required=True)
    p.add_argument('--families',nargs='+',choices=['transformer','recurrent_depth','ctm'],required=True)
    p.add_argument('--dataset_root',type=Path,default=Path('research/data/query_diagnostic_v1'))
    p.add_argument('--source_root',type=Path,default=Path('research/data/ordered_pointer_v1'))
    p.add_argument('--previous_results',type=Path,default=Path('research/results/ordered_long_v1'))
    p.add_argument('--results_root',type=Path,default=Path('research/results/query_diagnostic_v1'))
    args=p.parse_args()
    if len(set(args.families))!=len(args.families):p.error('Families must be unique')
    data=validate_query_suite(args.dataset_root,args.source_root)
    args.results_root.mkdir(parents=True,exist_ok=True)
    for family in args.families:
        output=args.results_root/f'{family}_seed17.eval.json'
        audit=args.results_root/f'{family}_seed17.audit.json'
        if output.exists() or audit.exists():p.error('Use fresh family result paths')
        prior_path=args.previous_results/f'ordered_{family}_seed17.eval.json'
        prior=json.loads(prior_path.read_text());checkpoint=prior['checkpoint']
        if file_hash(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Selected checkpoint changed')
        result=evaluate_checkpoint(checkpoint['path'],args.dataset_root,'pointer',args.device,output,
            splits=list(data),selection='Frozen lowest-validation-CE checkpoint from ordered_long_v1 seed17; no reselection')
        if result['model_family']!=family:raise ValueError('Checkpoint family mismatch')
        audit.write_text(json.dumps({'complete':True,'previous_evaluation_sha256':file_hash(prior_path),
            'checkpoint_sha256':checkpoint['sha256'],'dataset_manifest_sha256':file_hash(args.dataset_root/'manifest.json'),
            'source_sha256':{path:file_hash(path) for path in ('ctm_transformer/query_diagnostic.py',
                'ctm_transformer/pointer_diagnostics.py','ctm_transformer/ordered_pointer.py','scripts/run_query_diagnostic.py')}},indent=2)+'\n')
        print(f'Completed {family}: {sum(len(d) for d in data.values())} queries',flush=True)
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
