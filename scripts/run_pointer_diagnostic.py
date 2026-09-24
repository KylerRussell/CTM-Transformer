"""Run one pointer diagnostic condition for all three model families on a GPU."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch

from ctm_transformer.algorithmic_eval import evaluate_checkpoint
from ctm_transformer.experiment import train_experiment, file_hash
from ctm_transformer.pointer_diagnostics import validate_diagnostic_suite
from ctm_transformer.ordered_pointer import validate_ordered_suite
from ctm_transformer.fixed_pointer import validate_fixed_suite
from ctm_transformer.research import load_research_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', type=Path, required=True)
    parser.add_argument('--mode', choices=['tiny', 'onehop', 'ordered', 'fixedcopy'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--families', nargs='+', choices=['transformer','recurrent_depth','ctm'],
                        default=['transformer','recurrent_depth','ctm'])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--config_tag', default='algorithmic_v1',
                        help='Load research/configs/{family}_{config_tag}.json; effective config and hash are recorded')
    parser.add_argument('--run_root', type=Path, required=True)
    parser.add_argument('--results_root', type=Path, required=True)
    args = parser.parse_args()
    if len(args.families) != len(set(args.families)):
        parser.error('Families must be unique')
    if any((args.run_root/family).exists() for family in args.families):
        parser.error('Requested family output directory already exists')
    manifest = json.loads((args.dataset_root/'manifest.json').read_text())
    if manifest['mode'] != args.mode:
        parser.error('Dataset mode does not match the requested diagnostic')
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)
    for family in args.families:
        result_path = args.results_root/f'{args.mode}_{family}_seed{args.seed}.eval.json'
        if result_path.exists():
            parser.error(f'Existing evaluation: {result_path}')
        config, identity = load_research_config(f'research/configs/{family}_{args.config_tag}.json')
        validator = {'ordered':validate_ordered_suite, 'fixedcopy':validate_fixed_suite}.get(args.mode, validate_diagnostic_suite)
        validator(args.dataset_root, config.seq_len)
        run = args.run_root/family
        config = replace(config, data_path=str((args.dataset_root/'pointer/train.jsonl').resolve()),
            eval_data_path=str((args.dataset_root/'pointer/validation.jsonl').resolve()),
            checkpoint_dir=str(run.resolve()), device=args.device)
        summary = train_experiment(config, identity, args.seed, data_format='algorithmic')
        gc.collect(); torch.cuda.empty_cache()
        checkpoint_name = 'final.pt' if args.mode == 'tiny' else 'best.pt'
        selection = (f'Final checkpoint after the fixed {config.max_steps}-update budget; primary endpoint is training-set memorization'
                     if args.mode == 'tiny' else f'Lowest validation answer/EOS CE among {config.eval_interval}-update checkpoints, including the final update')
        evaluation = evaluate_checkpoint(run/checkpoint_name, args.dataset_root, 'pointer', args.device,
            result_path, splits=list(manifest['tasks']['pointer']), selection=selection)
        record = {'purpose':'pointer learning diagnostic; one seed; unmatched model sizes and compute',
            'mode':args.mode, 'seed':args.seed, 'config_identity':identity, 'training':summary,
            'selection':selection, 'checkpoint':evaluation['checkpoint'],
            'dataset_manifest_sha256':file_hash(args.dataset_root/'manifest.json'),
            'diagnostic_source_sha256':{p:file_hash(p) for p in (
                'ctm_transformer/pointer_diagnostics.py', 'ctm_transformer/ordered_pointer.py',
                'ctm_transformer/fixed_pointer.py',
                'scripts/run_pointer_diagnostic.py')},
            'metrics':{split:{depth:{k:v for k,v in values.items() if k != 'predictions'}
                for depth,values in result['depths'].items()} for split,result in evaluation['results'].items()}}
        (args.results_root/f'{args.mode}_{family}_seed{args.seed}.summary.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'completed':family,'mode':args.mode,'summary':summary}),flush=True)
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
