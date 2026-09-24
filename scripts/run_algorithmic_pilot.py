"""Train the three development models on one task, on one GPU, at one seed."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch

from ctm_transformer.algorithmic import load_algorithmic_split
from ctm_transformer.algorithmic_eval import evaluate_checkpoint
from ctm_transformer.experiment import train_experiment, file_hash
from ctm_transformer.research import load_research_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', type=Path, required=True)
    parser.add_argument('--task', choices=['addition', 'pointer'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--run_root', type=Path, required=True)
    parser.add_argument('--results_root', type=Path, required=True)
    args = parser.parse_args()
    if args.run_root.exists() and any(args.run_root.iterdir()):
        parser.error('Use an empty run_root')
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)
    for family in ['transformer', 'recurrent_depth', 'ctm']:
        preset = Path('research/configs') / f'{family}_algorithmic_v1.json'
        config, identity = load_research_config(preset)
        # Validate all declared data hashes and cross-split semantic disjointness before training.
        manifest = json.loads((args.dataset_root / 'manifest.json').read_text())
        seen = set()
        for split in manifest['tasks'][args.task]:
            data = load_algorithmic_split(args.dataset_root, args.task, split, config.seq_len)
            if seen & data.ids:
                raise ValueError('Semantic overlap across dataset splits')
            seen.update(data.ids)
        run = args.run_root / family
        config = replace(config, data_path=str((args.dataset_root / args.task / 'train.jsonl').resolve()),
            eval_data_path=str((args.dataset_root / args.task / 'validation.jsonl').resolve()),
            checkpoint_dir=str(run.resolve()), device=args.device)
        summary = train_experiment(config, identity, args.seed, data_format='algorithmic')
        gc.collect(); torch.cuda.empty_cache()
        evaluation = evaluate_checkpoint(run / 'best.pt', args.dataset_root, args.task, args.device,
                        args.results_root / f'{args.task}_{family}_seed{args.seed}.eval.json')
        record = {'purpose': 'single-seed exploratory pilot; unmatched compute and parameter budgets',
                  'task': args.task, 'seed': args.seed, 'training': summary,
                  'dataset_manifest_sha256': file_hash(args.dataset_root / 'manifest.json'),
                  'checkpoint': evaluation['checkpoint'],
                  'metrics': {split: {t: {k:v for k,v in values.items() if k != 'predictions'}
                      for t, values in result['depths'].items()} for split, result in evaluation['results'].items()}}
        (args.results_root / f'{args.task}_{family}_seed{args.seed}.summary.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps({'completed': family, 'task': args.task, 'summary': summary}), flush=True)
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
