"""Run a frozen CTM or baseline configuration through the shared GPU trainer.

Data paths, device, run seed, step budget, and output directory are explicit run
settings. Architecture and optimizer settings come only from the config file.
Use independent processes on cuda:0 and cuda:1 for independent experiments.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from ctm_transformer.research import load_research_config
from ctm_transformer.experiment import train_experiment, validate_training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_format', choices=['text', 'algorithmic'], default='text')
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--eval_data_path', type=Path, required=True)
    parser.add_argument('--checkpoint_dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--dry_run', action='store_true', help='Validate and print config without starting training')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('This launcher runs independent single-GPU experiments; do not launch it with torchrun')
    config, identity = load_research_config(args.config)
    if not args.data_path.is_file() or not args.eval_data_path.is_file():
        parser.error('Provide existing, separate training and evaluation files')
    if args.data_path.resolve() == args.eval_data_path.resolve():
        parser.error('Training and evaluation files must differ')
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error('max_steps must be positive')
        config.max_steps = args.max_steps
    config.device = args.device
    config.data_path = str(args.data_path.resolve())
    config.eval_data_path = str(args.eval_data_path.resolve())
    config.dataset = ''
    config.checkpoint_dir = str(args.checkpoint_dir.resolve())
    if config.warmup_steps >= config.max_steps:
        parser.error('max_steps must exceed the frozen warmup_steps')
    validate_training_config(config)
    record = {'config_identity': identity, 'seed': args.seed, 'effective_config': asdict(config)}
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        parser.error('A working CUDA device is required')
    train_experiment(config, identity, args.seed, data_format=args.data_format)


if __name__ == '__main__':
    main()
