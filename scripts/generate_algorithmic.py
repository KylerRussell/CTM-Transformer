"""Generate immutable, disjoint addition and pointer-chasing development splits."""
import argparse
import json
from pathlib import Path
from ctm_transformer.algorithmic import generate_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260921)
    args = parser.parse_args()
    manifest = generate_suite(args.output, args.seed)
    print(json.dumps({task: {split: info['examples'] for split, info in splits.items()}
                      for task, splits in manifest['tasks'].items()}, indent=2))


if __name__ == '__main__':
    main()
