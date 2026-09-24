"""Derive reproducible tiny-set and one-hop controls from the original pilot."""
import argparse
import json
from ctm_transformer.pointer_diagnostics import create_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--seed', type=int, default=20260922)
    args = parser.parse_args()
    results = create_diagnostics(args.source_root, args.output_root, args.seed)
    print(json.dumps({mode:{s:e['examples'] for s,e in m['tasks']['pointer'].items()}
                      for mode,m in results.items()}, indent=2))


if __name__ == '__main__':
    main()
