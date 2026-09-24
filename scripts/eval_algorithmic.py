"""Score an algorithmic-task checkpoint with exact greedy answers and CE."""
import argparse
from ctm_transformer.algorithmic_eval import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--task', choices=['addition', 'pointer'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    parser.add_argument('--depths', help='Comma-separated thought budgets; default is training budget')
    parser.add_argument('--splits', help='Comma-separated split names; default is every non-training split')
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.dataset_root, args.task, args.device, args.output,
                        [int(t) for t in args.depths.split(',')] if args.depths else None,
                        args.splits.split(',') if args.splits else None)


if __name__ == '__main__':
    main()
