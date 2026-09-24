"""Generate the ordered-edge version of the paired one-hop diagnostic."""
import argparse
import json
from ctm_transformer.ordered_pointer import create_ordered_suite


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_root',required=True)
    parser.add_argument('--output_root',required=True)
    args=parser.parse_args()
    manifest=create_ordered_suite(args.source_root,args.output_root)
    print(json.dumps({k:v['examples'] for k,v in manifest['tasks']['pointer'].items()},indent=2))


if __name__=='__main__':
    main()
