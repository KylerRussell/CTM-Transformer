"""Generate all source queries on ordered training-probe and validation maps."""
import argparse
from ctm_transformer.query_diagnostic import create_query_suite,validate_query_suite

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_root',default='research/data/ordered_pointer_v1')
    p.add_argument('--output_root',default='research/data/query_diagnostic_v1')
    args=p.parse_args()
    create_query_suite(args.source_root,args.output_root)
    data=validate_query_suite(args.output_root,args.source_root)
    print(f'{len(data)} paired splits; {sum(len(d) for d in data.values())} queries verified')

if __name__=='__main__':main()
