"""Evaluate frozen final checkpoints on paired development presentations only."""
import argparse,gc,json
from pathlib import Path
import torch
from ctm_transformer.algorithmic import load_algorithmic_split
from ctm_transformer.experiment import file_hash
from ctm_transformer.presentation_control import validate_presentation_control
from ctm_transformer.readout_selection import evaluate_readouts
from scripts.eval_harness import _load_checkpoint

ROOT=Path('research/results/presentation_control_v1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',required=True);p.add_argument('--families',nargs='+',required=True);args=p.parse_args()
    frozen=json.loads((ROOT/'checkpoints.json').read_text());assert frozen['complete'] and frozen['no_test_evaluation']
    assert frozen['all_primary_checkpoints_frozen_before_paired_evaluation']
    assert file_hash(ROOT/'registry.json')==frozen['registry_sha256']
    assert file_hash(Path(frozen['dataset'])/'manifest.json')==frozen['dataset_manifest_sha256']
    for source,h in frozen['source_sha256'].items():assert file_hash(source)==h,source
    validate_presentation_control(frozen['dataset'])
    assert len(args.families)==len(set(args.families)) and set(args.families)<={'ctm','transformer','recurrent_depth'}
    selected=[t for t in frozen['models'] if t['model_family'] in args.families]
    assert all(not (ROOT/f"{t['cell']}.evaluation.json").exists() for t in selected)
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    for t in selected:
        assert file_hash(t['checkpoint'])==t['checkpoint_sha256']
        model,config=_load_checkpoint(t['checkpoint']);model.to(args.device).eval();results={}
        for split in ('validation','validation_shuffled'):
            data=load_algorithmic_split(frozen['dataset'],'pointer',split,config.seq_len)
            values=evaluate_readouts(model,data,config,args.device,(t['readout'],));metrics=values[t['readout']]
            assert len(metrics['predictions'])==128 and metrics['target_tokens']==256
            if split=='validation':
                assert abs(metrics['loss']-t['ordered_validation_ce'])<1e-6
                assert metrics['generation']==t['ordered_validation_generation']
            results[split]={'dataset':data.metadata,'metrics':metrics,'diagnostics':values['diagnostics']}
        a,b=[results[s]['metrics']['predictions'] for s in ('validation','validation_shuffled')]
        assert [(r['id'],r['answer']) for r in a]==[(r['id'],r['answer']) for r in b]
        report={'complete':True,'cell':t['cell'],'selected':t,'results':results,'checkpoint':model._checkpoint_metadata,
            'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'source_sha256':frozen['source_sha256'],
            'device':args.device,'gpu':torch.cuda.get_device_name(args.device),'no_test_evaluation':True}
        with (ROOT/f"{t['cell']}.evaluation.json").open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        print(t['cell'],{s:results[s]['metrics']['generation']['exact_match'] for s in results},flush=True)
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
