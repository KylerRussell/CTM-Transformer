"""Evaluate frozen fresh-map pointer checkpoints on development splits.

The primary endpoint uses the declared family readout at the trained thought
depth. CTM and recurrent-depth checkpoints are also evaluated at other tick
budgets (a secondary, exploratory sweep); only the budget changes.
"""
import argparse,gc,json
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.algorithmic import load_algorithmic_split
from ctm_transformer.experiment import file_hash
from ctm_transformer.readout_selection import evaluate_readouts
from scripts.eval_harness import _load_checkpoint

ROOT=Path('research/results/online_pointer_v1')


def complete(path):
    try:return json.loads(path.read_text())['complete']
    except (FileNotFoundError,json.JSONDecodeError,KeyError):return False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',required=True);p.add_argument('--cells',nargs='+',required=True)
    a=p.parse_args()
    frozen=json.loads((ROOT/'checkpoints.json').read_text());assert frozen['complete'] and frozen['all_primary_checkpoints_frozen_before_evaluation']
    assert file_hash(ROOT/'registry.json')==frozen['registry_sha256']
    assert file_hash(Path(frozen['dataset'])/'manifest.json')==frozen['dataset_manifest_sha256']
    for source,h in frozen['source_sha256'].items():assert file_hash(source)==h,source
    models={t['cell']:t for t in frozen['models']};assert set(a.cells)<=set(models)
    registry=json.loads((ROOT/'registry.json').read_text())
    splits={s:load_algorithmic_split(frozen['dataset'],'pointer',s,128) for s in registry['evaluation_splits']}
    torch.set_num_threads(4);torch.cuda.set_device(a.device)
    for cell in a.cells:
        out=ROOT/f'{cell}.evaluation.json';t=models[cell]
        if complete(out):print(cell,'already evaluated',flush=True);continue
        assert file_hash(t['checkpoint'])==t['checkpoint_sha256']
        model,config=_load_checkpoint(t['checkpoint']);model.to(a.device).eval();results={}
        for split,data in splits.items():
            results[split]={}
            for depth in t['evaluation_thought_steps']:
                values=evaluate_readouts(model,data,replace(config,max_thought_steps=depth),a.device,(t['readout'],))
                metrics=values[t['readout']];assert len(metrics['predictions'])==len(data)
                if depth!=t['trained_thought_steps']:metrics.pop('predictions')  # Keep per-example rows for the primary budget only.
                results[split][str(depth)]={'metrics':metrics,'diagnostics':values['diagnostics']}
                print(cell,split,f'T={depth}',round(metrics['generation']['exact_match'],4),flush=True)
        report={'complete':True,'cell':cell,'selected':t,'dataset':{s:d.metadata for s,d in splits.items()},'results':results,
            'primary_thought_steps':t['trained_thought_steps'],'checkpoint':model._checkpoint_metadata,
            'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'device':a.device,
            'gpu':torch.cuda.get_device_name(a.device),'no_test_evaluation':True}
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
