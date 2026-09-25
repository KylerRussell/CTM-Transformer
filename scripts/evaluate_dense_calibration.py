"""Evaluate frozen dense-calibration checkpoints on development splits (teacher-forced, per answer)."""
import argparse,gc,json
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.dense_pointer import evaluate_dense,load_dense_split
from ctm_transformer.experiment import file_hash
from scripts.eval_harness import _load_checkpoint

ROOT=Path('research/results/dense_calibration_v1')


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
    splits={s:load_dense_split(frozen['dataset'],s) for s in registry['evaluation_splits']}
    torch.set_num_threads(4);torch.cuda.set_device(a.device)
    for cell in a.cells:
        out=ROOT/f'{cell}.evaluation.json';t=models[cell]
        if complete(out):print(cell,'already evaluated',flush=True);continue
        assert file_hash(t['checkpoint'])==t['checkpoint_sha256']
        model,config=_load_checkpoint(t['checkpoint']);model.to(a.device).eval();results={}
        for split,data in splits.items():
            results[split]={}
            for depth in t['evaluation_thought_steps']:
                values=evaluate_dense(model,data,replace(config,max_thought_steps=depth),a.device,(t['readout'],))
                results[split][str(depth)]={'metrics':values[t['readout']],'diagnostics':values['diagnostics']}
                print(cell,split,f'T={depth}',round(values[t['readout']]['answer_accuracy'],4),flush=True)
        report={'complete':True,'cell':cell,'selected':t,'dataset':{s:d.metadata for s,d in splits.items()},'results':results,
            'primary_thought_steps':t['trained_thought_steps'],'checkpoint':model._checkpoint_metadata,
            'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'device':a.device,
            'gpu':torch.cuda.get_device_name(a.device),'no_test_evaluation':True}
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
