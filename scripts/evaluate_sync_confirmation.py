"""Evaluate frozen confirmation checkpoints once on each seed's locked length-32 test words at every declared tick budget."""
import argparse,gc,json
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.dense_pointer import evaluate_dense
from ctm_transformer.experiment import file_hash
from ctm_transformer.group_suite import load_group_split
from ctm_transformer.research import build_model,config_from_dict
from scripts.run_sync_confirmation import factory_for

ROOT=Path('research/results/sync_confirmation_v1')


def complete(path):
    try:return json.loads(path.read_text())['complete']
    except (FileNotFoundError,json.JSONDecodeError,KeyError):return False


def load_model(t,device):
    payload=torch.load(t['checkpoint'],map_location='cpu',weights_only=False)
    config=config_from_dict(payload['config'],t['model_family'],require_all=True)
    model=(factory_for(t) or build_model)(config);model.load_state_dict(payload['model_state_dict'],strict=True)
    return model.to(device).eval(),config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',required=True);p.add_argument('--cells',nargs='+',required=True)
    a=p.parse_args();device=a.device.split('#')[0]
    frozen=json.loads((ROOT/'checkpoints.json').read_text());assert frozen['complete'] and frozen['all_primary_checkpoints_frozen_before_evaluation']
    assert file_hash(ROOT/'registry.json')==frozen['registry_sha256']
    assert file_hash(Path(frozen['dataset'])/'manifest.json')==frozen['dataset_manifest_sha256']
    for source,h in frozen['source_sha256'].items():assert file_hash(source)==h,source
    models={t['cell']:t for t in frozen['models']};assert set(a.cells)<=set(models)
    torch.set_num_threads(4);torch.cuda.set_device(device)
    for cell in a.cells:
        out=ROOT/f'{cell}.evaluation.json';t=models[cell]
        if complete(out):print(cell,'already evaluated',flush=True);continue
        assert file_hash(t['checkpoint'])==t['checkpoint_sha256']
        model,config=load_model(t,device)
        data=load_group_split(frozen['dataset'],t['seed'],'evaluation',config.seq_len);results={}
        for ticks in t['evaluation_thought_steps']:
            m=evaluate_dense(model,data,replace(config,max_thought_steps=ticks),device,(t['readout'],))[t['readout']]
            results[str(ticks)]={'answer_accuracy':m['answer_accuracy'],'sequence_exact':m['sequence_exact'],'loss':m['loss'],
                                 'by_position':m['answer_accuracy_by_position']}
        report={'complete':True,'cell':cell,'selected':t,'dataset':data.metadata,'readout':t['readout'],'results':results,
            'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'device':device,'gpu':torch.cuda.get_device_name(device),'no_test_evaluation':True}
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
        print(cell,{k:round(v['answer_accuracy'],4) for k,v in results.items()},flush=True)
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
