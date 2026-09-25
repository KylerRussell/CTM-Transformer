"""Evaluate frozen Sync-RDT retrieval checkpoints on each seed's held-out evaluation maps."""
import argparse,gc,json
from pathlib import Path
import torch
from ctm_transformer.dense_pointer import evaluate_dense
from ctm_transformer.experiment import file_hash
from ctm_transformer.mqar_suite import load_split
from ctm_transformer.research import config_from_dict
from ctm_transformer.sync_rdt import cell_factory

ROOT=Path('research/results/syncrdt_retrieval_v1')


def complete(path):
    try:return json.loads(path.read_text())['complete']
    except (FileNotFoundError,json.JSONDecodeError,KeyError):return False


def load_model(checkpoint,factory_cell,device):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=config_from_dict(payload['config'],'recurrent_depth',require_all=True)
    model=cell_factory(factory_cell)(config);model.load_state_dict(payload['model_state_dict'],strict=True)
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
        model,config=load_model(t['checkpoint'],t['factory_cell'],device)
        data=load_split(frozen['dataset'],t['seed'],'evaluation',config.seq_len)
        values=evaluate_dense(model,data,config,device,('confidence','final'))
        report={'complete':True,'cell':cell,'selected':t,'dataset':data.metadata,'primary_readout':'confidence',
            'results':{p:values[p] for p in ('confidence','final')},'diagnostics':values['diagnostics'],
            'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'device':device,
            'gpu':torch.cuda.get_device_name(device),'no_test_evaluation':True}
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
        print(cell,{p:round(values[p]['answer_accuracy_by_position'][0],4) for p in ('confidence','final')},flush=True)
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
