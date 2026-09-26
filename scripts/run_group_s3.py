"""Train declared S3 group-word cells with runner v3 (slots ``cuda:N#k`` share a GPU)."""
import argparse,gc,json,traceback
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.dense_experiment import train_dense_experiment
from ctm_transformer.dense_pointer import evaluate_dense
from ctm_transformer.experiment import file_hash
from ctm_transformer.group_suite import load_group_split
from ctm_transformer.research import load_research_config

ROOT=Path('research/results/group_s3_v1')


def factory_for(entry):
    if entry['factory_kind'] is None:return None
    if entry['factory_kind']=='sync_rdt':
        from ctm_transformer.sync_rdt import cell_factory;return cell_factory(entry['factory_arg'])
    if entry['factory_kind']=='ctm_variant':
        from ctm_transformer.ctm_variants import variant_factory;return variant_factory(entry['factory_arg'])
    raise ValueError(entry['factory_kind'])


def run_cell(registry,entry,device):
    if file_hash(entry['config'])!=entry['config_sha256']:raise ValueError('Config hash mismatch')
    for p,h in registry['training_source_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Frozen training source changed: '+p)
    c,identity=load_research_config(entry['config'])
    train=load_group_split(registry['dataset'],entry['seed'],'train',c.seq_len)
    valid=load_group_split(registry['dataset'],entry['seed'],'validation',c.seq_len)
    c=replace(c,device=device,data_path=train.metadata['path'],eval_data_path=valid.metadata['path'],
              checkpoint_dir=str(Path(entry['run_directory']).resolve()))
    result=train_dense_experiment(c,identity,entry['seed'],train,valid,evaluate_dense,entry['selection_readouts'],
        save_validation_checkpoints=False,model_factory=factory_for(entry),
        data_policy='S3 running products; fresh random words of lengths 1-16; train order shuffled by data seed; every position labeled')
    result.update(cell=entry['cell'],seed=entry['seed'],worker_source_sha256=file_hash(__file__))
    Path(entry['summary_path']).write_text(json.dumps(result,indent=2)+'\n')
    print('Completed '+entry['cell'],flush=True)
    del train;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',required=True);p.add_argument('--cells',nargs='+',required=True)
    a=p.parse_args();device=a.device.split('#')[0]
    r=json.loads((ROOT/'registry.json').read_text());pre=json.loads((ROOT/'pre_run_source.json').read_text())
    for path,h in pre['files'].items():assert file_hash(path)==h,path
    entries={e['cell']:e for e in r['entries']}
    if len(set(a.cells))!=len(a.cells) or any(c not in entries for c in a.cells):p.error('Unknown or duplicate cells')
    for cell in a.cells:
        try:run_cell(r,entries[cell],device)
        except Exception as exc:
            (ROOT/f'{cell}.failure.json').write_text(json.dumps({'cell':cell,'device':a.device,'error':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
            raise

if __name__=='__main__':main()
