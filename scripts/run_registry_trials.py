"""Run declared trials from an immutable registry, using validation only."""
import argparse,gc,json
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.experiment import train_experiment,file_hash
from ctm_transformer.research import load_research_config


def run_trial(registry,entry,device):
    if entry.get('reused'):raise ValueError('Reused trials must never be retrained')
    if file_hash(entry['config'])!=entry['config_sha256']:raise ValueError('Config hash mismatch')
    for p,h in registry['training_source_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Frozen training source changed: '+p)
    c,identity=load_research_config(entry['config'])
    c=replace(c,device=device,data_path=str(Path(registry['train_path']).resolve()),
        eval_data_path=str(Path(registry['validation_path']).resolve()),checkpoint_dir=str(Path(entry['run_directory']).resolve()))
    result=train_experiment(c,identity,entry['seed'],data_format='algorithmic',
        selection_readouts=entry['selection_readouts'],save_validation_checkpoints=True)
    result.update(cell=entry['cell'],seed=entry['seed'],worker_source_sha256=file_hash('scripts/run_registry_trials.py'))
    Path(entry['summary_path']).write_text(json.dumps(result,indent=2)+'\n')
    print('Completed '+entry['cell'],flush=True);gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--cells',nargs='+',required=True);p.add_argument('--device',required=True);a=p.parse_args()
    r=json.loads(a.registry.read_text());entries={e['cell']:e for e in r['entries']}
    if len(set(a.cells))!=len(a.cells) or any(c not in entries for c in a.cells):p.error('Unknown or duplicate cells')
    for cell in a.cells:run_trial(r,entries[cell],a.device)

if __name__=='__main__':main()
