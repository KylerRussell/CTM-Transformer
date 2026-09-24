"""Execute frozen development trials; never evaluate test data here."""
import argparse
from dataclasses import replace
import gc,json
from pathlib import Path
import torch
from ctm_transformer.experiment import train_experiment,file_hash
from ctm_transformer.research import load_research_config
from ctm_transformer.ordered_pointer import validate_ordered_suite


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cells',nargs='+',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();root=Path('research/results/readout_comparison_v1')
    registry=json.loads((root/'registry.json').read_text());entries={r['cell']:r for r in registry['entries']}
    if len(set(a.cells))!=len(a.cells) or any(c not in entries for c in a.cells):p.error('Unknown or repeated cell')
    validate_ordered_suite('research/data/ordered_pointer_v1')
    for cell in a.cells:
        e=entries[cell];assert file_hash(e['config'])==e['config_sha256']
        c,identity=load_research_config(e['config'])
        c=replace(c,device=a.device,data_path=str(Path('research/data/ordered_pointer_v1/pointer/train.jsonl').resolve()),
            eval_data_path=str(Path('research/data/ordered_pointer_v1/pointer/validation.jsonl').resolve()),checkpoint_dir=str(Path(c.checkpoint_dir).resolve()))
        result=train_experiment(c,identity,registry['seed'],data_format='algorithmic',selection_readouts=e['selection_readouts'],save_validation_checkpoints=True)
        result['cell']=cell;result['worker_source_sha256']=file_hash('scripts/run_readout_comparison.py')
        (root/f'{cell}.summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('Completed '+cell,flush=True);gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
