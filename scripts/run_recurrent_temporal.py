"""Execute frozen recurrent temporal-supervision cells on one GPU."""
import argparse,gc,json
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.experiment import train_experiment,file_hash
from ctm_transformer.research import load_research_config

ROOT=Path('research/results/recurrent_temporal_v1')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',required=True);parser.add_argument('--cells',nargs='+',required=True)
    args=parser.parse_args();registry=json.loads((ROOT/'registry.json').read_text())
    pre=json.loads((ROOT/'pre_run_source.json').read_text())
    for path,digest in pre['files'].items():assert file_hash(path)==digest,path
    entries={e['cell']:e for e in registry['entries']}
    assert len(args.cells)==len(set(args.cells)) and set(args.cells)<=set(entries)
    for cell in args.cells:
        entry=entries[cell];assert not entry['reused']
        assert file_hash(entry['config'])==entry['config_sha256']
        c,identity=load_research_config(entry['config'])
        c=replace(c,device=args.device,data_path=str(Path(registry['train_path']).resolve()),
                  eval_data_path=str(Path(registry['validation_path']).resolve()),checkpoint_dir=str(Path(entry['run_directory']).resolve()))
        try:
            summary=train_experiment(c,identity,entry['seed'],'algorithmic',['confidence'],True,recurrent_objective=entry['objective'])
        except Exception as exc:
            import traceback
            failure={'cell':cell,'seed':entry['seed'],'objective':entry['objective'],'error':repr(exc),'traceback':traceback.format_exc()}
            (ROOT/(cell+'.failure.json')).write_text(json.dumps(failure,indent=2)+'\n')
            raise
        summary.update(cell=cell,seed=entry['seed'],objective=entry['objective'],worker_source_sha256=file_hash('scripts/run_recurrent_temporal.py'))
        with Path(entry['summary_path']).open('x') as output:json.dump(summary,output,indent=2);output.write('\n')
        print('Completed '+cell,flush=True);gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
