"""Evaluate immutable winners and measure generation time on one GPU."""
import argparse,gc,json,statistics,time
from pathlib import Path
import torch
from ctm_transformer.experiment import file_hash
from ctm_transformer.algorithmic import load_algorithmic_split
from ctm_transformer.algorithmic_eval import greedy_answers
from ctm_transformer.confidence_readout import ConfidenceReadout
from ctm_transformer.readout_selection import evaluate_readouts
from scripts.eval_harness import _load_checkpoint

ROOT=Path('research/results/readout_comparison_v1')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cuda:0');a=parser.parse_args()
    selection=json.loads((ROOT/'selection.json').read_text());assert selection['complete']
    assert file_hash('scripts/select_readout_comparison.py')==selection['selection_source_sha256']
    torch.set_num_threads(4);torch.cuda.set_device(a.device)
    for family,winner in selection['winners'].items():
        out=ROOT/f'{family}.evaluation.json'
        if out.exists():raise ValueError('Fresh evaluation outputs required')
        assert file_hash(winner['checkpoint'])==winner['checkpoint_sha256']
        model,config=_load_checkpoint(winner['checkpoint']);model.to(a.device).eval()
        results={}
        for split in ('validation','test_id','test_shuffled'):
            dataset=load_algorithmic_split('research/data/ordered_pointer_v1','pointer',split,config.seq_len)
            values=evaluate_readouts(model,dataset,config,a.device,(winner['readout'],))
            results[split]={'dataset':dataset.metadata,'metrics':values[winner['readout']],'diagnostics':values['diagnostics']}
        assert abs(results['validation']['metrics']['loss']-winner['validation_ce'])<1e-6
        assert results['validation']['metrics']['generation']==winner['validation_generation']
        # Timing on validation prompts, common batch size32, no targets enter generation.
        data=load_algorithmic_split('research/data/ordered_pointer_v1','pointer','validation',config.seq_len)
        reader=model if winner['readout']=='final' else ConfidenceReadout(model)
        for _ in range(3):greedy_answers(reader,data.records[:32],config,a.device,config.max_thought_steps)
        times=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for _ in range(20):
            started=time.perf_counter();greedy_answers(reader,data.records[:32],config,a.device,config.max_thought_steps);torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
        report={'complete':True,'selection_sha256':file_hash(ROOT/'selection.json'),'winner':winner,
            'checkpoint':model._checkpoint_metadata,'gpu':torch.cuda.get_device_name(a.device),'device':a.device,
            'results':results,'generation_timing':{'batch_size':32,'warmup':3,'repeats':20,
                'seconds':times,'median_batch_seconds':statistics.median(times),'p95_batch_seconds':sorted(times)[18],
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'scope':'End-to-end unrestricted answer/EOS generation on first32 validation prompts; full prefixes recomputed; includes Python and device transfer; same GPU with other jobs finished.'},
            'source_sha256':{p:file_hash(p) for p in ('scripts/evaluate_readout_comparison.py','ctm_transformer/readout_selection.py','ctm_transformer/confidence_readout.py','ctm_transformer/algorithmic_eval.py','ctm_transformer/model.py','ctm_transformer/baselines.py')}}
        out.write_text(json.dumps(report,indent=2)+'\n');print(family,results['test_id']['metrics']['generation'],flush=True)
        del model,reader;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
