"""Measure full-size auxiliary training steps before freezing the GPU budget."""
import gc,json,statistics,time
from pathlib import Path
import torch
from ctm_transformer.algorithmic import AlgorithmicTokenizer,AnswerDataset
from ctm_transformer.experiment import file_hash
from ctm_transformer.recurrent_temporal import RecurrentTemporalTransformer
from ctm_transformer.research import load_research_config,parameter_counts


def main():
    torch.set_num_threads(4);torch.cuda.set_device(0)
    c,_=load_research_config('research/configs/fresh_confirmation_v1/recurrent_depth_seed23.json')
    data=AnswerDataset('research/data/ordered_pointer_v1/pointer/train.jsonl',AlgorithmicTokenizer(),c.seq_len)
    x,y,_=data.batch(slice(0,c.batch_size));x,y=x.cuda(),y.cuda();records=[]
    for objective in ('uniform','dynamic_aggregate'):
        torch.manual_seed(23);model=RecurrentTemporalTransformer(c,objective).cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=c.learning_rate,betas=(c.adam_beta1,c.adam_beta2),weight_decay=c.weight_decay)
        timings=[];torch.cuda.reset_peak_memory_stats()
        for i in range(10):
            torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(x,y)
            output['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c.grad_clip,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize()
            if i>=3:timings.append(time.perf_counter()-start)
        records.append({'objective':objective,'parameters':parameter_counts(model),'warmup':3,'measured_steps':7,
                        'median_update_seconds':statistics.median(timings),'times_seconds':timings,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'block_applications':output['block_applications'],'final_fixture_loss':float(output['loss'].detach())})
        del model,optimizer,output;gc.collect();torch.cuda.empty_cache()
    root=Path('research/results/recurrent_temporal_v1');root.mkdir(parents=True,exist_ok=True)
    report={'device':'cuda:0','gpu':torch.cuda.get_device_name(0),'records':records,
            'scope':'Repeated training-only batch for timing/memory feasibility; not a scientific training trial or validation selection.',
            'source_sha256':{p:file_hash(p) for p in ('scripts/profile_recurrent_temporal.py','ctm_transformer/recurrent_temporal.py')}}
    with (root/'profile.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
