"""Profile actual answer-masked CTM training batches at a frozen temporal preset."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from ctm_transformer.algorithmic import load_algorithmic_split
from ctm_transformer.research import load_research_config,build_model,parameter_counts
from ctm_transformer.experiment import file_hash


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
 p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('Fresh profile output required')
 torch.set_num_threads(4);torch.cuda.set_device(a.device);torch.manual_seed(17)
 c,identity=load_research_config(a.config)
 model=build_model(c).to(a.device).train()
 data=load_algorithmic_split('research/data/ordered_pointer_v1','pointer','train',c.seq_len)
 x,y,n=data.batch(slice(0,c.batch_size));x,y=x.to(a.device),y.to(a.device)
 opt=torch.optim.AdamW(model.parameters(),lr=c.learning_rate,betas=(c.adam_beta1,c.adam_beta2),weight_decay=c.weight_decay)
 def step():
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,targets=y)['loss']
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c.grad_clip,error_if_nonfinite=True);opt.step()
  return float(loss.detach())
 for _ in range(5):step()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 times=[];losses=[]
 for _ in range(20):
  start=time.perf_counter();loss=step();torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(loss)
 assert all(torch.isfinite(torch.tensor(losses)))
 report={'purpose':'training feasibility/timing on repeated first training batch; not a quality result',
         'config_identity':identity,'device':a.device,'gpu':torch.cuda.get_device_name(a.device),
         'warmup_steps':5,'timed_steps':20,'input_shape':list(x.shape),'real_input_tokens':n,
         'supervised_tokens':int((y!=-100).sum()),'parameters':parameter_counts(model),
         'median_step_seconds':statistics.median(times),'projected_3000_training_seconds':3000*statistics.median(times),
         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'step_seconds':times,'losses':losses,
         'source_sha256':{p:file_hash(p) for p in ('scripts/profile_temporal_depth.py','ctm_transformer/model.py','ctm_transformer/config.py','ctm_transformer/research.py')}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ('device','median_step_seconds','projected_3000_training_seconds','peak_allocated_bytes')},indent=2))

if __name__=='__main__':main()
