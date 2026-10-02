"""Pretrain one language-model arm on two GPUs (launch with torchrun --nproc_per_node 2).

    torchrun --nproc_per_node 2 -m scripts.pretrain --config research/configs/pretrain/<arm>.json

The run config names the family (transformer | rdt | ctm_lm), the model overrides
applied to the family's recipe, the depth sampler, and the training settings. An
RDT run may list CTM `mechanisms` (ctm_transformer/ctm_rdt.py); a CTM-LM run may
list `ctm_adaptations` (ctm_transformer/ctm_lm_adapt.py).
The run directory holds `run.json` (config, hashes, environment), `metrics.jsonl`
(rank 0), `latest.pt` (model, optimizer, step; written atomically every
`checkpoint_interval` steps) and periodic `step_XXXXXXX.pt` snapshots. A relaunch
resumes from `latest.pt`: data order and depths are functions of (seed, step),
so a resumed run sees exactly the batches and depths it would have seen.

Optional training settings: `eval_offset` (first validation window evaluated, so
that hyperparameter selection and the reported evaluation use disjoint windows),
`final_eval_windows` (a larger evaluation at the last step) and
`delete_checkpoint_on_complete`. A non-finite gradient norm ends the run with
`diverged.json` (a result, not a crash).
"""
import argparse,contextlib,datetime,hashlib,json,os,time
from dataclasses import asdict,replace
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ctm_transformer.ctm_lm import lm_config
from ctm_transformer.ctm_rdt import ctm_rdt_factory
from ctm_transformer.ctm_lm_adapt import adapted_factory
from ctm_transformer.depth_sampling import lognormal_poisson
from ctm_transformer.experiment import file_hash
from ctm_transformer.lm_scale import scaled_factory
from ctm_transformer.pretrain import OffloadedAdamW,TokenWindows,evaluate_lm,lr_at,micro_indices,set_lr,step_depth
from ctm_transformer.research import load_research_config

RECIPES={'transformer':'research/configs/presentation_control_v1/transformer_shuffled_seed23.json',
         'rdt':'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json',
         'ctm_lm':'research/configs/presentation_control_v1/ctm_shuffled_seed23.json'}
SOURCES=['scripts/pretrain.py','ctm_transformer/pretrain.py','ctm_transformer/lm_scale.py','ctm_transformer/positions.py','ctm_transformer/ctm_lm.py',
         'ctm_transformer/baselines.py','ctm_transformer/ctm_variants.py','ctm_transformer/depth_sampling.py','ctm_transformer/ctm_rdt.py','ctm_transformer/ctm_lm_adapt.py']


def build_config(run,vocab):
    c,_=load_research_config(RECIPES[run['family']])
    if run['family']=='ctm_lm':c=lm_config(c)
    t=run['train'];over=dict(run['model']);over.setdefault('n_heads',over['d_model']//64)
    c=replace(c,vocab_size=vocab,seq_len=t['seq_len'],max_seq_len=t['seq_len'],use_positional_encoding=False,dtype='bfloat16',
              learning_rate=t['lr'],batch_size=t['micro_batch'],**over)
    if run['family']=='rdt':c=replace(c,max_thought_steps=run['eval_depth'],train_depth_min=run['eval_depth'],train_depth_max=run['eval_depth'])
    return c


def sampler_for(run):
    s=run.get('depth_sampler')
    if not s:return None
    assert s['kind']=='lognormal_poisson';return lognormal_poisson(s['mean'],s['sigma'],s['maximum'])


def atomic_save(payload,path):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--max-steps',type=int,help='stop early (smoke tests)')
    a=p.parse_args()
    distributed='RANK' in os.environ
    if distributed:dist.init_process_group('nccl')
    rank,world=(dist.get_rank(),dist.get_world_size()) if distributed else (0,1)
    device=torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK",0))}');torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    # Compiled blocks run inside activation checkpoints; DDP-aware graph splitting would make the
    # recomputation save different tensors than the forward pass, so it is disabled.
    torch._dynamo.config.optimize_ddp=False;torch._dynamo.config.cache_size_limit=64
    run=json.loads(Path(a.config).read_text());t=run['train'];family=run['family']
    data=json.loads(Path(run['data']).read_text());root=Path(run['data']).parent
    out=Path(run['run_directory']);out.mkdir(parents=True,exist_ok=True)
    config=build_config(run,data['vocab_size']);sampler=sampler_for(run)
    torch.manual_seed(run['seed'])
    if run.get('mechanisms'):  # CTM-augmented RDT (ctm_transformer/ctm_rdt.py)
        assert family=='rdt';factory=ctm_rdt_factory(set(run['mechanisms']),t.get('checkpointing',False),t.get('backprop_steps'),t.get('compile',False),run.get('sync_pairs'))
    elif run.get('ctm_adaptations') is not None:  # CTM-LM with candidate fixes (ctm_transformer/ctm_lm_adapt.py)
        assert family=='ctm_lm';factory=adapted_factory(set(run['ctm_adaptations']),t.get('checkpointing',False),t.get('compile',False))
    else:factory=scaled_factory(family,t.get('checkpointing',False),t.get('backprop_steps'),t.get('compile',False))
    model=factory(config).to(device)
    train=TokenWindows([root/s for s in data['train']],t['seq_len'],run['seed'])
    valid=TokenWindows([root/s for s in data['validation']],t['seq_len'],0)
    offset=t.get('eval_offset',0);windows=lambda n:list(range(offset,min(offset+n,len(valid))))
    eval_indices,final_indices=windows(t['eval_windows']),windows(t.get('final_eval_windows',t['eval_windows']))
    params=[p for p in model.parameters() if p.requires_grad]
    if t.get('offload_optimizer'):optimizer=OffloadedAdamW(params,t['lr'],betas=tuple(t['betas']),weight_decay=t['weight_decay'],threads=t.get('cpu_threads',16))
    else:optimizer=torch.optim.AdamW(params,lr=t['lr'],betas=tuple(t['betas']),weight_decay=t['weight_decay'],fused=True)
    start=0;latest=out/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step']
        if rank==0 and (out/'metrics.jsonl').exists():  # drop rows logged after the checkpoint; they are recomputed identically
            rows=[r for r in (out/'metrics.jsonl').read_text().splitlines() if r.strip() and json.loads(r)['step']<=start]
            (out/'metrics.jsonl').write_text(''.join(r+'\n' for r in rows))
    if rank==0 and not (out/'run.json').exists():
        (out/'run.json').write_text(json.dumps({'run':run,'run_config_sha256':file_hash(a.config),'data_manifest_sha256':file_hash(run['data']),
            'effective_config':asdict(config),'parameters':sum(p.numel() for p in params),
            'non_embedding_parameters':sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('token_embedding','lm_head'))),
            'source_sha256':{s:file_hash(s) for s in SOURCES},'world_size':world,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(device),
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    net=DDP(model,device_ids=[device.index]) if distributed else model
    acc,micro=t['accumulation'],t['micro_batch'];tokens_per_step=world*micro*acc*t['seq_len']
    stop=min(t['total_steps'],a.max_steps) if a.max_steps else t['total_steps']
    log=open(out/'metrics.jsonl','a') if rank==0 else None
    if rank==0:print(json.dumps({'event':'start','step':start,'stop':stop,'tokens_per_step':tokens_per_step,'world':world}),flush=True)
    for step in range(start,stop):
        tick=time.perf_counter();lr=lr_at(step,t['lr'],t['warmup_steps'],t['total_steps'],t.get('lr_floor',0.1));set_lr(optimizer,lr)
        depth=step_depth(sampler,run['seed'],step) or (run.get('eval_depth') if family!='transformer' else None)
        total=torch.zeros((),device=device)
        for m in range(acc):
            x,y=train.batch(micro_indices(step,m,rank,world,micro,acc));x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            sync=net.no_sync() if distributed and m<acc-1 else contextlib.nullcontext()
            with sync:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(x,targets=y,max_thought_steps=depth)['loss']
                (loss/acc).backward()
            total+=loss.detach()/acc
        norm=torch.nn.utils.clip_grad_norm_(params,t['grad_clip'])
        if not torch.isfinite(norm):  # identical on every rank: the gradients are all-reduced
            if rank==0:(out/'diverged.json').write_text(json.dumps({'diverged':True,'step':step,'loss':float(total),'lr':lr})+'\n')
            break
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        if distributed:dist.all_reduce(total);total/=world
        done=step+1;row={'step':done,'tokens':done*tokens_per_step,'loss':float(total),'lr':lr,'grad_norm':float(norm),'depth':depth,
             'seconds':time.perf_counter()-tick,'peak_gib':torch.cuda.max_memory_allocated(device)/2**30}
        row['tokens_per_second']=tokens_per_step/row['seconds']
        if done%t['eval_interval']==0 or done==t['total_steps']:
            if rank==0:row['validation']=evaluate_lm(model,valid,final_indices if done==t['total_steps'] else eval_indices,t['eval_micro_batch'],device,family,
                depths=tuple(run.get('eval_depths',[run.get('eval_depth')])) if family=='rdt' else (None,))
            if distributed:dist.barrier()
        if rank==0:
            log.write(json.dumps(row)+'\n');log.flush()
            if done%t['log_interval']==0 or 'validation' in row:print(json.dumps(row),flush=True)
        if done%t['checkpoint_interval']==0 or done==t['total_steps']:
            if rank==0:
                payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':done,'run_config_sha256':file_hash(a.config)}
                atomic_save(payload,latest)
                if done%t.get('snapshot_interval',10**12)==0 or done==t['total_steps']:atomic_save(payload,out/f'step_{done:07d}.pt')
            if distributed:dist.barrier()
    if rank==0 and stop==t['total_steps'] and not (out/'diverged.json').exists():
        (out/'complete.json').write_text(json.dumps({'complete':True,'steps':stop,'tokens':stop*tokens_per_step})+'\n')
        if t.get('delete_checkpoint_on_complete'):latest.unlink(missing_ok=True)
    if distributed:dist.destroy_process_group()

if __name__=='__main__':main()
