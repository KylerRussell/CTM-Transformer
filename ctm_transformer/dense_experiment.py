"""Versioned shared research runner v3: injectable datasets and validation evaluator.

`ctm_transformer.experiment.train_experiment` (runner v2) builds its datasets
and validation metrics internally and is hash-frozen by earlier studies, so it
is left unchanged. This module reproduces its training loop exactly, taking
the training/evaluation datasets and the validation evaluator as arguments so
that new formats (the dense pointer task) can use it. Given the v2 datasets
and `evaluate_readouts`, it reproduces v2 losses, gradients, validation rows
and final weights exactly; `tests/test_dense_pointer.py` checks this against
the archived v2 source.
"""
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from ctm_transformer.experiment import file_hash,validate_training_config,objective_metadata,training_depth
from ctm_transformer.research import build_model,model_family,parameter_counts,block_applications

RUNNER='shared_research_v3'
SOURCES=['ctm_transformer/dense_experiment.py','ctm_transformer/experiment.py','ctm_transformer/research.py',
         'ctm_transformer/baselines.py','ctm_transformer/model.py','ctm_transformer/config.py',
         'ctm_transformer/algorithmic.py','ctm_transformer/dense_pointer.py','ctm_transformer/confidence_readout.py',
         'ctm_transformer/readout_selection.py','ctm_transformer/algorithmic_eval.py']


def batch_to(dataset,index,device):
    inputs,targets,real_tokens=dataset.batch(index)
    return inputs.to(device),targets.to(device),real_tokens


def train_dense_experiment(config,identity,seed,train_blocks,eval_blocks,evaluator,selection_readouts,
                           save_validation_checkpoints=True,data_policy=''):
    """Train with explicit dataset objects; `evaluator(model, data, config, device, policies)` scores validation."""
    validate_training_config(config)
    if not selection_readouts or len(set(selection_readouts))!=len(selection_readouts) or any(p not in ('final','confidence') for p in selection_readouts):
        raise ValueError('Selection readouts must be unique final/confidence policies')
    device=torch.device(config.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('A working CUDA device is required')
    torch.cuda.set_device(device)
    if config.dtype=='bfloat16' and not torch.cuda.is_bf16_supported():raise ValueError('This GPU does not support BF16')
    torch.set_num_threads(4)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    # Data ordering and depth budgets cannot depend on architecture RNG usage.
    data_rng=torch.Generator().manual_seed(seed+1)
    depth_rng=torch.Generator().manual_seed(seed+2)
    tokenizer=train_blocks.tokenizer
    if config.tokenizer!=tokenizer.name or tokenizer.n_vocab!=config.vocab_size:raise ValueError('Tokenizer and frozen vocabulary differ')
    train_data,eval_data=train_blocks.metadata,eval_blocks.metadata
    if train_blocks.ids&eval_blocks.ids:raise ValueError('Training and evaluation contain overlapping semantic instances')
    if train_data['task']!=eval_data['task']:raise ValueError('Training and evaluation task families differ')
    if train_data['sha256']==eval_data['sha256']:raise ValueError('Training and evaluation files have identical content')
    if len(train_blocks)<config.batch_size:raise ValueError('Training data must contain at least one full batch')
    out=Path(config.checkpoint_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty checkpoint_dir; resuming is not implemented')
    out.mkdir(parents=True,exist_ok=True)
    model=build_model(config).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,
        betas=(config.adam_beta1,config.adam_beta2),weight_decay=config.weight_decay)
    snapshot=json.dumps(tokenizer.snapshot(),sort_keys=True)
    (out/'tokenizer.json').write_text(snapshot)
    record={
        'schema_version':3,'runner':RUNNER,'data_format':'algorithmic',
        'created_utc':datetime.now(timezone.utc).isoformat(),'model_family':model_family(config),
        'config_identity':identity,'effective_config':asdict(config),'seed':seed,
        'data_seed':seed+1,'depth_seed':seed+2,
        'data':{'training':train_data,'evaluation':eval_data,'policy':data_policy},
        'tokenizer_sha256':hashlib.sha256(snapshot.encode()).hexdigest(),
        'code_sha256':{p:file_hash(p) for p in SOURCES},
        'evaluator':f'{evaluator.__module__}.{evaluator.__qualname__}',
        'packages':{p:version(p) for p in ('torch','numpy','tiktoken')},
        'gpu':torch.cuda.get_device_name(device),'cuda_runtime':torch.version.cuda,
        'precision':'FP32 parameters and AdamW moments; BF16 autocast' if config.dtype=='bfloat16' else 'FP32',
        'parameters':parameter_counts(model),'training_objective':objective_metadata(config),
        'checkpoint_selection':{'readouts':list(selection_readouts),'criterion':'minimum supervised-token validation CE across declared readouts and checkpoints; policy list breaks ties; earlier checkpoint retained on ties','save_every_validation':save_validation_checkpoints},
        'learning_rate_schedule':'linear warmup then cosine to 10% of peak; indexed by completed update count',
        'depth_sampling':'uniform inclusive range per update' if model_family(config)=='recurrent_depth' else 'fixed',
    }
    (out/'research_run.json').write_text(json.dumps(record,indent=2)+'\n')
    order=torch.randperm(len(train_blocks),generator=data_rng)
    cursor,tokens_seen,applications_seen,supervised_seen,examples_seen,padded_seen=0,0,0,0,0,0
    depth_counts=Counter()
    train_seconds,best_loss=0.0,float('inf')
    best_readout='final'
    best_by_policy={}
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    def save(name,step,readout='final'):
        payload={'model_family':model_family(config),'config':asdict(config),
            'model_state_dict':model.state_dict(),'optimizer_state_dicts':[optimizer.state_dict()],
            'step':step,'tokens_seen':tokens_seen,'config_identity':identity,'readout_policy':readout,
            'runner':RUNNER,'data_format':'algorithmic','supervised_tokens_seen':supervised_seen,'examples_seen':examples_seen,'depth_counts':dict(depth_counts)}
        temporary=out/(name+'.tmp')
        torch.save(payload,temporary)
        temporary.replace(out/name)

    with (out/'metrics.jsonl').open('w') as metrics:
        for step in range(1,config.max_steps+1):
            if config.warmup_steps and step<=config.warmup_steps:
                factor=step/config.warmup_steps
            else:
                progress=(step-config.warmup_steps)/(config.max_steps-config.warmup_steps)
                factor=0.1+0.9*0.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:
                group['lr']=config.learning_rate*factor
            depth=training_depth(config,depth_rng)
            depth_counts[depth]+=1
            if hasattr(model,'_train_step'):
                model._train_step.fill_(step-1)
            torch.cuda.synchronize(device)
            tick=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss=0.0
            accumulated_tick_loss=None
            microbatches=[]
            for _ in range(config.gradient_accumulation_steps):
                if cursor+config.batch_size>len(order):
                    order=torch.randperm(len(train_blocks),generator=data_rng)
                    cursor=0
                microbatches.append(batch_to(train_blocks,order[cursor:cursor+config.batch_size],device))
                cursor+=config.batch_size
            supervised_in_update=sum(int((targets!=-100).sum()) for _,targets,_ in microbatches)
            for inputs,targets,real_tokens in microbatches:
                target_count=int((targets!=-100).sum())
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.dtype=='bfloat16'):
                    model_output=model(inputs,targets=targets,max_thought_steps=depth)
                    loss=model_output['loss']
                weight=target_count/supervised_in_update
                (loss*weight).backward()
                accumulated_loss+=float(loss.detach())*weight
                if 'per_tick_loss' in model_output:
                    tick_loss=model_output['per_tick_loss'].detach()*weight
                    accumulated_tick_loss=tick_loss if accumulated_tick_loss is None else accumulated_tick_loss+tick_loss
                tokens_seen+=real_tokens
                padded_seen+=inputs.numel()
                supervised_seen+=target_count
                examples_seen+=inputs.shape[0]
                applications_seen+=block_applications(config,depth)*inputs.numel()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip,error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-tick
            train_seconds+=elapsed
            row={'step':step,'tokens_seen':tokens_seen,'supervised_tokens_seen':supervised_seen,
                 'examples_seen':examples_seen,'padded_token_positions':padded_seen,'loss':accumulated_loss,
                 'thought_steps':depth,'block_applications_per_sequence':block_applications(config,depth),
                 'lr':optimizer.param_groups[0]['lr'],'gradient_norm':float(norm),'update_seconds':elapsed}
            if accumulated_tick_loss is not None:
                row['per_tick_supervised_ce']=accumulated_tick_loss.float().tolist()
            if step%config.eval_interval==0 or step==config.max_steps:
                row['validation_readouts']=evaluator(model,eval_blocks,config,device,selection_readouts)
                selected_readout=min(selection_readouts,key=lambda p:row['validation_readouts'][p]['loss'])
                row['validation']={**row['validation_readouts'][selected_readout],'readout_policy':selected_readout}
                for policy in selection_readouts:
                    score=row['validation_readouts'][policy]['loss']
                    if score<best_by_policy.get(policy,float('inf')):
                        best_by_policy[policy]=score
                        save('best_'+policy+'.pt',step,policy)
                if save_validation_checkpoints:
                    save(f'step_{step:06d}.pt',step,selected_readout)
                if row['validation']['loss']<best_loss:
                    best_loss=row['validation']['loss']
                    best_readout=selected_readout
                    save('best.pt',step,best_readout)
            metrics.write(json.dumps(row)+'\n')
            metrics.flush()
            if step==1 or step%config.log_interval==0 or step==config.max_steps:
                print(json.dumps({k:v for k,v in row.items() if k!='validation_readouts'}),flush=True)
        save('final.pt',config.max_steps)
    summary={'complete':True,'model_family':model_family(config),'steps':config.max_steps,'runner':RUNNER,
        'tokens_seen':tokens_seen,'supervised_tokens_seen':supervised_seen,'examples_seen':examples_seen,
        'padded_token_positions':padded_seen,'token_block_applications':applications_seen,
        'depth_counts':dict(depth_counts),'training_seconds':train_seconds,
        'training_tokens_per_second':tokens_seen/train_seconds,
        'wall_seconds':time.perf_counter()-started,'best_validation_loss':best_loss,
        'best_readout_policy':best_readout,'best_validation_loss_by_readout':best_by_policy,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
        'parameters':parameter_counts(model),'config_identity':identity}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
