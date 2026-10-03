"""Language-model pretraining for the Transformer, RDT and CTM-LM arms (two GPUs, data parallel).

Components, each deterministic given the run's seed so that a resumed run
continues exactly where it stopped:

* ``TokenWindows``: memory-mapped uint16 token shards cut into non-overlapping
  windows of ``seq_len + 1`` tokens, presented in one seeded permutation.
  Optimizer step t uses windows [t·G, (t+1)·G) of that permutation (G = global
  batch in sequences), split over ranks and micro-batches.
* ``step_depth``: one recurrence depth per optimizer step, drawn from the
  arm's sampler with a generator seeded by (seed, step), so every rank and
  every micro-batch of the step uses the same depth, including after resume.
* ``OffloadedAdamW``: AdamW whose FP32 moments live in pinned CPU memory;
  gradients are copied to the CPU, the update runs there and parameters are
  copied back. It reproduces ``torch.optim.AdamW`` (tests/test_pretrain.py).
* ``lr_at``: linear warmup, then cosine decay to a floor fraction of the peak.
* ``evaluate_lm``: held-out cross-entropy per token. CTM-LM reports the final
  tick and the most certain tick (its inference readout); RDT reports each
  requested depth.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from ctm_transformer.ctm_lm import Synchronization


class TokenWindows:
    def __init__(self,shards,seq_len,seed):
        self.seq_len=seq_len;self.arrays=[np.memmap(p,dtype=np.uint16,mode='r') for p in shards]
        self.counts=np.array([(len(a)-1)//seq_len for a in self.arrays],dtype=np.int64)
        self.starts=np.concatenate(([0],np.cumsum(self.counts)))
        self.order=np.random.default_rng(seed).permutation(int(self.counts.sum()))

    def __len__(self):return len(self.order)

    def window(self,index):
        """Window `index` of the permutation: (input, target) of length seq_len, as int64 tensors."""
        w=int(self.order[index%len(self.order)]);shard=int(np.searchsorted(self.starts,w,side='right')-1)
        offset=(w-int(self.starts[shard]))*self.seq_len
        chunk=torch.from_numpy(self.arrays[shard][offset:offset+self.seq_len+1].astype(np.int64))
        return chunk[:-1],chunk[1:]

    def batch(self,indices):
        pairs=[self.window(i) for i in indices]
        return torch.stack([p[0] for p in pairs]),torch.stack([p[1] for p in pairs])


def micro_indices(step,micro,rank,world,micro_batch,accumulation):
    """Window indices for one micro-batch: the global batch of a step is laid out micro-batch-major, then rank."""
    global_batch=world*micro_batch*accumulation
    first=step*global_batch+(micro*world+rank)*micro_batch
    return list(range(first,first+micro_batch))


def step_depth(sampler,seed,step):
    if sampler is None:return None
    return sampler(torch.Generator().manual_seed(seed*1_000_003+step))


def lr_at(step,peak,warmup,total,floor=0.1):
    if step<warmup:return peak*(step+1)/warmup
    progress=min(1.0,(step-warmup)/max(1,total-warmup))
    return peak*(floor+(1-floor)*0.5*(1+math.cos(math.pi*progress)))


class OffloadedAdamW:
    """AdamW (decoupled weight decay, bias correction) with FP32 moments in pinned CPU memory."""
    def __init__(self,params,lr,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1,threads=None):
        self.params=[p for p in params if p.requires_grad];self.lr=lr;self.betas=betas;self.eps=eps;self.weight_decay=weight_decay;self.step_count=0
        self.master=[p.detach().float().cpu().pin_memory() for p in self.params]
        self.exp_avg=[torch.zeros_like(m) for m in self.master];self.exp_avg_sq=[torch.zeros_like(m) for m in self.master]
        self.grad=[torch.empty_like(m).pin_memory() for m in self.master]
        if threads:torch.set_num_threads(threads)

    @torch.no_grad()
    def step(self):
        self.step_count+=1;b1,b2=self.betas
        for p,g in zip(self.params,self.grad):g.copy_(p.grad if p.grad is not None else torch.zeros_like(p),non_blocking=True)
        torch.cuda.synchronize()
        bias1,bias2=1-b1**self.step_count,1-b2**self.step_count
        torch._foreach_mul_(self.master,1-self.lr*self.weight_decay)
        torch._foreach_lerp_(self.exp_avg,self.grad,1-b1)
        torch._foreach_mul_(self.exp_avg_sq,b2);torch._foreach_addcmul_(self.exp_avg_sq,self.grad,self.grad,1-b2)
        denominator=torch._foreach_sqrt(self.exp_avg_sq);torch._foreach_div_(denominator,math.sqrt(bias2));torch._foreach_add_(denominator,self.eps)
        torch._foreach_addcdiv_(self.master,self.exp_avg,denominator,-self.lr/bias1)
        for p,m in zip(self.params,self.master):p.copy_(m,non_blocking=True)
        torch.cuda.synchronize()

    def zero_grad(self,set_to_none=True):
        for p in self.params:p.grad=None

    def state_dict(self):
        return {'step_count':self.step_count,'lr':self.lr,'master':self.master,'exp_avg':self.exp_avg,'exp_avg_sq':self.exp_avg_sq}

    def load_state_dict(self,state):
        self.step_count=state['step_count'];self.lr=state['lr']
        for dst,src in (( self.master,state['master']),(self.exp_avg,state['exp_avg']),(self.exp_avg_sq,state['exp_avg_sq'])):
            for d,s in zip(dst,src):d.copy_(s)
        with torch.no_grad():
            for p,m in zip(self.params,self.master):p.copy_(m)

    @property
    def param_groups(self):return [{'lr':self.lr}]

    def set_lr(self,lr):self.lr=lr


def set_lr(optimizer,lr):
    if isinstance(optimizer,OffloadedAdamW):optimizer.set_lr(lr)
    else:
        for group in optimizer.param_groups:group['lr']=lr


@torch.no_grad()
def evaluate_lm(model,windows,indices,micro_batch,device,family,depths=(None,)):
    """Mean held-out cross-entropy per token. Returns {readout: loss}."""
    model.eval();totals={};count=0
    for i in range(0,len(indices),micro_batch):
        x,y=windows.batch(indices[i:i+micro_batch]);x,y=x.to(device),y.to(device);count+=y.numel()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if family=='ctm_lm':
                for name,value in ctm_readout_losses(model,x,y).items():totals[name]=totals.get(name,0.0)+value
            else:
                for depth in depths:
                    out=model(x,max_thought_steps=depth) if family=='rdt' else model(x)
                    name='final' if depth is None else f'depth_{depth}'
                    totals[name]=totals.get(name,0.0)+float(F.cross_entropy(out['logits'].float().reshape(-1,out['logits'].shape[-1]),y.reshape(-1),reduction='sum'))
    model.train()
    return {k:v/count for k,v in totals.items()}


def ctm_readout_losses(model,x,y):
    """Summed cross-entropy of CTM-LM's final tick and its most certain tick (label-free selection). A CTM-LM trained
    on a subset of ticks (sparse_tick_loss) also reports the most certain of its trained ticks."""
    out=model(x,return_all_logits=True);T=len(out['all_logits']);totals={'final_tick':0.0,'most_certain_tick':0.0}
    trained=None
    if getattr(model,'adaptations',{}).get('sparse_tick_loss'):
        from ctm_transformer.ctm_lm_adapt import sparse_ticks
        trained=torch.tensor(sparse_ticks(T),device=x.device);totals['most_certain_trained_tick']=0.0
    for b in range(x.shape[0]):
        logits=torch.stack([t[b].float() for t in out['all_logits']],dim=1)  # [S, T, V]
        logp=logits.log_softmax(-1);ce=-logp.gather(-1,y[b][:,None,None].expand(-1,T,1)).squeeze(-1)
        certainty=1+(logp.exp()*logp).sum(-1)/math.log(logp.shape[-1]);del logits,logp
        totals['final_tick']+=float(ce[:,-1].sum());totals['most_certain_tick']+=float(ce.gather(1,certainty.argmax(-1,keepdim=True)).sum())
        if trained is not None:
            pick=trained[certainty[:,trained].argmax(-1)];totals['most_certain_trained_tick']+=float(ce.gather(1,pick[:,None]).sum())
    return totals
