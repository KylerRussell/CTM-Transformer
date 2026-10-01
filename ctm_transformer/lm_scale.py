"""Language-model-scale training paths for the Transformer, RDT and CTM-LM (RoPE).

At 32k vocabulary and 1,024 tokens the frozen models materialize full FP32 logits;
CTM-LM materializes logits for every tick. These subclasses keep the models'
parameters, initialization and mathematics, and change only how the training
loss is computed and which activations are kept:

* ``chunked_cross_entropy``: mean token cross-entropy computed chunk by chunk;
  each chunk's logits are recomputed in the backward pass instead of stored.
* ``ctm_selected_tick_loss``: CTM's loss (mean of the minimum-loss tick's and the
  most certain tick's cross-entropy). Selecting the two ticks needs no gradient,
  so per-tick cross-entropy and certainty are computed chunk by chunk without
  gradient, and logits with gradient are recomputed only for the two selected
  ticks. Loss and gradients equal ``ctm_lm.ctm_loss`` (tests/test_lm_scale.py).
* activation checkpointing: per block for the Transformer and RDT (the frozen
  ``gradient_checkpointing`` switch), per tick for CTM-LM.
* optional truncated backpropagation for RDT (``backprop_steps=k``): the first
  depth - k recurrence steps run without gradient, as in Geiping et al. (2025).
  This changes the gradient, so it is a recipe choice. With k >= depth it is exact.
* optional ``torch.compile``: per block for the Transformer and RDT; for CTM-LM
  around the (checkpointed) tick, because compiling inside a checkpoint keeps
  the compiled graph's activations and defeats the checkpoint.

Without targets, or with ``return_all_logits``, the frozen forward runs unchanged.
"""
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ctm_transformer.ctm_lm import Synchronization
from ctm_transformer.positions import RoPEBaseline,RoPECTMLM,rope

CHUNK_ROWS=4096


def _logits(hidden,weight):
    """F.linear as autocast would compute it, with explicit casts. Autocast is disabled inside so that a checkpointed
    recomputation cannot differ from the forward pass through autocast's cache of cast weights."""
    device=hidden.device.type
    dtype=torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else hidden.dtype
    with torch.autocast(device_type=device,enabled=False):
        return F.linear(hidden.to(dtype),weight.to(dtype)).float()


def _cross_entropy_sum(hidden,weight,targets):
    return F.cross_entropy(_logits(hidden,weight),targets,reduction='sum')


def chunked_cross_entropy(hidden,weight,targets,chunk=CHUNK_ROWS):
    """Mean cross-entropy of `F.linear(hidden, weight)` over targets != -100, without storing the logits."""
    hidden=hidden.reshape(-1,hidden.shape[-1]);targets=targets.reshape(-1);keep=targets.ne(-100)
    hidden,targets=hidden[keep],targets[keep]
    total=hidden.new_zeros((),dtype=torch.float32)
    for i in range(0,len(targets),chunk):
        total=total+checkpoint(_cross_entropy_sum,hidden[i:i+chunk],weight,targets[i:i+chunk],use_reentrant=False)
    return total/len(targets)


def ctm_selected_tick_loss(readouts,weight,targets,chunk=CHUNK_ROWS):
    """CTM's loss from per-tick readouts [B, S, T, P] and the output head weight [V, P]."""
    T=readouts.shape[2];keep=targets.reshape(-1).ne(-100)
    r=readouts.reshape(-1,T,readouts.shape[-1])[keep];gold=targets.reshape(-1)[keep];n=len(gold)
    ce=torch.empty(n,T,device=r.device);certainty=torch.empty(n,T,device=r.device)
    rows=max(1,chunk//T)
    with torch.no_grad():
        for i in range(0,n,rows):
            logp=_logits(r[i:i+rows],weight).log_softmax(-1)  # [rows, T, V]
            ce[i:i+rows]=-logp.gather(-1,gold[i:i+rows,None,None].expand(-1,T,1)).squeeze(-1)
            certainty[i:i+rows]=1+(logp.exp()*logp).sum(-1)/math.log(logp.shape[-1])
            del logp
    selected=torch.stack((ce.argmin(-1),certainty.argmax(-1)),dim=1)  # [n, 2]
    chosen=r.gather(1,selected[...,None].expand(-1,-1,r.shape[-1]))      # [n, 2, P], with gradient
    loss=chunked_cross_entropy(chosen,weight,gold[:,None].expand(-1,2),chunk)  # mean over both choices = CTM's mean of the pair
    return {'loss':loss,'per_tick_loss':ce.mean(0),'certainty':certainty.mean(0),'selected_ticks':selected}


class ScaledBaseline(RoPEBaseline):
    """RoPE Transformer or RDT whose training loss never stores full logits."""
    def __init__(self,config,backprop_steps=None,compile_blocks=False):
        super().__init__(config);self.backprop_steps=backprop_steps
        if compile_blocks:
            for block in (*self.layers,*self.prelude,*self.core,*self.coda):block.forward=torch.compile(block.forward)
    def forward(self,input_ids,targets=None,max_thought_steps=None,return_all_logits=False):
        if targets is None or return_all_logits:return super().forward(input_ids,targets,max_thought_steps,return_all_logits)
        if input_ids.ndim!=2 or not 0<input_ids.shape[1]<=self.config.max_seq_len:raise ValueError('Expected [batch, sequence] input within max_seq_len')
        if targets.shape!=input_ids.shape:raise ValueError('Targets must be shifted and have the same shape as input_ids')
        depth=self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        if type(depth) is not int or depth<1:raise ValueError('max_thought_steps must be a positive integer')
        if self.config.model_family=='transformer' and depth!=1:raise ValueError('Standard Transformer depth is fixed; use T=1')
        x=self.token_embedding(input_ids)
        if self.injection is None:
            x=self._blocks(self.layers,x);applications=len(self.layers)
        else:
            embedded=self._blocks(self.prelude,x);x=torch.zeros_like(embedded)
            for i in range(depth):
                with torch.set_grad_enabled(torch.is_grad_enabled() and (self.backprop_steps is None or i>=depth-self.backprop_steps)):
                    x=self._blocks(self.core,self.injection(torch.cat((x,embedded),dim=-1)))
            x=self._blocks(self.coda,self.final_norm(x));applications=len(self.prelude)+depth*len(self.core)+len(self.coda)
        loss=chunked_cross_entropy(self.final_norm(x),self.lm_head.weight,targets)
        return {'logits':None,'loss':loss,'thought_steps':depth,'block_applications':applications}


class ScaledCTMLM(RoPECTMLM):
    """RoPE CTM-LM with CTM's loss over selected ticks only and optional per-tick activation checkpointing."""
    def __init__(self,config,checkpoint_ticks=False,compile_ticks=False,**kwargs):
        super().__init__(config,**kwargs);self.checkpoint_ticks=checkpoint_ticks
        self._compiled_tick=torch.compile(self._checkpointed_tick) if compile_ticks else None

    def _checkpointed_tick(self,*state):
        if self.checkpoint_ticks and torch.is_grad_enabled():return checkpoint(self._tick,*state,use_reentrant=False)
        return self._tick(*state)

    def _tick(self,z,history,alpha_a,beta_a,alpha_o,beta_o,keys,values,positions):
        B,S=keys.shape[0],keys.shape[2];d,H=self.config.d_model,self.config.n_heads;N=B*S
        q=rope(self.query(Synchronization.read(alpha_a,beta_a)).view(B,S,H,d//H).transpose(1,2),positions)
        o=F.scaled_dot_product_attention(q,keys,values,is_causal=True).transpose(1,2).reshape(N,d)
        pre=self.synapse(torch.cat((self.attn_out(o),z.to(o.dtype)),dim=-1))
        history=torch.cat((history[...,1:],pre.unsqueeze(-1).to(history.dtype)),dim=-1)
        z=self.nlm(history).float()
        alpha_a,beta_a=self.sync_action.step(alpha_a,beta_a,z);alpha_o,beta_o=self.sync_out.step(alpha_o,beta_o,z)
        return z,history,alpha_a,beta_a,alpha_o,beta_o,Synchronization.read(alpha_o,beta_o)

    def forward(self,input_ids,targets=None,max_thought_steps=None,return_all_logits=False):
        if targets is None or return_all_logits:return super().forward(input_ids,targets,max_thought_steps,return_all_logits)
        if input_ids.ndim!=2 or not 0<input_ids.shape[1]<=self.config.max_seq_len:raise ValueError('Expected [batch, sequence] input within max_seq_len')
        if targets.shape!=input_ids.shape:raise ValueError('Targets must be shifted and have the same shape as input_ids')
        T=self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        if type(T) is not int or T<1:raise ValueError('max_thought_steps must be a positive integer')
        B,S=input_ids.shape;d,D,H=self.config.d_model,self.config.d_latent,self.config.n_heads;N=B*S
        feats=self.features(input_ids);positions=torch.arange(S,device=input_ids.device)
        heads=lambda x:x.view(B,S,H,d//H).transpose(1,2)
        keys=rope(heads(self.key(feats)),positions);values=heads(self.value(feats))
        z=self.z_init.expand(N,D).float();history=self.history_init.expand(N,D,self.config.history_len)
        alpha_a,beta_a=self.sync_action.start(z);alpha_o,beta_o=self.sync_out.start(z)
        readouts=[]
        for _ in range(T):
            state=(z,history,alpha_a,beta_a,alpha_o,beta_o,keys,values,positions)
            out=(self._compiled_tick or self._checkpointed_tick)(*state)
            z,history,alpha_a,beta_a,alpha_o,beta_o,readout=out;readouts.append(readout.view(B,S,-1))
        result=ctm_selected_tick_loss(torch.stack(readouts,dim=2),self.lm_head.weight,targets)
        return {'logits':None,'thought_steps':T,**result}


def scaled_factory(model,checkpointing=False,backprop_steps=None,compile=False):
    """Model factory: model is transformer | rdt | ctm_lm; checkpointing enables activation checkpointing;
    backprop_steps truncates RDT backpropagation to the last k recurrence steps."""
    from dataclasses import replace
    if model not in ('transformer','rdt','ctm_lm'):raise ValueError(model)
    def factory(config):
        if model=='ctm_lm':return ScaledCTMLM(config,checkpoint_ticks=checkpointing,compile_ticks=compile)
        return ScaledBaseline(replace(config,gradient_checkpointing=checkpointing),backprop_steps=backprop_steps if model=='rdt' else None,compile_blocks=compile)
    if backprop_steps is not None and model!='rdt':raise ValueError('Truncated backpropagation is defined for RDT only')
    factory.__qualname__=f'scaled_factory[{model},{"checkpointing" if checkpointing else "plain"}'+(f',backprop{backprop_steps}' if backprop_steps else '')+(',compiled' if compile else '')+']'
    return factory
