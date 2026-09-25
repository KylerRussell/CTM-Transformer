"""Sync-RDT: CTM mechanisms inside the recurrent-depth scaffold (research/SYNC_RDT_DESIGN.md).

The recurrent-depth baseline is kept unchanged. Two CTM mechanisms act inside
its recurrent core, per position and per recurrence step, treating the core
output's d channels as CTM neurons:

* ``history`` (A): CTM's independent neuron-level temporal MLPs read each
  channel's last H core outputs and blend into the state through CTM's gate:
  ``s_t = σ(g)·tanh(NLM(p_{t-H+1..t})) + (1-σ(g))·p_t``. CTM's initialization is kept:
  g=0, so the state starts near 0.5·p_t (the NLM output starts near zero).
* ``sync``    (B): CTM's sparse-decay synchronization of the post-update state
  history adds a zero-initialized term to every core attention query:
  ``q = W_q·norm(u) + W_sync·sync_{t-1}``.

Histories are zero-filled before the first step, as in CTM. With both switches
off the model *is* the recurrent-depth baseline: same parameters, same
initialization and the baseline's own forward code, which the tests check
exactly. CTM's ``NeuronLevelModels`` and ``SynchronizationComputer`` are reused
without modification.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ctm_transformer.baselines import BaselineTransformer
from ctm_transformer.model import NeuronLevelModels, SynchronizationComputer

CELLS={'rdt':(False,False),'history':(True,False),'sync':(False,True),'sync_rdt':(True,True)}
HISTORY_LEN,NLM_HIDDEN,SYNC_PAIRS=8,16,128


class SyncRDT(BaselineTransformer):
    def __init__(self,config,history=False,sync=False):
        super().__init__(config)  # Baseline parameters are created first, so their initialization is unchanged.
        if config.model_family!='recurrent_depth':raise ValueError('Sync-RDT extends the recurrent-depth baseline')
        if config.gradient_checkpointing:raise ValueError('Sync-RDT does not implement gradient checkpointing')
        self.use_history,self.use_sync=history,sync
        d=config.d_model
        self.nlm=NeuronLevelModels(d,HISTORY_LEN,NLM_HIDDEN,nlm_groups=d,dropout=0.0) if history else None
        if sync:
            self.sync=SynchronizationComputer(d,HISTORY_LEN,method='sparse_decay',sync_sparse_pairs=SYNC_PAIRS)
            self.sync_query=nn.ModuleList(nn.Linear(SYNC_PAIRS,d,bias=False) for _ in self.core)
            for layer in self.sync_query:nn.init.zeros_(layer.weight)
        else:
            self.sync=None;self.sync_query=None

    @staticmethod
    def _attention(attn,x,q_extra):
        batch,length,width=x.shape
        def split(t):return t.view(batch,length,attn.heads,width//attn.heads).transpose(1,2)
        q=attn.q(x)+q_extra
        y=F.scaled_dot_product_attention(split(q),split(attn.k(x)),split(attn.v(x)),
            is_causal=True,dropout_p=attn.dropout if attn.training else 0.0)
        return attn.out(y.transpose(1,2).contiguous().view(batch,length,width))

    def _core_block(self,block,x,q_extra):
        # Same computation as DecoderBlock.forward, with an extra query term.
        x=block.attn_post(x+block.dropout(self._attention(block.attn,block.attn_norm(x),q_extra)))
        y=block.ffn_norm(x)
        return block.ffn_post(x+block.dropout(block.down(F.silu(block.gate(y))*block.up(y))))

    def forward(self,input_ids,targets=None,max_thought_steps=None,return_all_logits=False):
        if not (self.use_history or self.use_sync):
            return super().forward(input_ids,targets,max_thought_steps,return_all_logits)
        if input_ids.ndim!=2 or not 0<input_ids.shape[1]<=self.config.max_seq_len:
            raise ValueError('Expected [batch, sequence] input within max_seq_len')
        depth=self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        if type(depth) is not int or depth<1:raise ValueError('max_thought_steps must be a positive integer')
        batch,length=input_ids.shape;d=self.config.d_model
        x=self.token_embedding(input_ids)
        if self.pos_embedding is not None:
            x=x+self.pos_embedding(torch.arange(length,device=input_ids.device))
        embedded=self._blocks(self.prelude,x)
        x=torch.zeros_like(embedded)
        # Per-position FIFO histories over recurrence steps, zero-filled before step 1.
        # Histories are kept in FP32 so autocast dtypes cannot mix inside the buffers.
        pre_history=embedded.new_zeros(batch*length,HISTORY_LEN,d,dtype=torch.float32)
        post_history=embedded.new_zeros(batch*length,HISTORY_LEN,d,dtype=torch.float32)
        tick_logits=[]
        for _ in range(depth):
            u=self.injection(torch.cat((x,embedded),dim=-1))
            if self.use_sync:
                sync=self.sync.compute(post_history).view(batch,length,-1)
                for block,projection in zip(self.core,self.sync_query):u=self._core_block(block,u,projection(sync))
            else:
                u=self._blocks(self.core,u)
            if self.use_history:
                pre_history=torch.cat((pre_history[:,1:],u.reshape(batch*length,1,d).float()),dim=1)
                x=self.nlm(pre_history).view(batch,length,d)
            else:
                x=u
            post_history=torch.cat((post_history[:,1:],x.reshape(batch*length,1,d).float()),dim=1)
            if return_all_logits:
                decoded=self._blocks(self.coda,self.final_norm(x))
                tick_logits.append(self.lm_head(self.final_norm(decoded)))
        x=self._blocks(self.coda,self.final_norm(x))
        logits=self.lm_head(self.final_norm(x))
        result={'logits':logits,'thought_steps':depth,
                'block_applications':len(self.prelude)+depth*len(self.core)+len(self.coda)}
        if return_all_logits:result['all_logits']=tick_logits
        if targets is not None:
            if targets.shape!=input_ids.shape:raise ValueError('Targets must be shifted and have the same shape as input_ids')
            result['loss']=F.cross_entropy(logits.float().reshape(-1,self.config.vocab_size),targets.reshape(-1),ignore_index=-100)
        return result


def cell_factory(cell):
    """Model factory for runner v3; the cell name is recorded in each run manifest."""
    if cell not in CELLS:raise ValueError(f'Unknown Sync-RDT cell {cell}')
    history,sync=CELLS[cell]
    def factory(config):return SyncRDT(config,history=history,sync=sync)
    factory.__qualname__=f'cell_factory[{cell}]'
    return factory
