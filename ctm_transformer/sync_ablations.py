"""Mechanism ablations of the Sync-RDT ``sync`` cell (research/SYNC_CONFIRMATION_PLAN.md, part 1).

Each ablation changes one property of the ``sync`` cell (no temporal MLP;
128 sparse-decay pairs over an 8-step post-update state history; a
zero-initialized additive term on every core attention query) and keeps
everything else fixed:

* ``shuffled``  the history is randomly permuted along time before
  synchronization, independently per position and step (training draws from
  the seeded global CUDA generator; evaluation uses a fixed generator, so it is
  reproducible).
* ``nodecay``   decay rates are frozen at r = 0.
* ``current``   synchronization uses only the newest state (history length 1).
* ``linear``    the 128 synchronization features are replaced by a linear
  projection of the flattened 8-step history (no pairwise products).
* ``state``     the same synchronization term is added to the recurrent state
  after the core, instead of to the attention queries.
* ``identity``  no change: reproduces the ``sync`` cell exactly (verification).

The frozen ``sync_rdt.SyncRDT`` code is not modified. This forward reimplements
its sync-only path with hooks; the identity kind is tested for exact equality.
"""
import torch
from torch import nn

from ctm_transformer.sync_rdt import HISTORY_LEN,SYNC_PAIRS,SyncRDT

KINDS=('identity','shuffled','nodecay','current','linear','state')
EVAL_SHUFFLE_SEED=20260926


class SyncAblation(SyncRDT):
    def __init__(self,config,kind):
        if kind not in KINDS:raise ValueError(f'Unknown sync ablation {kind}')
        super().__init__(config,history=False,sync=True)  # Base sync parameters are created exactly as in the sync cell.
        self.kind=kind;d=config.d_model
        if kind=='nodecay':self.sync.r.requires_grad_(False)
        if kind=='linear':
            self.history_features=nn.Linear(HISTORY_LEN*d,SYNC_PAIRS,bias=False)
            nn.init.normal_(self.history_features.weight,std=0.02)
        if kind=='state':
            # The query projections are unused; a single zero-initialized projection adds the term to the state.
            self.sync_query=None
            self.state_projection=nn.Linear(SYNC_PAIRS,d,bias=False);nn.init.zeros_(self.state_projection.weight)

    def _features(self,post_history):
        if self.kind=='current':return self.sync.compute(post_history[:,-1:])
        if self.kind=='linear':return self.history_features(post_history.reshape(post_history.shape[0],-1))
        if self.kind=='shuffled':
            generator=None
            if not self.training:
                generator=torch.Generator(device=post_history.device);generator.manual_seed(EVAL_SHUFFLE_SEED+self._step)
            noise=torch.rand(post_history.shape[:2],device=post_history.device,generator=generator)
            order=noise.argsort(dim=1)
            post_history=post_history.gather(1,order[...,None].expand_as(post_history))
        return self.sync.compute(post_history)

    def forward(self,input_ids,targets=None,max_thought_steps=None,return_all_logits=False):
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
        post_history=embedded.new_zeros(batch*length,HISTORY_LEN,d,dtype=torch.float32)
        tick_logits=[]
        for step in range(depth):
            self._step=step
            u=self.injection(torch.cat((x,embedded),dim=-1))
            features=self._features(post_history).view(batch,length,-1)
            if self.kind=='state':
                x=self._blocks(self.core,u)+self.state_projection(features).to(u.dtype)
            else:
                for block,projection in zip(self.core,self.sync_query):u=self._core_block(block,u,projection(features))
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
            result['loss']=torch.nn.functional.cross_entropy(logits.float().reshape(-1,self.config.vocab_size),targets.reshape(-1),ignore_index=-100)
        return result


def ablation_factory(kind):
    """Model factory for runner v3; the ablation name is recorded in each run manifest."""
    if kind not in KINDS:raise ValueError(f'Unknown sync ablation {kind}')
    def factory(config):return SyncAblation(config,kind)
    factory.__qualname__=f'ablation_factory[{kind}]'
    return factory
