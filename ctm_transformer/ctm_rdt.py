"""CTM-augmented RDT: CTM mechanisms added to the language-model-scale RDT (RESEARCH_PLAN.md, CTM-inspired additions).

The base is ``lm_scale.ScaledBaseline`` (RoPE recurrent-depth model with chunked
cross-entropy, activation checkpointing, optional truncation and compile). The
core output x_t of each recurrence step is treated as the CTM's post-activation
state z_t, and three CTM mechanisms can be switched on:

* ``sync_query``: CTM's action synchronization, the recursive decayed pairwise
  products of the states x_0..x_{t-1} over P random neuron pairs (the same
  ``ctm_lm.Synchronization`` as CTM-LM), adds a term to every core attention
  query: q = W_q·norm(u) + W_sync·S_action. This is the LM-scale form of the
  Sync-RDT ``sync`` cell, with CTM's full decayed history instead of an
  8-step window.
* ``sync_readout``: CTM's output synchronization of x_1..x_T adds a term to the
  coda input: coda(norm(x_T) + W_out·S_out).
* ``learned_init``: a learned start state x_0 (zero in RDT), as in the CTM.

Every added projection, and the start state, is zero-initialized, so at
initialization the model computes exactly the RDT (tests/test_ctm_rdt.py).
The synchronization pairs are fixed buffers drawn from a seeded generator.
With synchronization on, activation checkpointing covers each whole recurrence
step, so only the step's inputs (state and synchronization accumulators) are
kept for backward.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ctm_transformer.baselines import DecoderBlock
from ctm_transformer.ctm_lm import Synchronization
from ctm_transformer.lm_scale import ScaledBaseline, chunked_cross_entropy
from ctm_transformer.positions import RoPECausalAttention, rope

MECHANISMS = ('sync_query', 'sync_readout', 'learned_init')


class QueryTermBlock(DecoderBlock):
    """The recurrent core's decoder block (sandwich norms, RoPE) with an optional additive query term."""
    def forward(self, x, q_extra=None):
        if q_extra is None:
            return super().forward(x)
        attn = self.attn
        h = self.attn_norm(x)
        batch, length, width = h.shape
        positions = attn.positions if attn.positions is not None else torch.arange(length, device=h.device)
        def split(t):
            return t.view(batch, length, attn.heads, width // attn.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(rope(split(attn.q(h) + q_extra), positions), rope(split(attn.k(h)), positions), split(attn.v(h)),
                                           is_causal=True, dropout_p=attn.dropout if attn.training else 0.0)
        x = self.attn_post(x + self.dropout(attn.out(y.transpose(1, 2).contiguous().view(batch, length, width))))
        y = self.ffn_norm(x)
        return self.ffn_post(x + self.dropout(self.down(F.silu(self.gate(y)) * self.up(y))))


class CTMAugmentedRDT(ScaledBaseline):
    def __init__(self, config, sync_query=False, sync_readout=False, learned_init=False, pairs=None, seed=0,
                 backprop_steps=None, compile_blocks=False):
        super().__init__(config, backprop_steps=backprop_steps)  # RDT parameters first: their initialization is unchanged
        if config.model_family != 'recurrent_depth':
            raise ValueError('The CTM-augmented RDT extends the recurrent-depth model')
        d = config.d_model
        self.pairs = pairs or d
        self.mechanisms = {'sync_query': sync_query, 'sync_readout': sync_readout, 'learned_init': learned_init}
        for block in self.core:
            assert type(block.attn) is RoPECausalAttention
            block.__class__ = QueryTermBlock
        generator = torch.Generator().manual_seed(seed)
        self.sync_action = Synchronization(d, self.pairs, generator) if sync_query else None
        self.sync_query = nn.ModuleList(nn.Linear(self.pairs, d, bias=False) for _ in self.core) if sync_query else None
        self.sync_out = Synchronization(d, self.pairs, generator) if sync_readout else None
        self.sync_readout = nn.Linear(self.pairs, d, bias=False) if sync_readout else None
        self.state_init = nn.Parameter(torch.zeros(d)) if learned_init else None
        for layer in [*(self.sync_query or []), *([self.sync_readout] if sync_readout else [])]:
            nn.init.zeros_(layer.weight)
        if compile_blocks:
            for block in (*self.prelude, *self.core, *self.coda):
                block.forward = torch.compile(block.forward)

    def _core(self, u, q_extras):
        for block, q in zip(self.core, q_extras):
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                u = checkpoint(block, u, q, use_reentrant=False)
            else:
                u = block(u, q)
        return u

    def _sync_step(self, x, embedded, alpha_a, beta_a, alpha_o, beta_o):
        """One recurrence step with synchronization; checkpointed as a whole, so only its inputs are kept for backward."""
        batch, length, d = x.shape
        q_extras = [None] * len(self.core)
        if self.sync_action is not None:
            s = Synchronization.read(alpha_a, beta_a).view(batch, length, -1)
            q_extras = [projection(s) for projection in self.sync_query]
        x = self.injection(torch.cat((x, embedded), dim=-1))
        for block, q in zip(self.core, q_extras):
            x = block(x, q)
        z = x.reshape(-1, d).float()
        if self.sync_action is not None:
            alpha_a, beta_a = self.sync_action.step(alpha_a, beta_a, z)
        if self.sync_out is not None:
            alpha_o, beta_o = self.sync_out.step(alpha_o, beta_o, z)
        return x, alpha_a, beta_a, alpha_o, beta_o

    def forward(self, input_ids, targets=None, max_thought_steps=None, return_all_logits=False):
        if return_all_logits:
            raise ValueError('The CTM-augmented RDT returns final logits only')
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError('Expected [batch, sequence] input within max_seq_len')
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError('Targets must be shifted and have the same shape as input_ids')
        depth = self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        if type(depth) is not int or depth < 1:
            raise ValueError('max_thought_steps must be a positive integer')
        batch, length = input_ids.shape
        d = self.config.d_model
        embedded = self._blocks(self.prelude, self.token_embedding(input_ids))
        x = self.state_init.to(embedded.dtype).expand_as(embedded) if self.state_init is not None else torch.zeros_like(embedded)
        synchronized = self.sync_action is not None or self.sync_out is not None
        sync = [None] * 4  # alpha_a, beta_a, alpha_o, beta_o; beta does not depend on the data, so it is one [1, P] row
        for i, module in ((0, self.sync_action), (2, self.sync_out)):
            if module is not None:
                sync[i] = module.products(x.reshape(-1, d).float())
                sync[i + 1] = sync[i].new_ones(1, self.pairs)
        for i in range(depth):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (self.backprop_steps is None or i >= depth - self.backprop_steps)):
                if not synchronized:
                    x = self._core(self.injection(torch.cat((x, embedded), dim=-1)), [None] * len(self.core))
                elif self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    x, *sync = checkpoint(self._sync_step, x, embedded, *sync, use_reentrant=False)
                else:
                    x, *sync = self._sync_step(x, embedded, *sync)
        h = self.final_norm(x)
        if self.sync_out is not None:
            h = h + self.sync_readout(Synchronization.read(sync[2], sync[3]).view(batch, length, -1))
        h = self.final_norm(self._blocks(self.coda, h))
        result = {'thought_steps': depth, 'block_applications': len(self.prelude) + depth * len(self.core) + len(self.coda)}
        if targets is None:
            result['logits'] = self.lm_head(h)
        else:
            result.update(logits=None, loss=chunked_cross_entropy(h, self.lm_head.weight, targets))
        return result


def ctm_rdt_factory(mechanisms, checkpointing=False, backprop_steps=None, compile=False, pairs=None):
    """Model factory; `mechanisms` is a subset of MECHANISMS."""
    unknown = set(mechanisms) - set(MECHANISMS)
    if unknown:
        raise ValueError(f'Unknown mechanisms {sorted(unknown)}')
    from dataclasses import replace
    def factory(config):
        return CTMAugmentedRDT(replace(config, gradient_checkpointing=checkpointing), **{m: m in mechanisms for m in MECHANISMS},
                               pairs=pairs, backprop_steps=backprop_steps, compile_blocks=compile)
    factory.__qualname__ = f'ctm_rdt_factory[{"+".join(sorted(mechanisms)) or "none"}]'
    return factory
