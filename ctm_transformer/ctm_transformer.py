"""The CTM Transformer: a Transformer adapted with CTM and CMM mechanisms (research/CTM_TRANSFORMER_DESIGN.md, 2026-10-14).

Starts from Dans (``rdt_recipe.RecipeRDT`` with pre-norm blocks, additive injection, no coda norm
and a state norm), which at one tick is exactly the 8-layer Transformer. The thinking group
(the core blocks) runs for T internal ticks. Three mechanisms adapt its own components, each
zero-initialized so that the untrained model equals Dans at any tick count:

* ``neuron_models``: in each thinking layer's SwiGLU feed-forward network, every hidden neuron
  keeps the history of its gate pre-activation over the last ``NLM_HISTORY`` ticks, and a
  private per-neuron MLP (``ctm_lm.NeuronLevelModels``) maps it to an activation that is added
  to silu(gate) with a per-neuron scale initialized to zero.
* ``sync_query``: each thinking layer's attention query is the Transformer's query plus a
  zero-initialized projection of the action synchronization of earlier ticks.
* ``sync_output``: the output synchronization is added to the state before the coda through a
  zero-initialized projection, so it reaches the logits through the residual stream.

Added 2026-10-18 (item 3 of the CTM Transformer plan):

* ``attention_sink`` (CMM): each thinking layer's attention also sees a learned per-head sink key
  with a zero value, so a tick can attend to nothing. The sink output is mixed in with a per-head
  gate initialized to zero, so the untrained model is unchanged.
* ``ctm_loss`` (CTM, a training objective, not zero-initialized): the loss is the CTM's, per token
  the mean of the cross-entropy at the minimum-loss tick and at the most certain tick, over every
  tick's readout (``lm_scale.ctm_selected_tick_loss``), in place of the final tick's cross-entropy.

Synchronization (``TickSynchronization``): random pairs of the normalized state's neurons, each
a decayed running average of the products of its two neurons over ticks, read as alpha / beta
(CMM's normalization, independent of the tick count). The decay is r = softplus(rho), so it has
a gradient everywhere, unlike the CTM's clamp at 0, where our CTM-LM's decays stuck.
"""
import math

import torch
from dataclasses import replace
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ctm_transformer.ctm_lm import NeuronLevelModels
from ctm_transformer.lm_scale import ctm_selected_tick_loss
from ctm_transformer.positions import rope
from ctm_transformer.rdt_recipe import RecipeRDT

DANS = {'norm': 'prenorm', 'normalize': False, 'state_init': 'zeros', 'injection': 'add', 'coda_norm': 'none', 'state_norm': True}
MECHANISMS = ('neuron_models', 'sync_query', 'sync_output', 'attention_sink')
OBJECTIVES = ('ctm_loss',)
NLM_HISTORY = 4  # ticks of pre-activation history per feed-forward neuron
NLM_HIDDEN = 4   # hidden width of each neuron's private MLP (4, not 8: per-token neuron MLPs are memory-bound, 2026-10-14)
DECAY_INIT = 0.01


def _neuron_mlp(x, w1, b1, w2, b2):
    """ctm_lm.NeuronLevelModels' computation (M -> 2h -GLU-> h -> 2 -GLU-> 1 per neuron) written elementwise, so that
    torch.compile fuses it into one kernel. x: [N, neurons, M]; weights [neurons, in, out]."""
    h = F.glu((x.unsqueeze(-1) * w1).sum(-2) + b1, dim=-1)
    return F.glu((h.unsqueeze(-1) * w2).sum(-2) + b2, dim=-1).squeeze(-1)


_compiled_neuron_mlp = None


def neuron_models(nlm, x):
    """Apply a NeuronLevelModels module: the fused compiled kernel on CUDA, the module itself elsewhere."""
    global _compiled_neuron_mlp
    if not x.is_cuda:
        return nlm(x.float())
    if _compiled_neuron_mlp is None:
        _compiled_neuron_mlp = torch.compile(_neuron_mlp)
    return _compiled_neuron_mlp(x.float(), nlm.first.weight, nlm.first.bias, nlm.second.weight, nlm.second.bias)


class TickSynchronization(nn.Module):
    """Decayed running averages of products of random neuron pairs over ticks, read as alpha / beta."""
    def __init__(self, neurons, pairs, generator):
        super().__init__()
        self.register_buffer('left', torch.randint(0, neurons, (pairs,), generator=generator))
        self.register_buffer('right', torch.randint(0, neurons, (pairs,), generator=generator))
        self.rho = nn.Parameter(torch.full((pairs,), math.log(math.expm1(DECAY_INIT))))  # r = softplus(rho)
        # The pair products as two matrix multiplies with one-hot selectors (twice as fast as gathering by index).
        self.register_buffer('select_left', F.one_hot(self.left, neurons).T.float(), persistent=False)
        self.register_buffer('select_right', F.one_hot(self.right, neurons).T.float(), persistent=False)

    def products(self, z):
        return (z @ self.select_left) * (z @ self.select_right)

    def step(self, state, z):
        """z: [..., neurons] (float32). state: None or (alpha [..., pairs], beta [pairs])."""
        products = self.products(z)
        if state is None:
            return products, torch.ones_like(self.rho)
        keep = torch.exp(-F.softplus(self.rho))
        alpha, beta = state
        return keep * alpha + products, keep * beta + 1

    @staticmethod
    def read(state):
        alpha, beta = state
        return alpha / beta


class CTMTransformer(RecipeRDT):
    def __init__(self, config, neuron_models=False, sync_query=False, sync_output=False, attention_sink=False, ctm_loss=False,
                 pairs=None, seed=0, **recipe):
        super().__init__(config, **recipe)
        if self.recipe['norm'] != 'prenorm' or not self.recipe['state_norm']:
            raise ValueError('The CTM Transformer builds on a pre-norm recipe with a state norm (Dans)')
        d, layers = config.d_model, len(self.core)
        pairs = pairs or d
        self.mechanisms = {'neuron_models': neuron_models, 'sync_query': sync_query, 'sync_output': sync_output, 'attention_sink': attention_sink}
        self.ctm_loss = ctm_loss
        self._hidden_readouts = False
        generator = torch.Generator().manual_seed(seed)
        if neuron_models:
            hidden = self.core[0].gate.out_features
            self.nlms = nn.ModuleList(NeuronLevelModels(hidden, NLM_HISTORY, NLM_HIDDEN) for _ in range(layers))
            self.nlm_scale = nn.Parameter(torch.zeros(layers, hidden))
        if sync_query:
            self.sync_action = TickSynchronization(d, pairs, generator)
            self.sync_queries = nn.ModuleList(nn.Linear(pairs, d, bias=False) for _ in range(layers))
            for layer in self.sync_queries:
                nn.init.zeros_(layer.weight)
        if attention_sink:
            heads = self.core[0].attn.heads
            self.sink_keys = nn.Parameter(torch.zeros(layers, heads, d // heads))
            self.sink_gate = nn.Parameter(torch.zeros(layers, heads))
        if sync_output:
            self.sync_out = TickSynchronization(d, pairs, generator)
            self.sync_projection = nn.Linear(pairs, d, bias=False)
            nn.init.zeros_(self.sync_projection.weight)

    # One thinking layer: the pre-norm block with the query and feed-forward adaptations.
    def _thinking_layer(self, index, x, alpha_action, beta_action, *history):
        # The action synchronization of earlier ticks is read here (alpha / beta) so that a checkpoint stores only the
        # running sum, which is shared by the layers.
        block = self.core[index]
        q_extra = None
        if alpha_action is not None:
            q_extra = self.sync_queries[index]((alpha_action / beta_action).to(x.dtype))
        attn = block.attn
        B, S, W = x.shape
        H = attn.heads
        h = block.attn_norm(x)
        q = attn.q(h)
        if q_extra is not None:
            q = q + q_extra.to(q.dtype)
        split = lambda t: t.view(B, S, H, W // H).transpose(1, 2)
        positions = attn.positions if getattr(attn, 'positions', None) is not None else torch.arange(S, device=x.device)
        q, k, v = rope(split(q), positions), rope(split(attn.k(h)), positions), split(attn.v(h))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.mechanisms['attention_sink']:
            # The sink is prepended as key 0 (zero value) with a dummy query 0, so that under the causal mask every real
            # query sees the sink and its own causal prefix.
            zero = torch.zeros_like(q[:, :, :1])
            sink = self.sink_keys[index][None, :, None, :].to(k.dtype).expand(B, -1, 1, -1)
            y_sink = F.scaled_dot_product_attention(torch.cat((zero, q), 2), torch.cat((sink, k), 2), torch.cat((zero, v), 2), is_causal=True)[:, :, 1:]
            y = y + self.sink_gate[index][None, :, None, None].to(y.dtype) * (y_sink - y)
        x = block.attn_post(x + attn.out(y.transpose(1, 2).contiguous().view(B, S, W)))
        u = block.ffn_norm(x)
        pre = block.gate(u)
        act = F.silu(pre)
        if self.mechanisms['neuron_models']:
            past = history if history else (torch.zeros_like(pre),) * (NLM_HISTORY - 1)
            stacked = torch.stack((*past, pre), dim=-1)  # [B, S, F, M]
            F_ = pre.shape[-1]
            neuron = neuron_models(self.nlms[index], stacked.reshape(B * S, F_, NLM_HISTORY)).view(B, S, F_)
            act = act + self.nlm_scale[index].to(act.dtype) * neuron.to(act.dtype)
        x = block.ffn_post(x + block.down(act * block.up(u)))
        return x, pre

    def _checkpointed(self, function, *args):
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def _sync_sums(self, x, *previous):
        """The new running sums alpha for each enabled synchronization (action, then output), from the state x.
        Recomputed in the backward pass when checkpointing, so the normalized state and the products are not stored."""
        z = self.state_norm(x).float()
        syncs = [s for s, on in ((getattr(self, 'sync_action', None), self.mechanisms['sync_query']),
                                 (getattr(self, 'sync_out', None), self.mechanisms['sync_output'])) if on]
        sums = []
        for k, sync in enumerate(syncs):
            products = sync.products(z)
            sums.append(products if not previous else torch.exp(-F.softplus(sync.rho)) * previous[k] + products)
        return tuple(sums)

    def _output(self, x, sync_out):
        if sync_out is None:
            return x
        return x + self.sync_projection(TickSynchronization.read(sync_out).to(x.dtype))

    def _recur(self, x, depth, readouts=None):
        embedded = self._blocks(self.prelude, x)
        injected = self.inject_norm(embedded)
        x = self._state(embedded)
        history = [() for _ in self.core]
        sync_action = sync_out = None  # (alpha, beta) pairs
        for i in range(depth):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (self.backprop_steps is None or i >= depth - self.backprop_steps)):
                h = self.injection(torch.cat((self.state_norm(x), injected), dim=-1))
                alpha_action, beta_action = sync_action if sync_action is not None else (None, None)
                for index in range(len(self.core)):
                    h, pre = self._checkpointed(self._thinking_layer, index, h, alpha_action, beta_action, *history[index])
                    if self.mechanisms['neuron_models']:
                        past = history[index] if history[index] else (torch.zeros_like(pre),) * (NLM_HISTORY - 1)
                        history[index] = (*past[1:], pre)
                x = h
                enabled = [(name, sync) for name, sync in (('sync_query', sync_action), ('sync_output', sync_out)) if self.mechanisms[name]]
                if enabled:
                    previous = tuple(state[0] for _, state in enabled) if i > 0 else ()
                    sums = iter(self._checkpointed(self._sync_sums, x, *previous))
                    for name, state in enabled:
                        module = self.sync_action if name == 'sync_query' else self.sync_out
                        beta = torch.ones_like(module.rho) if state is None else torch.exp(-F.softplus(module.rho)) * state[1] + 1
                        if name == 'sync_query':
                            sync_action = (next(sums), beta)
                        else:
                            sync_out = (next(sums), beta)
            if readouts is not None:
                hidden = self.final_norm(self._blocks(self.coda, self._coda_in(self._output(x, sync_out))))
                readouts.append(hidden if self._hidden_readouts else self.lm_head(hidden))
        return self._blocks(self.coda, self._coda_in(self._output(x, sync_out)))

    def forward(self, input_ids, targets=None, max_thought_steps=None, return_all_logits=False):
        if not (self.ctm_loss and targets is not None and not return_all_logits):
            return super().forward(input_ids, targets, max_thought_steps, return_all_logits)
        if targets.shape != input_ids.shape:
            raise ValueError('Targets must be shifted and have the same shape as input_ids')
        depth = self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        hidden = []
        self._hidden_readouts = True
        try:
            self._recur(self.token_embedding(input_ids), depth, hidden)
        finally:
            self._hidden_readouts = False
        out = ctm_selected_tick_loss(torch.stack(hidden, dim=2), self.lm_head.weight, targets)
        return {'logits': None, 'loss': out['loss'], 'thought_steps': depth,
                'block_applications': len(self.prelude) + depth * (len(self.core) + len(self.coda))}


def ctm_transformer_factory(mechanisms, recipe=DANS, checkpointing=False, backprop_steps=None):
    unknown = set(mechanisms) - set(MECHANISMS) - set(OBJECTIVES)
    if unknown:
        raise ValueError(f'Unknown mechanisms {sorted(unknown)}')
    def factory(config):
        return CTMTransformer(replace(config, gradient_checkpointing=checkpointing), backprop_steps=backprop_steps,
                              **{m: m in mechanisms for m in MECHANISMS + OBJECTIVES}, **recipe)
    factory.__qualname__ = 'ctm_transformer_factory[' + ('+'.join(sorted(mechanisms)) or 'dans') + ']'
    return factory
