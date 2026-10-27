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

Added 2026-10-21, after CTsqL (``ctm_loss``) collapsed on both seeds:

* ``final_tick_loss`` (with ``ctm_loss`` only): the loss is the mean of the CTM loss and the final tick's
  cross-entropy, so that every sampled tick count keeps being trained as in Dans.
* ``detached_readouts`` (with ``ctm_loss`` and ``final_tick_loss``): the CTM loss reads every tick from the
  detached state, so it trains only the coda and the output head; the final tick's cross-entropy keeps the
  full gradient, so the recurrence is trained as in Dans. Tests whether the CTM loss's per-tick gradient into
  the recurrence causes the state growth seen in the CTsqL pilots.
* From Leon (arXiv 2608.28184, 2026-10-27), biologically inspired regulation of the thinking layers'
  feed-forward neurons, applied after each optimizer step by ``bio_step``:
  - ``homeostasis``: each SwiGLU gate neuron has a non-gradient threshold theta (a buffer, zero at
    initialization, so the untrained model is unchanged) subtracted from its gate pre-activation. After each
    update, theta <- clip(theta + 0.01 * (rate - 0.12), -2, 2), where rate is the fraction of (token, tick)
    instances in the step with gate pre-activation - theta > 0 (the paper's ReLU-active, here before the silu).
  - ``structural_plasticity``: the gate, up and down weights are used through a binary mask keeping the 35%
    largest-magnitude entries of each matrix, chosen at initialization and refreshed 30 times over training.
    It masks weights from the start, so with it the untrained model is not Dans.
  The paper trains one-hidden-layer MLPs with SGD and reports Adam as worse; here the optimizer is AdamW.
* ``record_stats`` (attribute, off by default): each forward stores detached diagnostics in ``self.stats``:
  the state's RMS after every tick and, under the CTM loss, the per-tick cross-entropy and the share of
  tokens selecting each tick.

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
from ctm_transformer.lm_scale import chunked_cross_entropy, ctm_selected_tick_loss
from ctm_transformer.positions import rope
from ctm_transformer.rdt_recipe import RecipeRDT

DANS = {'norm': 'prenorm', 'normalize': False, 'state_init': 'zeros', 'injection': 'add', 'coda_norm': 'none', 'state_norm': True}
MECHANISMS = ('neuron_models', 'sync_query', 'sync_output', 'attention_sink')
OBJECTIVES = ('ctm_loss', 'final_tick_loss', 'detached_readouts')
BIO = ('homeostasis', 'structural_plasticity')
HOMEOSTASIS_TARGET, HOMEOSTASIS_RATE, HOMEOSTASIS_LIMIT = 0.12, 0.01, 2.0
SP_DENSITY, SP_REFRESHES = 0.35, 30
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
                 final_tick_loss=False, detached_readouts=False, homeostasis=False, structural_plasticity=False, pairs=None, seed=0, **recipe):
        super().__init__(config, **recipe)
        if self.recipe['norm'] != 'prenorm' or not self.recipe['state_norm']:
            raise ValueError('The CTM Transformer builds on a pre-norm recipe with a state norm (Dans)')
        d, layers = config.d_model, len(self.core)
        pairs = pairs or d
        self.mechanisms = {'neuron_models': neuron_models, 'sync_query': sync_query, 'sync_output': sync_output, 'attention_sink': attention_sink,
                           'homeostasis': homeostasis, 'structural_plasticity': structural_plasticity}
        if final_tick_loss and not ctm_loss:
            raise ValueError('final_tick_loss is added to the CTM loss; alone it is the default objective')
        self.ctm_loss = ctm_loss
        if detached_readouts and not (ctm_loss and final_tick_loss):
            raise ValueError('detached_readouts needs ctm_loss and final_tick_loss (the final tick trains the recurrence)')
        self.final_tick_loss = final_tick_loss
        self.detached_readouts = detached_readouts
        self._hidden_readouts = False
        self.record_stats = False
        self.stats = {}
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
        if homeostasis:
            self.register_buffer('homeostatic_threshold', torch.zeros(layers, self.core[0].gate.out_features))
            self._activity = [None] * layers
            self._activity_count = [0] * layers
        if structural_plasticity:
            for name in ('gate', 'up', 'down'):
                self.register_buffer(f'sp_mask_{name}', torch.stack([self._top_k(getattr(block, name).weight) for block in self.core]))
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
        bio = self.mechanisms['homeostasis'] or self.mechanisms['structural_plasticity']
        pre = F.linear(u, self._ffn_weight(index, 'gate')) if bio else block.gate(u)
        act = F.silu(pre - self.homeostatic_threshold[index].to(pre.dtype)) if self.mechanisms['homeostasis'] else F.silu(pre)
        if self.mechanisms['neuron_models']:
            past = history if history else (torch.zeros_like(pre),) * (NLM_HISTORY - 1)
            stacked = torch.stack((*past, pre), dim=-1)  # [B, S, F, M]
            F_ = pre.shape[-1]
            neuron = neuron_models(self.nlms[index], stacked.reshape(B * S, F_, NLM_HISTORY)).view(B, S, F_)
            act = act + self.nlm_scale[index].to(act.dtype) * neuron.to(act.dtype)
        if bio:
            x = block.ffn_post(x + F.linear(act * F.linear(u, self._ffn_weight(index, 'up')), self._ffn_weight(index, 'down')))
        else:
            x = block.ffn_post(x + block.down(act * block.up(u)))
        return x, pre

    @staticmethod
    def _top_k(weight):
        k = max(1, int(SP_DENSITY * weight.numel()))
        threshold = weight.detach().abs().flatten().kthvalue(weight.numel() - k + 1).values
        return weight.detach().abs() >= threshold

    def _ffn_weight(self, index, name):
        weight = getattr(self.core[index], name).weight
        return weight * getattr(self, f'sp_mask_{name}')[index].to(weight.dtype) if self.mechanisms['structural_plasticity'] else weight

    def _record_activity(self, index, pre):
        active = ((pre.detach().float() - self.homeostatic_threshold[index]) > 0).float().mean(dim=tuple(range(pre.ndim - 1)))
        self._activity[index] = active if self._activity[index] is None else self._activity[index] + active
        self._activity_count[index] += 1

    @torch.no_grad()
    def bio_step(self, step, total_steps):
        """After an optimizer step: the homeostatic threshold update and, on schedule, the structural mask refresh."""
        if self.mechanisms['homeostasis']:
            for i, total in enumerate(self._activity):
                if total is not None:
                    rate = total / self._activity_count[i]
                    self.homeostatic_threshold[i].add_(HOMEOSTASIS_RATE * (rate - HOMEOSTASIS_TARGET)).clamp_(-HOMEOSTASIS_LIMIT, HOMEOSTASIS_LIMIT)
            self._activity = [None] * len(self.core)
            self._activity_count = [0] * len(self.core)
        if self.mechanisms['structural_plasticity'] and (step + 1) % max(1, total_steps // SP_REFRESHES) == 0:
            for name in ('gate', 'up', 'down'):
                getattr(self, f'sp_mask_{name}').copy_(torch.stack([self._top_k(getattr(block, name).weight) for block in self.core]))

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
        state_rms = []
        for i in range(depth):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (self.backprop_steps is None or i >= depth - self.backprop_steps)):
                h = self.injection(torch.cat((self.state_norm(x), injected), dim=-1))
                alpha_action, beta_action = sync_action if sync_action is not None else (None, None)
                for index in range(len(self.core)):
                    h, pre = self._checkpointed(self._thinking_layer, index, h, alpha_action, beta_action, *history[index])
                    if self.mechanisms['homeostasis'] and self.training and torch.is_grad_enabled():
                        self._record_activity(index, pre)  # outside the checkpoint, so a backward recompute is not counted
                    if self.mechanisms['neuron_models']:
                        past = history[index] if history[index] else (torch.zeros_like(pre),) * (NLM_HISTORY - 1)
                        history[index] = (*past[1:], pre)
                x = h
                if self.record_stats:
                    state_rms.append(x.detach().float().pow(2).mean().sqrt())
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
                read = (x.detach(), None if sync_out is None else tuple(t.detach() for t in sync_out)) if self.detached_readouts else (x, sync_out)
                hidden = self.final_norm(self._blocks(self.coda, self._coda_in(self._output(*read))))
                readouts.append(hidden if self._hidden_readouts else self.lm_head(hidden))
        if self.record_stats:
            self.stats = {'state_rms': torch.stack(state_rms)}
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
            last = self._recur(self.token_embedding(input_ids), depth, hidden)
        finally:
            self._hidden_readouts = False
        out = ctm_selected_tick_loss(torch.stack(hidden, dim=2), self.lm_head.weight, targets)
        loss = out['loss']
        if self.final_tick_loss:
            final = self.final_norm(last) if self.detached_readouts else hidden[-1]  # the full-gradient final tick
            loss = (loss + chunked_cross_entropy(final, self.lm_head.weight, targets)) / 2
        if self.record_stats:
            counts = [torch.bincount(out['selected_ticks'][:, j], minlength=depth).float() / len(out['selected_ticks']) for j in (0, 1)]
            self.stats.update(per_tick_ce=out['per_tick_loss'].detach(), min_loss_tick_share=counts[0], most_certain_tick_share=counts[1])
        return {'logits': None, 'loss': loss, 'thought_steps': depth,
                'block_applications': len(self.prelude) + depth * (len(self.core) + len(self.coda))}


def ctm_transformer_factory(mechanisms, recipe=DANS, checkpointing=False, backprop_steps=None):
    unknown = set(mechanisms) - set(MECHANISMS) - set(OBJECTIVES) - set(BIO)
    if unknown:
        raise ValueError(f'Unknown mechanisms {sorted(unknown)}')
    def factory(config):
        return CTMTransformer(replace(config, gradient_checkpointing=checkpointing), backprop_steps=backprop_steps,
                              **{m: m in mechanisms for m in MECHANISMS + OBJECTIVES + BIO}, **recipe)
    factory.__qualname__ = 'ctm_transformer_factory[' + ('+'.join(sorted(mechanisms)) or 'dans') + ']'
    return factory
