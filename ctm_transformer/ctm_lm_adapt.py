"""Candidate fixes for CTM-LM's failure to use its input at language-model scale (research/CTM_LM_DESIGN.md, 2026-10-02).

The faithful CTM-LM reads the sequence only through its tick attention. At
initialization that attention is nearly uniform (tick-0 query rms about 0.01) and its
query and key weights receive gradients near 1e-6. Uniform causal attention over
1,024 tokens gives each position the mean of its prefix, which carries almost no
information about the current token. The model learns only token frequencies and
its backbone collapses to a token-independent vector.

* ``unit_query`` (A, initialization only): the query projection is rescaled so
  that the tick-0 query has unit rms. The tick-0 query depends only on the
  learned start state, so the scale is data-independent. The architecture is the
  faithful CTM-LM.
* ``observe_token`` (B, an adaptation for sequence models): each position's CTM
  also receives its own backbone feature. The synapse input becomes
  [attention output, backbone feature, state] in place of [attention output,
  state]. This departs from the CTM's rule that input arrives only through
  attention.
* ``mean_tick_loss`` (C, a loss change): training minimizes the mean
  cross-entropy over all ticks in place of CTM's loss (the mean of the
  minimum-loss tick, selected with the label, and the most certain tick). At
  language-model scale CTM's loss is satisfied by ticks that make different
  confident guesses: on held-out data the label-selected tick scores 5.3-5.8
  nats while every label-free readout scores 7.3-7.8 (2026-10-02 probes).

Further candidates (2026-10-02, after B + C passed):

* ``query_feature`` (E): each position's tick-attention query also gets a
  projection of its own backbone feature, q = W_q S_action + W_f f_i. Input
  still enters only through attention.
* ``token_start`` (D): the start state is token-conditioned,
  z_0 = z_init + W_s f_i, through the CTM's own start-state mechanism.
* ``final_tick_loss`` / ``certainty_loss``: alternatives to C. Train the final
  tick only, or CTM's label-free half alone (the most certain tick, selected
  without gradient).
* ``token_history`` (H): the initial pre-activation history is token-conditioned too,
  history_0 = history_init + (W_h f_i) broadcast over the memory slots.
* ``decay_softplus`` / ``decay_spread`` (2026-10-06): the synchronization decay r is
  clamped to [0, 15] and initialized at exactly 0, so a decay nudged below 0 receives
  no gradient and stays at "no decay" forever. In trained models 97-99% of decays
  were stuck there. ``decay_softplus`` uses r = softplus(rho), with rho initialized
  so that r is about 0.01. ``decay_spread`` keeps the clamp and initializes r
  uniformly in [0, 3].
* ``sparse_tick_loss``: the mean cross-entropy over every (T/4)-th tick ending at
  the final tick. It keeps periodic per-tick predictions at a lower cost than
  the mean over all ticks.
* ``state_readout``: the output reads the neuron state as well as the output
  synchronization, readout = S_out + W_r z_t. This tests whether the
  synchronization readout is a bottleneck.

Training tests of tick use (research/CTM_TICK_DIAGNOSTICS.md, 2026-10-06):

* ``improvement_loss``: per-tick cross-entropy weighted in proportion to the tick
  index (weights sum to 1), plus a hinge max(0, CE_t - sg(CE_{t-1})) averaged over
  tokens and summed over ticks 2..T, so that a tick worse than the one before is
  penalized. Used with a randomized tick count (the run's depth sampler).
* ``cross_position``: at each tick the keys and values are the backbone's plus a
  projection of every causal position's current state z, so refinement at one
  position reaches later positions (recurrent depth with CTM dynamics). Both
  projections are zero-initialized: at initialization the model equals D + sparse.
* ``feature_head``: the logits are a linear map of the backbone feature f_i, and the
  ticks are not run. With a backbone loaded from a trained CTM-LM and frozen
  (pretrain run keys ``init_from`` and ``freeze_prefixes``), this measures how much
  of the CTM's prediction a plain readout recovers.

From Continuous Memory Machines (Regan et al., Sakana AI, arXiv 2610.07907), 2026-10-07:

* ``sync_mean``: synchronization is read as alpha / beta, a properly normalized
  exponential moving average whose magnitude does not change with the tick
  count, in place of the CTM's alpha / sqrt(beta) (their Appendix A.3).
* ``attention_sink``: the tick attention gets a learned key with a zero value
  (initialized to zero), so a tick can attend to nothing.
* ``tick_memory``: CMM's joint memory update. Each tick, one pre-norm Transformer
  block reads [sink; long-term memory slots; the pre-activation history, one token
  per tick] with learned slot embeddings on queries and keys only and a zero sink
  value. The neuron-level models read its updated history, and the long-term slots
  carry its output to the next tick. A residual gate initialized to zero makes the
  block the identity at initialization.

Against drift past the best tick (deep-research review, 2026-10-08):

* ``gated_state``: z_{t+1} = z_t + g * (NLM output - z_t), with a learned per-neuron
  gate g = sigmoid(c) initialized at 0.5. Re-injection of f_i every tick is the
  existing ``observe_token``.
* ``decay_data_clamp``: the official CTM code's decay handling. The decays are clamped
  to [0, 15] in place before each forward pass, and exp(-r) is computed on the parameter,
  so the gradient reaches r even at 0.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

from ctm_transformer.ctm_lm import Synchronization, UNetSynapse
from ctm_transformer.lm_scale import ScaledCTMLM, _logits, chunked_cross_entropy, ctm_selected_tick_loss
from ctm_transformer.positions import rope

ADAPTATIONS = ('unit_query', 'observe_token', 'mean_tick_loss', 'query_feature', 'token_start', 'final_tick_loss', 'certainty_loss', 'state_readout',
               'sparse_tick_loss', 'token_history', 'decay_softplus', 'decay_spread', 'improvement_loss', 'cross_position', 'feature_head',
               'sync_mean', 'attention_sink', 'tick_memory', 'gated_state', 'decay_data_clamp')
MEMORY_SLOTS = 8  # tick_memory: long-term memory slots per position
LOSSES = ('mean_tick_loss', 'final_tick_loss', 'certainty_loss', 'sparse_tick_loss', 'improvement_loss')  # at most one; none means CTM's loss
SPARSE_TICKS = 4  # sparse_tick_loss: mean cross-entropy over every (T / 4)-th tick, ending at the final tick (ticks 4, 8, 12, 16 of 16)


class DataClampSynchronization(Synchronization):
    """Decay applied unclamped; the model clamps the parameter data before each forward pass (official CTM code)."""
    def step(self, alpha, beta, z):
        keep = torch.exp(-self.decay)
        return keep * alpha + self.products(z), keep * beta + 1


class SoftplusSynchronization(Synchronization):
    """CTM synchronization with decay r = softplus(rho): always >= 0, with a gradient everywhere."""
    def step(self, alpha, beta, z):
        keep = torch.exp(-F.softplus(self.decay))
        return keep * alpha + self.products(z), keep * beta + 1


def sparse_ticks(T):
    step = max(1, T // SPARSE_TICKS)
    return list(range(T - 1, -1, -step))[:SPARSE_TICKS][::-1]


def _token_cross_entropy(hidden, weight, targets):
    return F.cross_entropy(_logits(hidden, weight), targets, reduction='none')


def improvement_tick_loss(readouts, weight, targets, chunk=4096):
    """Tick-weighted cross-entropy plus a hinge on any tick worse than the one before. readouts: [B, S, T, P]."""
    T = readouts.shape[2]
    keep = targets.reshape(-1).ne(-100)
    r = readouts.reshape(-1, T, readouts.shape[-1])[keep]
    gold = targets.reshape(-1)[keep]
    rows = max(1, chunk // T)
    ce = torch.cat([checkpoint(_token_cross_entropy, r[i:i + rows].reshape(-1, r.shape[-1]), weight,
                               gold[i:i + rows, None].expand(-1, T).reshape(-1), use_reentrant=False).view(-1, T)
                    for i in range(0, len(gold), rows)])  # [n, T]
    weights = torch.arange(1, T + 1, device=ce.device, dtype=ce.dtype)
    weighted = (ce.mean(0) * weights).sum() / weights.sum()
    hinge = F.relu(ce[:, 1:] - ce[:, :-1].detach()).mean(0).sum() if T > 1 else ce.new_zeros(())
    return weighted + hinge


def certain_tick_loss(readouts, weight, targets, chunk=4096):
    """Cross-entropy at each token's most certain tick (selected without gradient). readouts: [B, S, T, P]."""
    T = readouts.shape[2]
    keep = targets.reshape(-1).ne(-100)
    r = readouts.reshape(-1, T, readouts.shape[-1])[keep]
    gold = targets.reshape(-1)[keep]
    certainty = torch.empty(len(gold), T, device=r.device)
    rows = max(1, chunk // T)
    with torch.no_grad():
        for i in range(0, len(gold), rows):
            logp = _logits(r[i:i + rows], weight).log_softmax(-1)
            certainty[i:i + rows] = (logp.exp() * logp).sum(-1)  # negative entropy: argmax is the most certain tick
            del logp
    chosen = r.gather(1, certainty.argmax(-1)[:, None, None].expand(-1, 1, r.shape[-1]))
    return chunked_cross_entropy(chosen, weight, gold[:, None], chunk)


class TickMemory(nn.Module):
    """CMM joint memory update: one gated pre-norm Transformer block over [sink; long-term slots; short-term history]."""
    def __init__(self, D, memory, slots, heads):
        super().__init__()
        if D % heads:
            raise ValueError('tick_memory needs d_latent divisible by n_heads')
        self.heads, self.slots = heads, slots
        self.sink = nn.Parameter(torch.zeros(D))
        self.slot_embedding = nn.Parameter(torch.randn(1 + slots + memory, D) * 0.02)  # added to queries and keys only
        self.norm1, self.norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qk = nn.Linear(D, 2 * D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.ffn = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, ltm, stm):  # [N, L, D], [N, M, D]
        N = stm.shape[0]
        x = torch.cat((self.sink.to(stm.dtype).expand(N, 1, -1), ltm.to(stm.dtype), stm), dim=1)
        h = self.norm1(x)
        q, k = self.qk(h + self.slot_embedding.to(h.dtype)).chunk(2, dim=-1)
        v = self.v(h)
        v = torch.cat((torch.zeros_like(v[:, :1]), v[:, 1:]), dim=1)  # the sink's value is zero
        split = lambda t: t.view(N, t.shape[1], self.heads, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(split(q), split(k), split(v)).transpose(1, 2).reshape(N, x.shape[1], -1)
        x = x + self.gate * self.o(a)
        x = x + self.gate * self.ffn(self.norm2(x))
        return x[:, 1:1 + self.slots], x[:, 1 + self.slots:]


class AdaptedCTMLM(ScaledCTMLM):
    def __init__(self, config, unit_query=False, observe_token=False, mean_tick_loss=False, query_feature=False, token_start=False,
                 final_tick_loss=False, certainty_loss=False, state_readout=False, sparse_tick_loss=False, token_history=False, decay_softplus=False,
                 decay_spread=False, improvement_loss=False, cross_position=False, feature_head=False, sync_mean=False, attention_sink=False,
                 tick_memory=False, gated_state=False, decay_data_clamp=False, checkpoint_ticks=False, compile_ticks=False, **kwargs):
        super().__init__(config, checkpoint_ticks=checkpoint_ticks, compile_ticks=False, **kwargs)
        self.adaptations = {'unit_query': unit_query, 'observe_token': observe_token, 'mean_tick_loss': mean_tick_loss, 'query_feature': query_feature,
                            'token_start': token_start, 'final_tick_loss': final_tick_loss, 'certainty_loss': certainty_loss, 'state_readout': state_readout,
                            'sparse_tick_loss': sparse_tick_loss, 'token_history': token_history, 'decay_softplus': decay_softplus,
                            'decay_spread': decay_spread, 'improvement_loss': improvement_loss, 'cross_position': cross_position,
                            'feature_head': feature_head, 'sync_mean': sync_mean, 'attention_sink': attention_sink, 'tick_memory': tick_memory,
                            'gated_state': gated_state, 'decay_data_clamp': decay_data_clamp}
        if decay_data_clamp and (decay_softplus or decay_spread):
            raise ValueError('Choose one decay change')
        if decay_softplus and decay_spread:
            raise ValueError('Choose one decay change')
        if sum(self.adaptations[k] for k in LOSSES) > 1:
            raise ValueError(f'At most one of {LOSSES}')
        d, D = config.d_model, config.d_latent
        pairs = self.lm_head.in_features
        self.query_feature = nn.Linear(d, d, bias=False) if query_feature else None
        self.start_projection = nn.Linear(d, D, bias=False) if token_start else None
        self.state_readout = nn.Linear(D, pairs, bias=False) if state_readout else None
        self.history_projection = nn.Linear(d, D, bias=False) if token_history else None
        self.tick_key = nn.Linear(D, d, bias=False) if cross_position else None
        self.tick_value = nn.Linear(D, d, bias=False) if cross_position else None
        for layer in (self.tick_key, self.tick_value):
            if layer is not None:
                nn.init.zeros_(layer.weight)
        self.feature_head = nn.Linear(d, config.vocab_size, bias=False) if feature_head else None
        self.sink_key = nn.Parameter(torch.zeros(config.n_heads, d // config.n_heads)) if attention_sink else None
        self.tick_memory = TickMemory(D, config.history_len, MEMORY_SLOTS, config.n_heads) if tick_memory else None
        self.ltm_init = nn.Parameter(torch.randn(MEMORY_SLOTS, D) * 0.1) if tick_memory else None
        self.state_gate = nn.Parameter(torch.zeros(D)) if gated_state else None
        if decay_data_clamp:
            for sync in (self.sync_action, self.sync_out):
                sync.__class__ = DataClampSynchronization
        with torch.no_grad():
            for sync in (self.sync_action, self.sync_out):
                if decay_softplus:
                    sync.__class__ = SoftplusSynchronization
                    sync.decay.fill_(math.log(math.expm1(0.01)))  # r = softplus(rho) = 0.01
                if decay_spread:
                    sync.decay.uniform_(0.0, 3.0)
        if observe_token:
            self.synapse = UNetSynapse(2 * d + D, self.synapse.out.in_features, D)
        if unit_query:
            with torch.no_grad():
                z = self.z_init[None].float()
                self.query.weight.div_(self.query(Synchronization.read(*self.sync_action.start(z))).pow(2).mean().sqrt())
        self._compiled_tick = torch.compile(self._checkpointed_tick) if compile_ticks else None

    def _read(self, alpha, beta):
        return alpha / beta if self.adaptations['sync_mean'] else Synchronization.read(alpha, beta)

    def _tick(self, z, history, alpha_a, beta_a, alpha_o, beta_o, keys, values, positions, observed=None, query_extra=None, ltm=None):
        """One tick. Returns the new state and the output readout; with tick_memory, the long-term memory slots last."""
        B, S = keys.shape[0], keys.shape[2]
        d, H = self.config.d_model, self.config.n_heads
        N = B * S
        if self.tick_key is not None:
            heads = lambda t: t.view(B, S, H, d // H).transpose(1, 2)
            keys = keys + rope(heads(self.tick_key(z)), positions).to(keys.dtype)  # RoPE is linear: rope(k) + rope(k') = rope(k + k')
            values = values + heads(self.tick_value(z)).to(values.dtype)
        q = self.query(self._read(alpha_a, beta_a))
        if query_extra is not None:
            q = q + query_extra.to(q.dtype)
        q = rope(q.view(B, S, H, d // H).transpose(1, 2), positions)
        if self.sink_key is not None:
            sink = self.sink_key[None, :, None, :].to(keys.dtype).expand(B, -1, 1, -1)
            keys, values = torch.cat((sink, keys), dim=2), torch.cat((torch.zeros_like(sink), values), dim=2)
            visible = torch.ones(S, S + 1, dtype=torch.bool, device=keys.device).tril(diagonal=1)  # the sink, then keys at positions <= own
            o = F.scaled_dot_product_attention(q, keys, values, attn_mask=visible)
        else:
            o = F.scaled_dot_product_attention(q, keys, values, is_causal=True)
        o = o.transpose(1, 2).reshape(N, d)
        parts = [self.attn_out(o)] + ([observed.to(o.dtype)] if observed is not None else []) + [z.to(o.dtype)]
        pre = self.synapse(torch.cat(parts, dim=-1))
        history = torch.cat((history[..., 1:], pre.unsqueeze(-1).to(history.dtype)), dim=-1)
        if self.tick_memory is not None:
            ltm, short = self.tick_memory(ltm, history.transpose(1, 2))
            z = self.nlm(short.transpose(1, 2).to(history.dtype)).float()
        else:
            z_new = self.nlm(history).float()
            z = z_new if self.state_gate is None else z + torch.sigmoid(self.state_gate).float() * (z_new - z)
        alpha_a, beta_a = self.sync_action.step(alpha_a, beta_a, z)
        alpha_o, beta_o = self.sync_out.step(alpha_o, beta_o, z)
        readout = self._read(alpha_o, beta_o)
        if self.state_readout is not None:
            readout = readout + self.state_readout(z.to(o.dtype)).float()
        if self.tick_memory is not None:
            return z, history, alpha_a, beta_a, alpha_o, beta_o, readout, ltm
        return z, history, alpha_a, beta_a, alpha_o, beta_o, readout

    def forward(self, input_ids, targets=None, max_thought_steps=None, return_all_logits=False):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError('Expected [batch, sequence] input within max_seq_len')
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError('Targets must be shifted and have the same shape as input_ids')
        T = self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        if type(T) is not int or T < 1:
            raise ValueError('max_thought_steps must be a positive integer')
        B, S = input_ids.shape
        d, D, H = self.config.d_model, self.config.d_latent, self.config.n_heads
        N = B * S
        if self.adaptations['decay_data_clamp']:
            with torch.no_grad():
                for sync in (self.sync_action, self.sync_out):
                    sync.decay.clamp_(0, 15)
        feats = self.features(input_ids)
        if self.feature_head is not None:
            if targets is not None and not return_all_logits:
                return {'logits': None, 'thought_steps': 0, 'loss': chunked_cross_entropy(feats, self.feature_head.weight, targets)}
            logits = self.feature_head(feats)
            return {'logits': logits, 'all_logits': [logits], 'thought_steps': 0}
        positions = torch.arange(S, device=input_ids.device)
        heads = lambda x: x.view(B, S, H, d // H).transpose(1, 2)
        keys = rope(heads(self.key(feats)), positions)
        values = heads(self.value(feats))
        observed = feats.reshape(N, d) if self.adaptations['observe_token'] else None
        query_extra = self.query_feature(feats).reshape(N, d) if self.query_feature is not None else None
        z = self.z_init.expand(N, D).float()
        if self.start_projection is not None:
            z = z + self.start_projection(feats).reshape(N, D).float()
        history = self.history_init.expand(N, D, self.config.history_len)
        if self.history_projection is not None:
            history = history + self.history_projection(feats).reshape(N, D, 1).to(history.dtype)
        alpha_a, beta_a = self.sync_action.start(z)
        alpha_o, beta_o = self.sync_out.start(z)
        training = targets is not None and not return_all_logits
        step = (self._compiled_tick or self._checkpointed_tick) if training else self._tick
        readouts = []
        ltm = self.ltm_init.expand(N, -1, -1) if self.tick_memory is not None else None
        for _ in range(T):
            out = step(z, history, alpha_a, beta_a, alpha_o, beta_o, keys, values, positions, observed, query_extra, ltm)
            z, history, alpha_a, beta_a, alpha_o, beta_o, readout = out[:7]
            if self.tick_memory is not None:
                ltm = out[7]
            readouts.append(readout.view(B, S, -1))
        if training and self.adaptations['mean_tick_loss']:
            stacked = torch.stack(readouts, dim=2)  # [B, S, T, P]
            loss = chunked_cross_entropy(stacked, self.lm_head.weight, targets[:, :, None].expand(-1, -1, T))
            return {'logits': None, 'thought_steps': T, 'loss': loss}
        if training and self.adaptations['sparse_tick_loss']:
            ticks = sparse_ticks(T)
            stacked = torch.stack([readouts[t] for t in ticks], dim=2)
            loss = chunked_cross_entropy(stacked, self.lm_head.weight, targets[:, :, None].expand(-1, -1, len(ticks)))
            return {'logits': None, 'thought_steps': T, 'loss': loss}
        if training and self.adaptations['improvement_loss']:
            return {'logits': None, 'thought_steps': T, 'loss': improvement_tick_loss(torch.stack(readouts, dim=2), self.lm_head.weight, targets)}
        if training and self.adaptations['final_tick_loss']:
            return {'logits': None, 'thought_steps': T, 'loss': chunked_cross_entropy(readouts[-1], self.lm_head.weight, targets)}
        if training and self.adaptations['certainty_loss']:
            return {'logits': None, 'thought_steps': T, 'loss': certain_tick_loss(torch.stack(readouts, dim=2), self.lm_head.weight, targets)}
        if training:
            return {'logits': None, 'thought_steps': T, **ctm_selected_tick_loss(torch.stack(readouts, dim=2), self.lm_head.weight, targets)}
        tick_logits = [self.lm_head(r) for r in readouts]
        return {'logits': tick_logits[-1], 'all_logits': tick_logits, 'thought_steps': T}

    def _checkpointed_tick(self, *state):
        if self.checkpoint_ticks and torch.is_grad_enabled():
            return checkpoint(self._tick, *state, use_reentrant=False)
        return self._tick(*state)


def adapted_factory(adaptations, checkpointing=False, compile=False, unet_width=None):
    """unet_width: hidden width of the synapse U-Net (default 2 x d_latent); the neuron-level models' width is config.nlm_hidden_dim."""
    unknown = set(adaptations) - set(ADAPTATIONS)
    if unknown:
        raise ValueError(f'Unknown adaptations {sorted(unknown)}')
    def factory(config):
        return AdaptedCTMLM(config, **{a: a in adaptations for a in ADAPTATIONS}, checkpoint_ticks=checkpointing, compile_ticks=compile,
                            unet_width=unet_width)
    factory.__qualname__ = f'adapted_factory[{"+".join(sorted(adaptations)) or "none"}' + (f',unet{unet_width}' if unet_width else '') + ']'
    return factory
