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
"""
import torch
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

from ctm_transformer.ctm_lm import Synchronization, UNetSynapse
from ctm_transformer.lm_scale import ScaledCTMLM, chunked_cross_entropy, ctm_selected_tick_loss
from ctm_transformer.positions import rope

ADAPTATIONS = ('unit_query', 'observe_token', 'mean_tick_loss')


class AdaptedCTMLM(ScaledCTMLM):
    def __init__(self, config, unit_query=False, observe_token=False, mean_tick_loss=False, checkpoint_ticks=False, compile_ticks=False, **kwargs):
        super().__init__(config, checkpoint_ticks=checkpoint_ticks, compile_ticks=False, **kwargs)
        self.adaptations = {'unit_query': unit_query, 'observe_token': observe_token, 'mean_tick_loss': mean_tick_loss}
        d, D = config.d_model, config.d_latent
        if observe_token:
            self.synapse = UNetSynapse(2 * d + D, self.synapse.out.in_features, D)
        if unit_query:
            with torch.no_grad():
                z = self.z_init[None].float()
                self.query.weight.div_(self.query(Synchronization.read(*self.sync_action.start(z))).pow(2).mean().sqrt())
        self._compiled_tick = torch.compile(self._checkpointed_tick) if compile_ticks else None

    def _tick(self, z, history, alpha_a, beta_a, alpha_o, beta_o, keys, values, positions, observed=None):
        B, S = keys.shape[0], keys.shape[2]
        d, H = self.config.d_model, self.config.n_heads
        N = B * S
        q = rope(self.query(Synchronization.read(alpha_a, beta_a)).view(B, S, H, d // H).transpose(1, 2), positions)
        o = F.scaled_dot_product_attention(q, keys, values, is_causal=True).transpose(1, 2).reshape(N, d)
        parts = [self.attn_out(o)] + ([observed.to(o.dtype)] if observed is not None else []) + [z.to(o.dtype)]
        pre = self.synapse(torch.cat(parts, dim=-1))
        history = torch.cat((history[..., 1:], pre.unsqueeze(-1).to(history.dtype)), dim=-1)
        z = self.nlm(history).float()
        alpha_a, beta_a = self.sync_action.step(alpha_a, beta_a, z)
        alpha_o, beta_o = self.sync_out.step(alpha_o, beta_o, z)
        return z, history, alpha_a, beta_a, alpha_o, beta_o, Synchronization.read(alpha_o, beta_o)

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
        feats = self.features(input_ids)
        positions = torch.arange(S, device=input_ids.device)
        heads = lambda x: x.view(B, S, H, d // H).transpose(1, 2)
        keys = rope(heads(self.key(feats)), positions)
        values = heads(self.value(feats))
        observed = feats.reshape(N, d) if self.adaptations['observe_token'] else None
        z = self.z_init.expand(N, D).float()
        history = self.history_init.expand(N, D, self.config.history_len)
        alpha_a, beta_a = self.sync_action.start(z)
        alpha_o, beta_o = self.sync_out.start(z)
        training = targets is not None and not return_all_logits
        step = (self._compiled_tick or self._checkpointed_tick) if training else self._tick
        readouts = []
        for _ in range(T):
            z, history, alpha_a, beta_a, alpha_o, beta_o, readout = step(z, history, alpha_a, beta_a, alpha_o, beta_o, keys, values, positions, observed)
            readouts.append(readout.view(B, S, -1))
        if training and self.adaptations['mean_tick_loss']:
            stacked = torch.stack(readouts, dim=2)  # [B, S, T, P]
            loss = chunked_cross_entropy(stacked, self.lm_head.weight, targets[:, :, None].expand(-1, -1, T))
            return {'logits': None, 'thought_steps': T, 'loss': loss}
        if training:
            return {'logits': None, 'thought_steps': T, **ctm_selected_tick_loss(torch.stack(readouts, dim=2), self.lm_head.weight, targets)}
        tick_logits = [self.lm_head(r) for r in readouts]
        return {'logits': tick_logits[-1], 'all_logits': tick_logits, 'thought_steps': T}

    def _checkpointed_tick(self, *state):
        if self.checkpoint_ticks and torch.is_grad_enabled():
            return checkpoint(self._tick, *state, use_reentrant=False)
        return self._tick(*state)


def adapted_factory(adaptations, checkpointing=False, compile=False):
    unknown = set(adaptations) - set(ADAPTATIONS)
    if unknown:
        raise ValueError(f'Unknown adaptations {sorted(unknown)}')
    def factory(config):
        return AdaptedCTMLM(config, **{a: a in adaptations for a in ADAPTATIONS}, checkpoint_ticks=checkpointing, compile_ticks=compile)
    factory.__qualname__ = f'adapted_factory[{"+".join(sorted(adaptations)) or "none"}]'
    return factory
