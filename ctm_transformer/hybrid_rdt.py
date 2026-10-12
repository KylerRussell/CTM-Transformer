"""RDT + CTM hybrids on the adopted RDT recipe (research/RDT_RECIPE.md, 2026-10-11).

The base is ``Dans``: ``RecipeRDT`` with pre-norm blocks, additive injection, no coda norm
and an RMSNorm on the recurrent state. It beats its 8-layer Transformer twin on two seeds
at 400M tokens. Each CTM mechanism below is added so that the model equals Dans at
initialization (zero-initialized output or scale), keeping the RDT's residual path intact:

* ``sync_readout``: the CTM's synchronization. Running decayed pairwise products of the
  normalized state's neurons over iterations (alpha, beta as in ``ctm_lm.Synchronization``,
  learned decay per pair) are read as alpha / sqrt(beta) and added to the state before the
  coda through a zero-initialized projection. This gives the output an integral over
  iterations, which the CTM reads in place of the last state.
* ``neuron_models``: the CTM's neuron-level models. Each neuron's last ``history``
  normalized core outputs go through a private per-neuron MLP (``ctm_lm.NeuronLevelModels``),
  whose output is added to the core output with a per-neuron scale initialized to zero.
* ``state_carry``: a learned per-neuron carry of the previous state, s' = core(...) + c * s
  with c initialized to zero. It is the residual analogue of the gated state update that
  helped the CTM on S3 (cross_anchor).

The CTM's other drift fix there, re-injecting the backbone feature every tick, is already
part of the RDT, which adds the prelude output at every iteration.
"""
import torch
from dataclasses import replace
from torch import nn
from torch.utils.checkpoint import checkpoint

from ctm_transformer.ctm_lm import NeuronLevelModels, Synchronization
from ctm_transformer.rdt_recipe import RECIPE_KEYS, RecipeRDT

DANS = {'norm': 'prenorm', 'normalize': False, 'state_init': 'zeros', 'injection': 'add', 'coda_norm': 'none', 'state_norm': True}
MECHANISMS = ('sync_readout', 'neuron_models', 'state_carry')
HISTORY = 4      # neuron_models: iterations of history per neuron
NLM_HIDDEN = 16  # neuron_models: hidden width of each neuron's MLP


class HybridRDT(RecipeRDT):
    def __init__(self, config, sync_readout=False, neuron_models=False, state_carry=False, pairs=None, seed=0, **recipe):
        super().__init__(config, **recipe)
        if not recipe.get('state_norm'):
            raise ValueError('HybridRDT reads the normalized state; use a recipe with state_norm')
        d = config.d_model
        self.mechanisms = {'sync_readout': sync_readout, 'neuron_models': neuron_models, 'state_carry': state_carry}
        if sync_readout:
            pairs = pairs or d
            self.sync = Synchronization(d, pairs, torch.Generator().manual_seed(seed))
            self.sync_out = nn.Linear(pairs, d, bias=False)
            nn.init.zeros_(self.sync_out.weight)
        if neuron_models:
            self.nlm = NeuronLevelModels(d, HISTORY, NLM_HIDDEN)
            self.nlm_scale = nn.Parameter(torch.zeros(d))
        if state_carry:
            self.carry = nn.Parameter(torch.zeros(d))

    def _neuron_models(self, *history):
        B, S, d = history[0].shape
        return self.nlm(torch.stack(history, dim=-1).reshape(B * S, d, HISTORY)).view(B, S, d)

    def _iterate(self, x, injected, history):
        new = self._blocks(self.core, self.injection(torch.cat((self.state_norm(x), injected), dim=-1)))
        if self.mechanisms['neuron_models']:
            # The history is a tuple of the last normalized core outputs, so each iteration stores one new tensor;
            # the per-neuron MLPs are recomputed in the backward pass when checkpointing.
            normed = self.state_norm(new)
            history = (torch.zeros_like(normed),) * (HISTORY - 1) + (normed,) if history is None else history[1:] + (normed,)
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                out = checkpoint(self._neuron_models, *history, use_reentrant=False)
            else:
                out = self._neuron_models(*history)
            new = new + self.nlm_scale.to(new.dtype) * out.to(new.dtype)
        if self.mechanisms['state_carry']:
            new = new + self.carry.to(new.dtype) * x
        return new, history

    def _readout(self, x, sync):
        if sync is None:
            return x
        return x + self.sync_out(Synchronization.read(*sync).to(x.dtype))

    def _recur(self, x, depth, readouts=None):
        embedded = self._blocks(self.prelude, x)
        injected = self.inject_norm(embedded)
        x = self._state(embedded)
        history = None
        sync = None
        for i in range(depth):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (self.backprop_steps is None or i >= depth - self.backprop_steps)):
                x, history = self._iterate(x, injected, history)
                if self.mechanisms['sync_readout']:
                    z = self.state_norm(x).float().flatten(0, 1)
                    sync = self.sync.start(z) if sync is None else self.sync.step(*sync, z)
            if readouts is not None:
                readouts.append(self.lm_head(self.final_norm(self._blocks(self.coda, self._coda_in(self._readout(x, self._shaped(sync, x)))))))
        return self._blocks(self.coda, self._coda_in(self._readout(x, self._shaped(sync, x))))

    @staticmethod
    def _shaped(sync, x):
        if sync is None:
            return None
        return tuple(t.view(*x.shape[:2], -1) for t in sync)


def hybrid_factory(mechanisms, recipe=DANS, checkpointing=False, backprop_steps=None, compile=False):
    unknown = set(mechanisms) - set(MECHANISMS)
    if unknown or set(recipe) - RECIPE_KEYS:
        raise ValueError(f'Unknown mechanisms {sorted(unknown)} or recipe keys')
    def factory(config):
        return HybridRDT(replace(config, gradient_checkpointing=checkpointing), backprop_steps=backprop_steps, compile_blocks=compile,
                         **{m: m in mechanisms for m in MECHANISMS}, **recipe)
    factory.__qualname__ = 'hybrid_factory[' + ('+'.join(sorted(mechanisms)) or 'dans') + ']'
    return factory
