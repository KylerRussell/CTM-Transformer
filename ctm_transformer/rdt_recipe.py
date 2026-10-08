"""RDT recipe variants (research/RDT_RECIPE.md, 2026-10-08).

Our RDT copies Huginn's large-scale stability recipe: sandwich norm in every block,
a zero initial state, a concatenation adapter and (in our runs) unit-scale input
embeddings. Huginn reports that at small scale "most normalization strategies ...
work almost equally well". At depth 1, our 24-layer RDT trails a Transformer with the
same unique layers by 0.26 nats at 400M tokens and diverges at its learning rate.

``RecipeRDT`` is ``ScaledBaseline`` with three switches; the defaults reproduce it exactly:

* ``norm``: ``sandwich`` (every block, as Huginn), ``core_sandwich`` (pre-norm
  prelude and coda, sandwich only in the recurrent core) or ``prenorm`` (pre-norm
  everywhere).
* ``normalize``: an RMSNorm on the prelude output before every injection and an
  RMSNorm on the state at the core exit, which keep a pre-norm recurrence bounded
  (Parcae, arXiv 2604.12946, normalizes the injected input; Huginn-Linear in arXiv
  2610.06833 normalizes the core exit).
* ``state_init``: ``zeros``, or ``random`` (Huginn: truncated normal, variance 2/5,
  drawn on every forward pass).
"""
import torch
from dataclasses import replace
from torch import nn
from torch.nn import functional as F

from ctm_transformer.lm_scale import ScaledBaseline, chunked_cross_entropy

NORMS = ('sandwich', 'core_sandwich', 'prenorm')
STATE_INITS = ('zeros', 'random')


class RecipeRDT(ScaledBaseline):
    def __init__(self, config, norm='sandwich', normalize=False, state_init='zeros', backprop_steps=None, compile_blocks=False):
        if config.model_family != 'recurrent_depth':
            raise ValueError('RecipeRDT is for the recurrent-depth family')
        if norm not in NORMS or state_init not in STATE_INITS:
            raise ValueError(f'Unknown recipe {norm}/{state_init}')
        super().__init__(config, backprop_steps=backprop_steps, compile_blocks=False)
        self.recipe = {'norm': norm, 'normalize': normalize, 'state_init': state_init}
        unsandwiched = {'sandwich': (), 'core_sandwich': (self.prelude, self.coda), 'prenorm': (self.prelude, self.core, self.coda)}[norm]
        for blocks in unsandwiched:
            for block in blocks:
                block.attn_post, block.ffn_post = nn.Identity(), nn.Identity()
        self.inject_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps) if normalize else nn.Identity()
        self.exit_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps) if normalize else nn.Identity()
        if compile_blocks:
            for block in (*self.prelude, *self.core, *self.coda):
                block.forward = torch.compile(block.forward)

    def _state(self, embedded):
        if self.recipe['state_init'] == 'zeros':
            return torch.zeros_like(embedded)
        return nn.init.trunc_normal_(torch.empty_like(embedded), std=0.4 ** 0.5, a=-2 * 0.4 ** 0.5, b=2 * 0.4 ** 0.5)

    def _recur(self, x, depth, readouts=None):
        embedded = self._blocks(self.prelude, x)
        injected = self.inject_norm(embedded)
        x = self._state(embedded)
        for i in range(depth):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (self.backprop_steps is None or i >= depth - self.backprop_steps)):
                x = self.exit_norm(self._blocks(self.core, self.injection(torch.cat((x, injected), dim=-1))))
            if readouts is not None:
                readouts.append(self.lm_head(self.final_norm(self._blocks(self.coda, self.final_norm(x)))))
        return self._blocks(self.coda, self.final_norm(x))

    def forward(self, input_ids, targets=None, max_thought_steps=None, return_all_logits=False):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError('Expected [batch, sequence] input within max_seq_len')
        depth = self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        if type(depth) is not int or depth < 1:
            raise ValueError('max_thought_steps must be a positive integer')
        readouts = [] if return_all_logits else None
        x = self._recur(self.token_embedding(input_ids), depth, readouts)
        applications = len(self.prelude) + depth * len(self.core) + len(self.coda)
        if targets is not None and not return_all_logits:
            if targets.shape != input_ids.shape:
                raise ValueError('Targets must be shifted and have the same shape as input_ids')
            return {'logits': None, 'loss': chunked_cross_entropy(self.final_norm(x), self.lm_head.weight, targets),
                    'thought_steps': depth, 'block_applications': applications}
        logits = self.lm_head(self.final_norm(x))
        result = {'logits': logits, 'thought_steps': depth, 'block_applications': applications}
        if return_all_logits:
            result['all_logits'] = readouts
        if targets is not None:
            result['loss'] = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return result


def recipe_factory(recipe, checkpointing=False, backprop_steps=None, compile=False):
    """recipe: {'norm': ..., 'normalize': bool, 'state_init': ...}."""
    unknown = set(recipe) - {'norm', 'normalize', 'state_init'}
    if unknown:
        raise ValueError(f'Unknown recipe keys {sorted(unknown)}')
    def factory(config):
        return RecipeRDT(replace(config, gradient_checkpointing=checkpointing), backprop_steps=backprop_steps, compile_blocks=compile, **recipe)
    factory.__qualname__ = 'recipe_factory[' + ','.join(f'{k}={v}' for k, v in sorted(recipe.items())) + ']'
    return factory
