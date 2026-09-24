"""Recurrent-depth temporal supervision with the existing CTM loss semantics.

The base architecture/config/state dictionary are unchanged. Inference uses the
base forward. Auxiliary training decodes every recurrent state with the shared
coda/head; decoded states never feed back into the recurrence.
"""
import math
import torch
from torch.nn import functional as F
from ctm_transformer.baselines import BaselineTransformer

OBJECTIVES = ('final_ce', 'uniform', 'dynamic_aggregate')


def validate_recurrent_objective(config, objective):
    if objective not in OBJECTIVES:
        raise ValueError(f'Unknown recurrent temporal objective: {objective}')
    if config.model_family != 'recurrent_depth':
        raise ValueError('Recurrent temporal supervision requires recurrent_depth')
    if config.dropout != 0:
        raise ValueError('The matched temporal control requires dropout=0')


def temporal_loss(all_logits, targets, objective):
    """Match CTM masking, earliest-tick ties, and native-dtype certainty.

    Training certainty intentionally uses CTM's softmax/log arithmetic in the
    logits dtype (BF16 under autocast). Deployment confidence remains the
    separate existing FP32-entropy readout. Gold selects one training-loss
    branch only; it never enters deployment selection.
    """
    if objective not in OBJECTIVES or not all_logits:
        raise ValueError('Expected a declared objective and nonempty tick logits')
    if any(z.shape[:-1] != targets.shape for z in all_logits):
        raise ValueError('Targets and all tick logits must have matching token shapes')
    valid = targets.reshape(-1) != -100
    if not valid.any():
        raise ValueError('Temporal supervision requires at least one supervised token')
    vocab = all_logits[0].shape[-1]
    if objective == 'dynamic_aggregate':
        ce = torch.stack([F.cross_entropy(z.reshape(-1, vocab), targets.reshape(-1), reduction='none')
                          for z in all_logits], dim=1)
        with torch.no_grad():
            certainty = []
            for z in all_logits:
                probs = F.softmax(z, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                certainty.append(1.0 - entropy / math.log(vocab))
            certainty = torch.stack(certainty)
            chosen = certainty.reshape(len(all_logits), -1).T.argmax(dim=-1)
            lowest = ce.argmin(dim=-1)
        loss = ((ce.gather(1, lowest[:, None]).squeeze(1) +
                 ce.gather(1, chosen[:, None]).squeeze(1)) / 2)[valid].mean()
        return {'loss': loss, 'per_tick_loss': ce[valid].mean(0), 'certainties': certainty}
    per_tick = torch.stack([F.cross_entropy(z.reshape(-1, vocab), targets.reshape(-1)) for z in all_logits])
    return {'loss': per_tick[-1] if objective == 'final_ce' else per_tick.mean(), 'per_tick_loss': per_tick}


def recurrent_objective_metadata(config, objective):
    validate_recurrent_objective(config, objective)
    depth = config.max_thought_steps
    weights = ([0.] * (depth - 1) + [1.] if objective == 'final_ce' else
               [1. / depth] * depth if objective == 'uniform' else None)
    return {'type': objective, 'direct_ce_tick_weights_at_max_depth': weights,
            'mono_penalty_weight': 0., 'validation_metric': 'supervised-token CE under the declared checkpoint-selection readout',
            'dynamic_rule': 'mean over supervised tokens of (minimum-tick CE + maximum-certainty-tick CE)/2' if objective == 'dynamic_aggregate' else None,
            'training_certainty': 'CTM native-logits-dtype normalized entropy; earliest tie' if objective == 'dynamic_aggregate' else None,
            'auxiliary_decoder': 'shared coda/head at every tick; no feedback into recurrent state' if objective != 'final_ce' else None}


class RecurrentTemporalTransformer(BaselineTransformer):
    def __init__(self, config, objective):
        validate_recurrent_objective(config, objective)
        super().__init__(config)
        self.temporal_objective = objective

    def forward(self, input_ids, targets=None, max_thought_steps=None, return_all_logits=False):
        if targets is None or self.temporal_objective == 'final_ce':
            return super().forward(input_ids, targets, max_thought_steps, return_all_logits)
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError('Expected [batch, sequence] input within max_seq_len')
        if targets.shape != input_ids.shape:
            raise ValueError('Targets must be shifted and have the same shape as input_ids')
        depth = self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        if type(depth) is not int or depth < 1:
            raise ValueError('max_thought_steps must be a positive integer')
        embedded = self.token_embedding(input_ids)
        if self.pos_embedding is not None:
            embedded = embedded + self.pos_embedding(torch.arange(input_ids.shape[1], device=input_ids.device))
        embedded = self._blocks(self.prelude, embedded)
        state = torch.zeros_like(embedded)
        ticks = []
        for _ in range(depth):
            state = self.injection(torch.cat((state, embedded), dim=-1))
            state = self._blocks(self.core, state)
            decoded = self._blocks(self.coda, self.final_norm(state))
            ticks.append(self.lm_head(self.final_norm(decoded)))
        return {'logits': ticks[-1], 'all_logits': ticks, 'thought_steps': depth,
                'block_applications': len(self.prelude) + depth * (len(self.core) + len(self.coda)),
                **temporal_loss(ticks, targets, self.temporal_objective)}
