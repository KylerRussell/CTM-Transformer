"""Inference-only, per-token maximum-certainty readout for CTM diagnostics."""
import torch
from torch import nn


class ConfidenceReadout(nn.Module):
    def __init__(self, model):
        super().__init__();self.model=model

    def forward(self, input_ids, **kwargs):
        if 'targets' in kwargs:raise ValueError('Confidence readout is inference-only')
        if getattr(self.model.config, 'model_family', 'ctm') != 'ctm':
            kwargs['return_all_logits'] = True
        result=self.model(input_ids,**kwargs)
        logits=torch.stack(result['all_logits'],dim=2)  # B,S,T,V
        # Same FP32 entropy rule for every objective; avoid BF16 normalization ties.
        logp=logits.float().log_softmax(-1)
        certainty=(logp.exp()*logp).sum(-1)
        chosen=certainty.argmax(dim=2)  # B,S; no gold targets
        selected=logits.gather(2,chosen[...,None,None].expand(-1,-1,1,logits.shape[-1])).squeeze(2)
        return {'logits':selected,'selected_ticks':chosen+1}
