"""Named CTM architecture variants built without changing `CTMConfig`.

`CTMConfig` must load older, fully specified configs unchanged, so variant
choices live here and are recorded by name alongside each run:

* ``reference``      the canonical CTM: cross-attention keys/values are the static
                     token-plus-position embeddings, computed once per forward pass.
* ``attn_res``       the existing Kimi-style attention residuals
                     (``use_attention_residuals=True``). They mix each position's
                     earlier *layer outputs*; they do not exchange information across
                     positions, so keys/values stay static.
* ``contextual_kv``  one causal self-attention block over the embedded sequence
                     before CTM reads it, so keys/values (and the output head's text
                     features) are contextual. This lets CTM form induction heads,
                     as the Transformer and the recurrent-depth prelude can. Thought
                     ticks, synchronization, temporal MLPs and the readout are unchanged.
"""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from ctm_transformer.model import CTMTransformer

VARIANTS=('reference','attn_res','contextual_kv')


class ContextualPrelude(nn.Module):
    """Pre-norm causal self-attention and SwiGLU block over token embeddings."""
    def __init__(self,d_model,n_heads,ffn_hidden,eps=1e-5):
        super().__init__()
        self.heads=n_heads;self.attn_norm=nn.RMSNorm(d_model,eps=eps);self.ffn_norm=nn.RMSNorm(d_model,eps=eps)
        self.qkv=nn.Linear(d_model,3*d_model,bias=False);self.out=nn.Linear(d_model,d_model,bias=False)
        self.gate=nn.Linear(d_model,ffn_hidden,bias=False);self.up=nn.Linear(d_model,ffn_hidden,bias=False)
        self.down=nn.Linear(ffn_hidden,d_model,bias=False)
        for module in (self.qkv,self.out,self.gate,self.up,self.down):nn.init.normal_(module.weight,std=0.02)

    def forward(self,x):
        batch,length,width=x.shape
        q,k,v=self.qkv(self.attn_norm(x)).view(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(batch,length,width)
        x=x+self.out(y);h=self.ffn_norm(x)
        return x+self.down(F.silu(self.gate(h))*self.up(h))


class ContextualKVCTM(CTMTransformer):
    """CTM whose cross-attention reads a causal, contextual encoding of the text."""
    def __init__(self,config):
        super().__init__(config)
        self.contextual_prelude=ContextualPrelude(config.d_model,config.n_heads,2*config.d_model)

    def _embed_text(self,input_ids):
        return self.contextual_prelude(super()._embed_text(input_ids))


def variant_config(config,variant):
    if variant not in VARIANTS:raise ValueError(f'Unknown CTM variant {variant}')
    if variant=='attn_res':return replace(config,use_attention_residuals=True)
    if variant=='reference' and config.use_attention_residuals:raise ValueError('The reference variant disables attention residuals')
    return config


def build_variant(config,variant):
    # Callers apply variant_config first, so the recorded run config is the one actually built.
    if variant_config(config,variant)!=config:raise ValueError('Apply variant_config before building; the config does not match the variant')
    return ContextualKVCTM(config) if variant=='contextual_kv' else CTMTransformer(config)


def variant_factory(variant):
    """Model factory for runner v3; the name is recorded in each run manifest."""
    def factory(config):return build_variant(config,variant)
    factory.__qualname__=f'variant_factory[{variant}]'
    return factory
