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
* ``contextual_kv_token_init``  ``contextual_kv``, and each position's latent starts from
                     z0 + W·e_i instead of the shared z0, so its first queries can depend
                     on its own token.
* ``contextual_kv_injection``   ``contextual_kv``, and W·e_i is added to the latent before
                     every thought tick (input injection, as in the recurrent-depth model).

In the reference CTM every position starts from the same learned z0 and forms
queries from its own synchronization, so no path gives a position's latent its
own token before readout. The two injection variants test that limitation.
"""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from ctm_transformer.model import CTMTransformer

VARIANTS=('reference','attn_res','contextual_kv','contextual_kv_token_init','contextual_kv_injection')


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


class InjectedContextualKVCTM(ContextualKVCTM):
    """Contextual K/V plus a projection of each position's text features into its latent.

    ``every_tick=False`` injects before tick 0 only (token-conditioned initial state);
    ``every_tick=True`` injects before every tick. The projection starts small, so
    initial behavior stays close to ``contextual_kv``.
    """
    def __init__(self,config,every_tick):
        super().__init__(config)
        self.every_tick=every_tick
        self.input_injection=nn.Linear(config.d_model,config.d_latent,bias=False)
        nn.init.normal_(self.input_injection.weight,std=0.02)

    def _thought_step(self,z,t,text_emb,*args,**kwargs):
        if self.every_tick or t==0:z=z+self.input_injection(text_emb).to(z.dtype)
        return super()._thought_step(z,t,text_emb,*args,**kwargs)


def variant_config(config,variant):
    if variant not in VARIANTS:raise ValueError(f'Unknown CTM variant {variant}')
    if variant=='attn_res':return replace(config,use_attention_residuals=True)
    if variant=='reference' and config.use_attention_residuals:raise ValueError('The reference variant disables attention residuals')
    return config


def build_variant(config,variant):
    # Callers apply variant_config first, so the recorded run config is the one actually built.
    if variant_config(config,variant)!=config:raise ValueError('Apply variant_config before building; the config does not match the variant')
    if variant=='contextual_kv':return ContextualKVCTM(config)
    if variant.startswith('contextual_kv_'):return InjectedContextualKVCTM(config,every_tick=variant.endswith('injection'))
    return CTMTransformer(config)


def variant_factory(variant):
    """Model factory for runner v3; the name is recorded in each run manifest."""
    def factory(config):return build_variant(config,variant)
    factory.__qualname__=f'variant_factory[{variant}]'
    return factory
