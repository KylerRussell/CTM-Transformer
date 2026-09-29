"""Relative position encoding (RoPE) for the Transformer, RDT and CTM-LM families.

The frozen models add a learned absolute position table to the token embeddings.
On S3, training words have lengths 1..16, so table rows 17 and later never
receive a gradient and length extrapolation cannot be measured
(research/results/randdepth_s3_v1/INTERPRETATION.md). These factories build the
same models without the table and with rotary position embedding on the
queries and keys of every self-attention layer (prelude, core and coda
included, and on every recurrence). Parameters and their initialization are
unchanged apart from the removed table. With all positions set to 0, RoPE is the
identity, so each model reduces exactly to its no-position-encoding (NoPE) form,
which the tests check.

Odd head widths (the Transformer's 35) rotate the largest even prefix of each
head and pass the last channel through (partial rotary embedding).
"""
import torch
from torch.nn import functional as F

from ctm_transformer.baselines import BaselineTransformer,CausalAttention
from ctm_transformer.ctm_lm import CTMLM
from ctm_transformer.ctm_variants import ContextualPrelude


def rope(x,positions,base=10000.0):
    """Rotary embedding over the largest even prefix of the last dimension; x: [B, H, S, dh], positions: [S]."""
    rotated=x.shape[-1]//2*2;half=rotated//2
    freqs=base**(-torch.arange(half,device=x.device,dtype=torch.float32)/half)
    angles=positions.float()[:,None]*freqs[None]
    cos,sin=angles.cos().to(x.dtype),angles.sin().to(x.dtype)
    x1,x2,rest=x[...,:half],x[...,half:rotated],x[...,rotated:]
    return torch.cat((x1*cos-x2*sin,x1*sin+x2*cos,rest),dim=-1)


class RoPECausalAttention(CausalAttention):
    """The baseline's causal attention with RoPE on queries and keys (same parameters)."""
    positions=None  # set per forward by the owning model; None means 0..S-1

    def forward(self,x):
        batch,length,width=x.shape
        positions=self.positions if self.positions is not None else torch.arange(length,device=x.device)
        def split(projection):
            return projection(x).view(batch,length,self.heads,width//self.heads).transpose(1,2)
        y=F.scaled_dot_product_attention(rope(split(self.q),positions),rope(split(self.k),positions),split(self.v),
            is_causal=True,dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1,2).contiguous().view(batch,length,width))


class RoPEContextualPrelude(ContextualPrelude):
    """CTM-LM backbone block with RoPE on queries and keys (same parameters)."""
    positions=None

    def forward(self,x):
        batch,length,width=x.shape
        positions=self.positions if self.positions is not None else torch.arange(length,device=x.device)
        q,k,v=self.qkv(self.attn_norm(x)).view(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        y=F.scaled_dot_product_attention(rope(q,positions),rope(k,positions),v,is_causal=True).transpose(1,2).reshape(batch,length,width)
        x=x+self.out(y);h=self.ffn_norm(x)
        return x+self.down(F.silu(self.gate(h))*self.up(h))


def _require_no_table(config):
    if config.use_positional_encoding:raise ValueError('RoPE models replace the absolute table; set use_positional_encoding=False')


class RoPEBaseline(BaselineTransformer):
    """Transformer or recurrent-depth baseline with RoPE in every attention layer and no absolute table."""
    def __init__(self,config):
        _require_no_table(config)
        super().__init__(config)
        for block in (*self.layers,*self.prelude,*self.core,*self.coda):block.attn.__class__=RoPECausalAttention


class RoPECTMLM(CTMLM):
    """CTM-LM with RoPE in its causal backbone as well as its tick attention, and no absolute table."""
    def __init__(self,config,**kwargs):
        _require_no_table(config)
        super().__init__(config,**kwargs)
        for block in self.backbone:block.__class__=RoPEContextualPrelude


def position_factory(model,scheme):
    """Model factory for the runner. model: transformer | rdt | ctm_lm; scheme: rope | nope.

    NoPE uses the unchanged frozen model classes with the table disabled by config;
    the factory checks the config agrees with the declared scheme.
    """
    if model not in ('transformer','rdt','ctm_lm') or scheme not in ('rope','nope'):raise ValueError(f'Unknown position cell {model}/{scheme}')
    def factory(config):
        _require_no_table(config)
        if model=='ctm_lm':return RoPECTMLM(config) if scheme=='rope' else CTMLM(config)
        return RoPEBaseline(config) if scheme=='rope' else BaselineTransformer(config)
    factory.__qualname__=f'position_factory[{model},{scheme}]'
    return factory
