"""CTM-LM: a Continuous Thought Machine as close to the published design as LM scale allows.

See research/CTM_LM_DESIGN.md. Per position, over T internal ticks:

    S_action = α_a / √β_a                      recursive decayed synchronization (action pairs)
    q        = RoPE(W_in · S_action)           queries; keys/values come from a causal backbone
    o        = CausalCrossAttention(q, K, V)   position p attends backbone features at positions ≤ p
    a        = Synapse([o ; z])                U-Net-style MLP with skip connections
    A        = FIFO(A, a)                      pre-activation history (memory length M)
    z        = NLM(A)                          private per-neuron MLPs (GLU)
    α, β     ← e^{−r}·α + z_i·z_j,  e^{−r}·β + 1   for both pair sets, r ≥ 0 learned per pair
    logits_t = W_out · (α_o / √β_o)            output synchronization readout

The loss is CTM's: per supervised token, the mean of the minimum-loss tick's and
the maximum-certainty tick's cross-entropy (certainty = 1 − normalized entropy).
The initial post-activation state and pre-activation history are learned.
Positions think independently (no latent-to-latent attention), as in the
published CTM, so training parallelizes over positions and generation needs
only a cache of backbone keys and values. Synchronization accumulators are
kept in FP32.
"""
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from ctm_transformer.ctm_variants import ContextualPrelude

DECAY_MAX=15.0


def rope(x,positions,base=10000.0):
    """Rotary position embedding over the last dimension; x: [B, H, S, dh], positions: [S]."""
    half=x.shape[-1]//2
    freqs=base**(-torch.arange(half,device=x.device,dtype=torch.float32)/half)
    angles=positions.float()[:,None]*freqs[None]
    cos,sin=angles.cos().to(x.dtype),angles.sin().to(x.dtype)
    x1,x2=x[...,:half],x[...,half:]
    return torch.cat((x1*cos-x2*sin,x1*sin+x2*cos),dim=-1)


class PerNeuronLinear(nn.Module):
    """Independent linear maps per neuron: [N, D, in] -> [N, D, out]."""
    def __init__(self,neurons,fan_in,fan_out):
        super().__init__()
        bound=1/math.sqrt(fan_in)
        self.weight=nn.Parameter(torch.empty(neurons,fan_in,fan_out).uniform_(-bound,bound))
        self.bias=nn.Parameter(torch.zeros(neurons,fan_out))

    def forward(self,x):return torch.einsum('ndi,dio->ndo',x,self.weight)+self.bias


class NeuronLevelModels(nn.Module):
    """Private per-neuron MLPs over the pre-activation history: M -> 2h -GLU-> h -> 2 -GLU-> 1."""
    def __init__(self,neurons,memory,hidden):
        super().__init__()
        self.first=PerNeuronLinear(neurons,memory,2*hidden);self.second=PerNeuronLinear(neurons,hidden,2)

    def forward(self,history):  # history: [N, D, M]
        return F.glu(self.second(F.glu(self.first(history),dim=-1)),dim=-1).squeeze(-1)


class UNetSynapse(nn.Module):
    """U-Net-style MLP: in -> w -> w/2 (bottleneck) -> w (+skip) -> out, with LayerNorm and SiLU."""
    def __init__(self,fan_in,width,fan_out):
        super().__init__()
        self.down1=nn.Linear(fan_in,width);self.norm1=nn.LayerNorm(width)
        self.down2=nn.Linear(width,width//2);self.norm2=nn.LayerNorm(width//2)
        self.up=nn.Linear(width//2,width);self.norm3=nn.LayerNorm(width)
        self.out=nn.Linear(width,fan_out)

    def forward(self,x):
        h1=F.silu(self.norm1(self.down1(x)));h2=F.silu(self.norm2(self.down2(h1)))
        return self.out(F.silu(self.norm3(self.up(h2)))+h1)


class Synchronization(nn.Module):
    """Recursive decayed pairwise products over randomly sampled neuron pairs."""
    def __init__(self,neurons,pairs,generator):
        super().__init__()
        self.register_buffer('left',torch.randint(0,neurons,(pairs,),generator=generator))
        self.register_buffer('right',torch.randint(0,neurons,(pairs,),generator=generator))
        self.decay=nn.Parameter(torch.zeros(pairs))  # r, clamped to [0, DECAY_MAX] as in CTM

    def products(self,z):return z[:,self.left]*z[:,self.right]

    def start(self,z):
        return self.products(z),torch.ones(z.shape[0],self.left.shape[0],device=z.device,dtype=z.dtype)

    def step(self,alpha,beta,z):
        keep=torch.exp(-self.decay.clamp(0,DECAY_MAX))
        return keep*alpha+self.products(z),keep*beta+1

    @staticmethod
    def read(alpha,beta):return alpha/torch.sqrt(beta)


class CTMLM(nn.Module):
    def __init__(self,config,unet_width=None,seed=0):
        super().__init__()
        self.config=config;d,D,H=config.d_model,config.d_latent,config.n_heads
        if d%H or (d//H)%2:raise ValueError('Head width must be even and divide d_model')
        pairs=config.sync_sparse_pairs
        self.token_embedding=nn.Embedding(config.vocab_size,d)
        self.pos_embedding=nn.Embedding(config.max_seq_len,d) if config.use_positional_encoding else None
        for e in (self.token_embedding,self.pos_embedding):
            if e is not None:nn.init.normal_(e.weight,std=0.02)
        self.backbone=nn.ModuleList(ContextualPrelude(d,H,2*d) for _ in range(config.n_layers))
        self.backbone_norm=nn.RMSNorm(d)
        self.key=nn.Linear(d,d,bias=False);self.value=nn.Linear(d,d,bias=False)
        self.query=nn.Linear(pairs,d,bias=False);self.attn_out=nn.Linear(d,d,bias=False)
        self.synapse=UNetSynapse(d+D,unet_width or 2*D,D)
        self.nlm=NeuronLevelModels(D,config.history_len,config.nlm_hidden_dim)
        self.z_init=nn.Parameter(torch.randn(D)*0.1)
        self.history_init=nn.Parameter(torch.randn(D,config.history_len)*0.1)
        generator=torch.Generator().manual_seed(seed)  # Pair sampling is fixed and recorded as buffers.
        self.sync_action=Synchronization(D,pairs,generator);self.sync_out=Synchronization(D,pairs,generator)
        self.lm_head=nn.Linear(pairs,config.vocab_size,bias=False)

    def features(self,input_ids):
        x=self.token_embedding(input_ids)
        if self.pos_embedding is not None:x=x+self.pos_embedding(torch.arange(input_ids.shape[1],device=input_ids.device))
        for block in self.backbone:x=block(x)
        return self.backbone_norm(x)

    def forward(self,input_ids,targets=None,max_thought_steps=None,return_all_logits=False):
        if input_ids.ndim!=2 or not 0<input_ids.shape[1]<=self.config.max_seq_len:
            raise ValueError('Expected [batch, sequence] input within max_seq_len')
        T=self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        if type(T) is not int or T<1:raise ValueError('max_thought_steps must be a positive integer')
        B,S=input_ids.shape;d,D,H=self.config.d_model,self.config.d_latent,self.config.n_heads;dh=d//H;N=B*S
        feats=self.features(input_ids);positions=torch.arange(S,device=input_ids.device)
        heads=lambda x:x.view(B,S,H,dh).transpose(1,2)
        keys=rope(heads(self.key(feats)),positions);values=heads(self.value(feats))
        z=self.z_init.expand(N,D).float();history=self.history_init.expand(N,D,self.config.history_len)
        alpha_a,beta_a=self.sync_action.start(z);alpha_o,beta_o=self.sync_out.start(z)
        tick_logits=[]
        for _ in range(T):
            q=rope(heads(self.query(Synchronization.read(alpha_a,beta_a)).view(B,S,d)),positions)
            o=F.scaled_dot_product_attention(q,keys,values,is_causal=True).transpose(1,2).reshape(N,d)
            pre=self.synapse(torch.cat((self.attn_out(o),z.to(o.dtype)),dim=-1))
            history=torch.cat((history[...,1:],pre.unsqueeze(-1).to(history.dtype)),dim=-1)
            z=self.nlm(history).float()
            alpha_a,beta_a=self.sync_action.step(alpha_a,beta_a,z);alpha_o,beta_o=self.sync_out.step(alpha_o,beta_o,z)
            tick_logits.append(self.lm_head(Synchronization.read(alpha_o,beta_o)).view(B,S,-1))
        result={'logits':tick_logits[-1],'all_logits':tick_logits,'thought_steps':T}
        if targets is not None:
            if targets.shape!=input_ids.shape:raise ValueError('Targets must be shifted and have the same shape as input_ids')
            result.update(ctm_loss(torch.stack(tick_logits,dim=2),targets))
        return result


def ctm_loss(logits,targets):
    """CTM loss: per token, mean of the min-loss tick's and max-certainty tick's CE. logits: [B, S, T, V]."""
    mask=targets.ne(-100);gold=targets[mask];ticks=logits[mask].float()  # [n, T, V]
    logp=ticks.log_softmax(-1)
    ce=-logp.gather(-1,gold[:,None,None].expand(-1,ticks.shape[1],1)).squeeze(-1)  # [n, T]
    certainty=1+(logp.exp()*logp).sum(-1)/math.log(ticks.shape[-1])                  # 1 - normalized entropy
    index=torch.arange(len(gold),device=gold.device)
    selected=(ce[index,ce.argmin(-1)]+ce[index,certainty.argmax(-1)])/2
    return {'loss':selected.mean(),'per_tick_loss':ce.mean(0),'certainty':certainty.mean(0)}


def lm_config(config):
    """Record CTM's loss in the run config: 'dynamic_aggregate' is min-loss plus max-certainty."""
    return replace(config,temporal_loss_type='dynamic_aggregate',mono_penalty_weight=0.0)


def ctm_lm_factory(size='tiny'):
    """Model factory for runner v3; the name is recorded in each run manifest."""
    def factory(config):return CTMLM(config)
    factory.__qualname__=f'ctm_lm_factory[{size}]'
    return factory
