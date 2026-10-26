"""How does CTsqS seed 1 extrapolate parity with ticks? (2026-10-26)

Analysis of the parity follow-up's checkpoints (scripts/parity_ticks.py). CTsqS seed 1 needs more ticks
for longer inputs and reaches inputs three times the trained length. Controls: CTsqS seed 3 (partial
tick gains), CTsq seed 3 and Dans seed 3 (partial extrapolation at a fixed tick count, no tick gains).

1. Ablations (CTsqS only): accuracy by length band at 16 / 64 / 128 ticks with the sink removed (its gate
   set to zero), the synchronization queries removed (their projections set to zero), and both.
2. Ticks against length: for exact lengths 10-160, the per-tick readout's accuracy at the answer
   position for ticks 1-128, and the first tick reaching 90%.
3. Which bits are processed by tick t: for length 60 (and 80), at each tick, the agreement of the model's
   answer with the parity of the first m bits and of the last m bits, for every m; the best m.
4. Sink use: a zero-value sink scales each head's attention output by (1 - g * w), with g the head's gate
   and w the attention weight on the sink. Reported per layer and tick (length 60): g * w at the answer
   position and averaged over the bit positions.

Usage: python -m scripts.parity_analysis --gpu N
"""
import argparse,json,os,random
from pathlib import Path

from scripts.cmm_tasks import BIT,build,example
from scripts.parity_ticks import BANDS,CKPT,accuracy,band_batch

OUT=Path('research/results/parity_ticks')
RUNS=[('CTsqS',1),('CTsqS',3),('CTsq',3),('Dans',3)]
LENGTHS=(10,20,30,40,50,60,70,80,100,120,140,160);T_MAX=128


def load(cell,seed,device):
    import torch
    model,_=build(cell);model.load_state_dict(torch.load(CKPT/f'{cell}_seed{seed}.pt',map_location='cpu',weights_only=False)['model'])
    return model.to(device).eval()


def fixed_length(length,n,seed):
    import torch
    rng=random.Random(f'analysis-{length}-{seed}');rows=[example('parity',length,rng)[0] for _ in range(n)]
    x=torch.tensor([s[:-1] for s in rows]);answer=torch.tensor([s[-1] for s in rows])
    bits=torch.tensor([[1 if t==BIT[1] else 0 for t in s[1:1+length]] for s in rows])
    return x,answer,bits


def per_tick_predictions(model,x,ticks,device):
    import torch
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(x.to(device),max_thought_steps=ticks,return_all_logits=True)['all_logits']
    return torch.stack([l[:,-1].argmax(-1) for l in logits]).cpu()  # [ticks, n]


def ablations(model,device):
    import torch
    state={k:v.clone() for k,v in model.state_dict().items()};out={}
    for name,params in (('full',[]),('no_sink',['sink_gate']),('no_sync_query',[f'sync_queries.{i}.weight' for i in range(len(model.core))]),
                        ('neither',['sink_gate',*[f'sync_queries.{i}.weight' for i in range(len(model.core))]])):
        with torch.no_grad():
            for p in params:dict(model.named_parameters())[p].zero_()
        out[name]={f'{lo}-{hi}':{str(t):accuracy(model,lo,hi,t,device) for t in (16,64,128)} for lo,hi in BANDS[:5]}
        model.load_state_dict(state)
    return out


def ticks_vs_length(model,device):
    out={}
    for L in LENGTHS:
        x,answer,_=fixed_length(L,256,0);pred=per_tick_predictions(model,x,T_MAX,device)
        acc=[100*float((p==answer).float().mean()) for p in pred]
        out[str(L)]={'first_tick_90':next((t+1 for t,a in enumerate(acc) if a>=90),None),'accuracy_by_tick':[round(a,1) for a in acc]}
    return out


def processed_bits(model,device,L):
    import torch
    x,_,bits=fixed_length(L,512,1);pred=per_tick_predictions(model,x,T_MAX,device)
    pred_bit=(pred==BIT[1]).long()  # 1 = odd
    prefix=torch.cumsum(bits,1)%2;suffix=torch.flip(torch.cumsum(torch.flip(bits,[1]),1),[1])%2  # suffix[:, j] = parity of bits j..L-1
    out={}
    for t in (1,2,4,8,12,16,24,32,48,64,96,128):
        p=pred_bit[t-1]
        first=[float((p==prefix[:,m-1]).float().mean()) for m in range(1,L+1)]
        last=[float((p==suffix[:,L-m]).float().mean()) for m in range(1,L+1)]
        bf=max(range(L),key=lambda i:first[i]);bl=max(range(L),key=lambda i:last[i])
        out[str(t)]={'best_first_m':bf+1,'agreement_first':round(100*first[bf],1),'best_last_m':bl+1,'agreement_last':round(100*last[bl],1),
                     'agreement_full':round(100*first[-1],1)}
    return out


def sink_use(model,device,L=60,ticks=64):
    """g * w per layer and tick, at the answer position and over bit positions, from the sink attention calls."""
    import torch
    from torch.nn import functional as F
    import ctm_transformer.ctm_transformer as ct
    records=[];heads=model.core[0].attn.heads;layers=len(model.core)
    class Proxy:
        def __getattr__(self,name):return getattr(F,name)
        @staticmethod
        def scaled_dot_product_attention(q,k,v,is_causal=False,**kw):
            if is_causal and float(q[:,:,0].abs().sum())==0:  # the sink call: a zero dummy query is prepended
                c=len(records);layer=c%layers
                s=q.shape[-2];logits=(q.float()@k.float().transpose(-1,-2))/q.shape[-1]**0.5
                logits=logits.masked_fill(~torch.ones(s,s,dtype=torch.bool,device=q.device).tril(),float('-inf'))
                w=logits.softmax(-1)[:,:,1:,0]  # [B, H, S] weight on the sink for every real query
                g=model.sink_gate[layer].float()[None,:,None]
                a=(g*w).mean(0)  # [H, S]
                records.append({'layer':layer,'tick':c//layers+1,'answer':a[:,-1].tolist(),'bits':a[:,1:-1].mean(-1).tolist()})
            return F.scaled_dot_product_attention(q,k,v,is_causal=is_causal,**kw)
    x,_,_=fixed_length(L,64,2);old=ct.F;ct.F=Proxy()
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):model(x.to(device),max_thought_steps=ticks)
    finally:ct.F=old
    return {'gate':model.sink_gate.float().tolist(),'records':records}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch;device=torch.device('cuda')
    for cell,seed in RUNS:
        model=load(cell,seed,device);r={'cell':cell,'seed':seed}
        if cell=='CTsqS':r['ablations']=ablations(model,device);r['sink_use']=sink_use(model,device)
        r['ticks_vs_length']=ticks_vs_length(model,device);r['processed_bits']={str(L):processed_bits(model,device,L) for L in (60,80)}
        (OUT/f'analysis_{cell}_seed{seed}.json').write_text(json.dumps(r,indent=1)+'\n')
        print(cell,seed,'first tick reaching 90% by length:',{L:v['first_tick_90'] for L,v in r['ticks_vs_length'].items()},flush=True)
        if 'ablations' in r:print('  ablations 41-60 / 61-80 / 81-120 at 64 ticks:',{k:[round(v[b]['64']) for b in ('41-60','61-80','81-120')] for k,v in r['ablations'].items()},flush=True)
        print('  L=60 processed bits by tick (best first m, agreement / best last m, agreement / full):',
              {t:(v['best_first_m'],v['agreement_first'],v['best_last_m'],v['agreement_last'],v['agreement_full']) for t,v in r['processed_bits']['60'].items()},flush=True)


if __name__=='__main__':main()
