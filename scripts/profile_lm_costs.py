"""Profile training cost of Transformer, RDT and CTM-LM at language-model scale on one GPU (planning measurement, not a paper endpoint).

For each architecture and size, trains on random tokens (vocabulary 32,768) with
BF16 autocast and AdamW, as in pretraining, and records parameters, the largest
power-of-two batch that fits at the sequence length, throughput (tokens per
second, median over timed updates) and peak memory. RDT is profiled at its mean
training depth (16) and, for memory, at the sampler cap (48). RDT and CTM-LM
use RoPE with no absolute table; the Transformer too.
"""
import argparse,json,statistics,time
from dataclasses import replace
from pathlib import Path
import torch
from ctm_transformer.ctm_lm import lm_config
from ctm_transformer.lm_scale import scaled_factory
from ctm_transformer.positions import position_factory
from ctm_transformer.research import load_research_config

RECIPES={'transformer':'research/configs/presentation_control_v1/transformer_shuffled_seed23.json',
         'rdt':'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json',
         'ctm_lm':'research/configs/presentation_control_v1/ctm_shuffled_seed23.json'}
VOCAB=32768
# Sizes target about 10M, 25M, 50M and 500M non-embedding parameters per family (head width 64).
SIZES={
 'transformer':{'10M':dict(d_model=384,n_layers=6,ffn_hidden_dim=1024),'25M':dict(d_model=512,n_layers=8,ffn_hidden_dim=1408),
                '50M':dict(d_model=640,n_layers=10,ffn_hidden_dim=1728),'500M':dict(d_model=1280,n_layers=24,ffn_hidden_dim=3456)},
 'rdt':{'10M':dict(d_model=384,prelude_layers=1,core_layers=4,coda_layers=1,ffn_hidden_dim=1024),
        '25M':dict(d_model=512,prelude_layers=2,core_layers=4,coda_layers=2,ffn_hidden_dim=1408),
        '50M':dict(d_model=640,prelude_layers=2,core_layers=6,coda_layers=2,ffn_hidden_dim=1728),
        '500M':dict(d_model=2304,prelude_layers=2,core_layers=4,coda_layers=2,ffn_hidden_dim=6144),
        '500M_B':dict(d_model=1280,prelude_layers=11,core_layers=2,coda_layers=11,ffn_hidden_dim=3456)},
 'ctm_lm':{'10M':dict(d_model=384,n_layers=4,d_latent=512,sync_sparse_pairs=512,history_len=8,nlm_hidden_dim=32),
           '25M':dict(d_model=512,n_layers=4,d_latent=1024,sync_sparse_pairs=1024,history_len=8,nlm_hidden_dim=32),
           '50M':dict(d_model=640,n_layers=6,d_latent=1536,sync_sparse_pairs=1536,history_len=8,nlm_hidden_dim=32),
           '500M':dict(d_model=1536,n_layers=12,d_latent=4096,sync_sparse_pairs=4096,history_len=8,nlm_hidden_dim=32),
           '500M_B':dict(d_model=1280,n_layers=24,d_latent=2048,sync_sparse_pairs=2048,history_len=8,nlm_hidden_dim=32)}}


def build(family,size,seq_len,depth,checkpointing=False,scaled=False):
    c,_=load_research_config(RECIPES[family])
    if family=='ctm_lm':c=lm_config(c)
    over=dict(SIZES[family][size]);over['n_heads']=over['d_model']//64
    c=replace(c,vocab_size=VOCAB,max_seq_len=seq_len,seq_len=seq_len,use_positional_encoding=False,max_thought_steps=depth,**over)
    if family=='rdt':c=replace(c,train_depth_min=depth,train_depth_max=depth,gradient_checkpointing=checkpointing)
    if scaled:return scaled_factory(family,checkpointing)(c),c
    return position_factory(family,'rope')(c),c


def count(model):
    total=sum(p.numel() for p in model.parameters())
    embedding=sum(p.numel() for n,p in model.named_parameters() if n.startswith(('token_embedding','lm_head')))
    return total,total-embedding


def trial(family,size,seq_len,depth,batch,device,steps=6,checkpointing=False,scaled=False):
    torch.manual_seed(0);model,config=build(family,size,seq_len,depth,checkpointing,scaled);model=model.to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(0.9,0.95),weight_decay=0.1)
    x=torch.randint(0,VOCAB,(batch,seq_len),device=device);times=[]
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for i in range(steps):
            torch.cuda.synchronize(device);t=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,targets=x,max_thought_steps=depth)['loss']
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            torch.cuda.synchronize(device)
            if i>=2:times.append(time.perf_counter()-t)
        result={'ok':True,'seconds_per_update':statistics.median(times),'tokens_per_second':batch*seq_len/statistics.median(times),
                'peak_gib':torch.cuda.max_memory_allocated(device)/2**30,'loss':float(loss)}
    except torch.cuda.OutOfMemoryError:
        result={'ok':False}
    total,non_embedding=count(model)
    del model,optimizer,x;torch.cuda.empty_cache()
    return {**result,'parameters':total,'non_embedding_parameters':non_embedding}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0');p.add_argument('--seq-len',type=int,default=1024)
    p.add_argument('--families',nargs='+',default=list(SIZES));p.add_argument('--sizes',nargs='+',default=['10M','25M','50M'])
    p.add_argument('--out',default='research/results/lm_cost_profile/profile.json');p.add_argument('--max-batch',type=int,default=64)
    p.add_argument('--scaled',action='store_true',help='use the lm_scale training paths (chunked losses)');p.add_argument('--settings',help='family-independent list depth:ckpt,... overriding the defaults')
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=True
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    rows=json.loads(out.read_text())['rows'] if out.exists() else []
    done={(r['family'],r['size'],r['seq_len'],r['depth'],r['checkpointing'],r.get('scaled',False)) for r in rows}
    for family in a.families:
        for size in a.sizes:
            settings=[(16,False),(48,False),(48,True)] if family=='rdt' else [(16,False)] if family=='ctm_lm' else [(1,False)]
            if a.settings:settings=[(1 if family=='transformer' else int(x.split(':')[0]),x.split(':')[1]=='1') for x in a.settings.split(',')]
            settings=list(dict.fromkeys(settings))
            for depth,ckpt in settings:
                if (family,size,a.seq_len,depth,ckpt,a.scaled) in done:continue
                best=None;batch=1
                while batch<=a.max_batch:
                    r=trial(family,size,a.seq_len,depth,batch,a.device,checkpointing=ckpt,scaled=a.scaled)
                    print(json.dumps({'family':family,'size':size,'depth':depth,'checkpointing':ckpt,'scaled':a.scaled,'batch':batch,**{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()}}),flush=True)
                    if not r['ok']:break
                    best={**r,'batch':batch};batch*=2
                rows.append({'family':family,'size':size,'seq_len':a.seq_len,'depth':depth,'checkpointing':ckpt,'scaled':a.scaled,'max_batch':best['batch'] if best else 0,
                             **({k:best[k] for k in ('tokens_per_second','seconds_per_update','peak_gib','parameters','non_embedding_parameters')} if best else {}),
                             'gpu':torch.cuda.get_device_name(a.device)})
                out.write_text(json.dumps({'role':'planning measurement; random tokens; one GPU; not a paper endpoint','vocab':VOCAB,'sizes':SIZES,'rows':rows},indent=2)+'\n')

if __name__=='__main__':main()
