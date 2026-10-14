"""Split a matched-CTM checkpoint's synchronization readout into the start-state term and the tick terms, and score each (2026-10-14).

Written for research/runs/ctm_matched/M_T16_lr0.004 (latest.pt); prints RMS of z0, of each tick's z, and losses on 4 held-out windows.
"""
import json,torch
from pathlib import Path
from torch.nn import functional as F
from scripts.pretrain import build_config
from ctm_transformer.ctm_lm_adapt import adapted_factory
from ctm_transformer.pretrain import TokenWindows
R=Path('research/runs/ctm_matched/M_T16_lr0.004');run=json.loads((R/'config.json').read_text());data=json.loads(Path(run['data']).read_text())
m=adapted_factory(set(run['ctm_adaptations']))(build_config(run,data['vocab_size']));m.load_state_dict(torch.load(R/'latest.pt',map_location='cpu',weights_only=False)['model'])
dev=torch.device('cuda');m.to(dev).eval()
valid=TokenWindows([Path(run['data']).parent/s for s in data['validation']],1024,0)
rms=lambda t:t.float().pow(2).mean().sqrt().item()
# capture z0 and each tick's z
zs=[];orig=m._tick
def rec(*a,**k):
    o=orig(*a,**k);zs.append(o[0].detach());return o
m._tick=rec
tot={'full':0,'start_only':0,'ticks_only':0,'no_ctm':0};n=0
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for i in range(40000,40004):
        x,y=valid.batch([i]);x,y=x.to(dev),y.to(dev);n+=y.numel();zs.clear()
        stream=m.token_embedding(x)
        for b in m.backbone:stream=b(stream)
        feats=m.backbone_norm(stream);B,S=x.shape;N=B*S
        z0=(m.z_init.expand(N,-1).float()+m.start_projection(feats).reshape(N,-1).float())
        m(x,max_thought_steps=16,return_all_logits=True)
        so=m.sync_out;keep=torch.exp(-so.decay.clamp(0,15)).float()
        def readout(states,start):
            a=b_=None
            for t,z in enumerate(states):
                p=so.products(z);a=p if a is None else keep*a+p
            beta=sum(keep**k for k in range(len(states)))
            return a/torch.sqrt(beta)
        start_p=so.products(z0);tick_p=sum(keep**(15-t)*so.products(z) for t,z in enumerate(zs))
        beta=sum(keep**k for k in range(17))
        parts={'full':(keep**16*start_p+tick_p)/beta.sqrt(),'start_only':keep**16*start_p/beta.sqrt(),'ticks_only':tick_p/beta.sqrt()}
        for k,r in parts.items():
            h=m.final_norm(stream+m.residual_out(r.view(B,S,-1).to(stream.dtype)))
            tot[k]+=float(F.cross_entropy(m.lm_head(h).float().reshape(-1,m.lm_head.out_features),y.reshape(-1),reduction='sum'))
        h=m.final_norm(stream);tot['no_ctm']+=float(F.cross_entropy(m.lm_head(h).float().reshape(-1,m.lm_head.out_features),y.reshape(-1),reduction='sum'))
        if i==40000:print('rms z0',round(rms(z0),3),'rms z per tick',[round(rms(z),3) for z in zs][:4],'... | start products',round(rms(start_p),3),'sum of tick products',round(rms(tick_p),3))
print({k:round(v/n,3) for k,v in tot.items()})
