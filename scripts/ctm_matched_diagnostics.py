"""Where does the matched CTM's capacity go? (research/RDT_RECIPE.md, 2026-10-14)

For a trained matched-CTM checkpoint, on held-out windows:
- the loss with the CTM's contribution to the residual stream removed (residual_out zeroed), so
  the logits come from the 4-layer backbone alone;
- the RMS of the backbone stream, of the CTM's write into it, of the neuron state z and of
  the synchronization readout, per tick;
- the loss at each tick;
- the synchronization decays: the fraction at exactly 0 (no forgetting).

Usage: python -m scripts.ctm_matched_diagnostics --run NAME [--checkpoint latest.pt] [--windows 8] [--ticks 1 4 16]
"""
import argparse,json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.pretrain import build_config
from ctm_transformer.ctm_lm_adapt import adapted_factory
from ctm_transformer.ctm_lm import Synchronization
from ctm_transformer.pretrain import TokenWindows

RUNS=Path('research/runs/ctm_matched');OUT=Path('research/results/ctm_matched')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--checkpoint',default=None)
    p.add_argument('--windows',type=int,default=8);p.add_argument('--ticks',type=int,nargs='+',default=None);a=p.parse_args()
    run=json.loads((RUNS/a.run/'config.json').read_text());t=run['train']
    data=json.loads(Path(run['data']).read_text());root=Path(run['data']).parent
    model=adapted_factory(set(run['ctm_adaptations']),unet_width=run.get('unet_width'))(build_config(run,data['vocab_size']))
    ckpt=a.checkpoint or (f'step_{t["total_steps"]:07d}.pt' if (RUNS/a.run/f'step_{t["total_steps"]:07d}.pt').exists() else 'latest.pt')
    payload=torch.load(RUNS/a.run/ckpt,map_location='cpu',weights_only=False);model.load_state_dict(payload['model']);step=payload['step']
    device=torch.device('cuda');model.to(device).eval()
    valid=TokenWindows([root/s for s in data['validation']],t['seq_len'],0);offset=t.get('eval_offset',0)
    T=run['model']['max_thought_steps'];ticks=a.ticks or sorted({1,T,2*T} if T>1 else {1})
    stats={'stream':[],'write':[],'z':[],'sync':[]};losses={};ablated={};n=0
    hooks={}
    original_tick=model._tick
    def recording_tick(*args,**kw):
        out=original_tick(*args,**kw);z,readout=out[0],out[6]
        hooks.setdefault('z',[]).append(z.float().pow(2).mean().sqrt().item());hooks.setdefault('sync',[]).append(readout.float().pow(2).mean().sqrt().item())
        return out
    model._tick=recording_tick
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(offset,offset+a.windows):
            x,y=valid.batch([i]);x,y=x.to(device),y.to(device);n+=y.numel()
            for k in ticks:
                hooks.clear()
                out=model(x,max_thought_steps=k,return_all_logits=True)
                losses.setdefault(k,[0.0]*k)
                for j,l in enumerate(out['all_logits']):losses[k][j]+=float(F.cross_entropy(l.float().reshape(-1,l.shape[-1]),y.reshape(-1),reduction='sum'))
                if k==max(ticks):
                    stats['z'].append(hooks['z']);stats['sync'].append(hooks['sync'])
            # backbone stream and the CTM's write at the largest tick count
            stream=model.token_embedding(x)
            for block in model.backbone:stream=block(stream)
            stats['stream'].append(stream.float().pow(2).mean().sqrt().item())
            hooks.clear();k=max(ticks)
            readouts=[]
            model._tick=lambda *args,**kw:(lambda o:(readouts.append(o[6]),o)[1])(original_tick(*args,**kw))
            model(x,max_thought_steps=k,return_all_logits=True);model._tick=recording_tick
            stats['write'].append([model.residual_out(r.view(*stream.shape[:2],-1).to(stream.dtype)).float().pow(2).mean().sqrt().item() for r in readouts])
            # the backbone alone: the CTM's write removed
            logits=model.lm_head(model.final_norm(stream))
            ablated['backbone_only']=ablated.get('backbone_only',0.0)+float(F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum'))
    mean=lambda rows:[sum(c)/len(c) for c in zip(*rows)]
    decays={name:(s.decay.detach().cpu()) for name,s in (('action',model.sync_action),('out',model.sync_out))}
    result={'run':a.run,'step':step,'windows':a.windows,
            'loss_by_tick':{str(k):[v/n for v in vals] for k,vals in losses.items()},
            'loss_backbone_only':ablated['backbone_only']/n,
            'rms':{'backbone_stream':sum(stats['stream'])/len(stats['stream']),'ctm_write_by_tick':mean(stats['write']),'z_by_tick':mean(stats['z']),'sync_readout_by_tick':mean(stats['sync'])},
            'decay':{k:{'fraction_at_zero':float((v<=0).float().mean()),'mean':float(v.clamp(0,15).mean()),'max':float(v.max())} for k,v in decays.items()},
            'residual_out_weight_rms':model.residual_out.weight.float().pow(2).mean().sqrt().item()}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'diagnostics_{a.run}_step{step}.json').write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps(result,indent=1))


if __name__=='__main__':main()
