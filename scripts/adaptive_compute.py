"""Adaptive compute from certainty: does the CTM Transformer reach its loss with fewer ticks than Dans? (2026-10-18)

The CTM's claim is that a model with internal ticks can stop thinking early on easy tokens. This
analysis uses only trained checkpoints. For each model, every held-out token's readout is computed at
every tick 1..T_MAX (the coda and the output head applied to that tick's state, as in training), with:
- cross-entropy (CE) and certainty = 1 - normalized entropy (the CTM's definition) per token and tick.

Readouts:
- fixed T: every token read at tick T;
- threshold exit: each token is read at the first tick whose certainty reaches tau (at T_MAX if none), for
  a sweep of tau; reported as (mean exit tick, CE);
- most certain: each token read at its most certain tick among 1..16 (the CTM's label-free readout).

Caveat: every tick is run for every position, since later positions attend to earlier positions' states.
The exit tick measures how much thinking a token's own prediction needs, not wall-clock savings, which
would need cached states for exited positions. Dans and the CTM Transformer are measured identically.

Usage: python -m scripts.adaptive_compute --gpu N --runs DIR... [--windows 64]
"""
import argparse,json,os
from pathlib import Path

T_MAX=32
TAUS=(0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
OUT=Path('research/results/adaptive_compute')


def load(run_dir,device):
    import torch
    from scripts.pretrain import build_config
    from ctm_transformer.rdt_recipe import recipe_factory
    from ctm_transformer.ctm_transformer import ctm_transformer_factory
    run=json.loads((run_dir/'config.json').read_text());data=json.loads(Path(run['data']).read_text())
    config=build_config(run,data['vocab_size'])
    factory=ctm_transformer_factory(set(run['ctm_transformer']),run['rdt_recipe']) if run.get('ctm_transformer') is not None else recipe_factory(run['rdt_recipe'])
    model=factory(config);t=run['train']
    model.load_state_dict(torch.load(run_dir/f'step_{t["total_steps"]:07d}.pt',map_location='cpu',weights_only=False)['model'])
    return model.to(device).eval(),run,data


def analyse(run_dir,windows,device):
    import torch
    from torch.nn import functional as F
    from ctm_transformer.pretrain import TokenWindows
    model,run,data=load(run_dir,device);t=run['train']
    valid=TokenWindows([Path(run['data']).parent/s for s in data['validation']],t['seq_len'],0);offset=t.get('eval_offset',0)
    ce_all,cert_all=[],[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(offset,offset+windows):
            x,y=valid.batch([i]);x,y=x.to(device),y.to(device)
            logits=model(x,max_thought_steps=T_MAX,return_all_logits=True)['all_logits']
            ce,cert=[],[]
            for l in logits:
                logp=l.float().reshape(-1,l.shape[-1]).log_softmax(-1)
                ce.append(-logp.gather(-1,y.reshape(-1,1)).squeeze(-1))
                cert.append(1+(logp.exp()*logp).sum(-1)/torch.log(torch.tensor(float(l.shape[-1]))))
            ce_all.append(torch.stack(ce));cert_all.append(torch.stack(cert))  # [T, S]
    ce=torch.cat(ce_all,1);cert=torch.cat(cert_all,1)  # [T, tokens]
    fixed={str(k+1):float(ce[k].mean()) for k in range(T_MAX)}
    exits={}
    for tau in TAUS:
        reached=cert>=tau
        first=torch.where(reached.any(0),reached.float().argmax(0),torch.full_like(reached[0],T_MAX-1,dtype=torch.long))
        exits[str(tau)]={'mean_exit_tick':float(first.float().mean()+1),'ce':float(ce.gather(0,first[None]).mean())}
    best=cert[:16].argmax(0)
    result={'run':run_dir.name,'windows':windows,'tokens':int(ce.shape[1]),'fixed':fixed,'threshold_exit':exits,
            'most_certain_1_16':{'mean_tick':float(best.float().mean()+1),'ce':float(ce[:16].gather(0,best[None]).mean())}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'{run_dir.name}.json').write_text(json.dumps(result,indent=1)+'\n')
    print(run_dir.name,'fixed 1/2/4/8/16:',[round(fixed[k],3) for k in ('1','2','4','8','16')],
          '| exits:',[(round(v['mean_exit_tick'],2),round(v['ce'],3)) for v in exits.values()],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--windows',type=int,default=64);a=p.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch
    for r in a.runs:analyse(Path(r),a.windows,torch.device('cuda'))


if __name__=='__main__':main()
