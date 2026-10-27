"""Parity follow-up: is CTsqS's tick-scaling length extrapolation reliable? (2026-10-25)

In the CMM-task grid (scripts/cmm_tasks.py), one of 72 recurrent runs gained accuracy with more ticks:
CTsqS parity seed 1, on lengths 41-80 (trained 1-40), 54.7% at 16 ticks, 89.6% at 32, 98.4% at 64.
Its other seeds and every other model were flat. That grid saved no checkpoints.

This retrains parity exactly as in that grid (same task, model, data per seed and training) for
CTsqS, CTsq (without the sink) and Dans (neither mechanism), on seeds 0-9, saving the final
checkpoint. Seeds 0-2 repeat the grid's runs (GPU nondeterminism may make them differ).

Evaluation of each checkpoint: sequence accuracy on 512 fixed examples per length band (1-40, 41-60,
61-80, 81-120, 121-160, 161-250) at 1, 4, 8, 16, 32, 64 and 128 ticks.

Rules, fixed before the runs:
- A run extrapolates with ticks if its accuracy on lengths 41-80 at 64 ticks is at least 80% and at
  least 20 points above its accuracy at 16 ticks.
- A model has the property reliably if at least 3 of its 10 seeds extrapolate with ticks, and it has
  it more than another model if its count exceeds the other's by at least 3.
- Reported: for each run, accuracy by length and ticks; for extrapolating runs, the smallest tick count
  reaching 90% in each band, which shows whether the ticks needed grow with the length (a serial
  algorithm) or not.

More training ticks (option 2, added 2026-10-26, after scripts/parity_analysis.py): the extrapolating run
needs about 0.55 ticks per bit, so about 22 for the trained maximum of 40 bits, while the tick sampler
(mean 15) gives fewer on most updates. CTsq_m32 and Dans_m32 train with log-normal Poisson ticks of mean 32,
at most 96 (everything else unchanged), on seeds 0-9. The mean-15 arm is the runs above (CTsq, Dans).
Rules, fixed before these runs: the rules above, per cell; the sampler helps a model if its m32 count
exceeds its mean-15 count by at least 3; the synchronization queries matter under m32 if CTsq_m32's count
exceeds Dans_m32's by at least 3. Secondary (reported, since a model trained with more ticks may use more
ticks per bit): the same extrapolation rule with 128 ticks in place of 64.

Length curriculum (added 2026-10-26, after the mean-32 sampler did not help): CTsq_cur and Dans_cur train for
10,000 updates (twice as long; the learning-rate schedule spans them) with the maximum training length
growing linearly from 4 to 40 bits over the first 60% of updates (each example's length uniform up to the
current maximum), then 1-40 as before; ticks of mean 15 as in the first arm. Seeds 0-9. This changes two
things at once (curriculum and length of training) and is judged as one intervention. Rules, fixed before
these runs: the rules above per cell; the curriculum helps a model if its count exceeds its mean-15 count by
at least 3; the synchronization queries matter under it if CTsq_cur's count exceeds Dans_cur's by at least 3.

Leon (2026) mechanisms (stage 1, added 2026-10-27; ctm_transformer/ctm_transformer.py): homeostasis and
structural plasticity on the thinking layers' feed-forward neurons, in the first arm's setting (5,000
updates, ticks of mean 15, no curriculum). CTsqH = CTsq + homeostasis; CTsqHS = CTsq + homeostasis +
structural plasticity; DansH = Dans + homeostasis. Seeds 0-9. Rules, fixed before these runs: the rules
above per cell; a mechanism helps a model if its count exceeds the model's own first-arm count (CTsq 0,
Dans 0) by at least 3; it needs the synchronization queries if CTsqH's count exceeds DansH's by at least 3;
structural plasticity adds to homeostasis if CTsqHS's count exceeds CTsqH's by at least 3. Optimizer:
AdamW, as everywhere here (the paper used SGD and reports Adam as worse).

Usage: python -m scripts.parity_ticks --gpu N --runs CELL:SEED... | --summarize
"""
import argparse,json,os,random,time
from pathlib import Path

from scripts.cmm_tasks import BATCH,STEPS,build,example

OUT=Path('research/results/parity_ticks');CKPT=Path('research/runs/parity_ticks')
CELLS=('CTsqS','CTsq','Dans','CTsq_m32','Dans_m32','CTsq_cur','Dans_cur','CTsqH','CTsqHS','DansH');SEEDS=range(10)
SAMPLERS={'':(15,48),'_m32':(32,96),'_cur':(15,48)}  # tick sampler (mean, maximum) by cell suffix
CURRICULUM_STEPS=10000


def curriculum_max(step,steps):
    return min(40,4+int(36*step/(0.6*steps)))


def split(cell):
    base,_,suffix=cell.partition('_');return base,('_'+suffix if suffix else '')
BANDS=((1,40),(41,60),(61,80),(81,120),(121,160),(161,250));TICKS=(1,4,8,16,32,64,128);EXAMPLES=512


def band_batch(lo,hi,n,rng):
    import torch
    rows=[example('parity',rng.randint(lo,hi),rng)[0] for _ in range(n)];length=max(len(s) for s in rows)-1
    x=torch.zeros((n,length),dtype=torch.long);y=torch.full((n,length),-100)
    for i,s in enumerate(rows):x[i,:len(s)-1]=torch.tensor(s[:-1]);y[i,len(s)-2]=s[-1]
    return x,y


def accuracy(model,lo,hi,ticks,device):
    import torch
    rng=random.Random(f'parity-band-{lo}-{hi}');hits=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(EXAMPLES//64):
            x,y=band_batch(lo,hi,64,rng);x,y=x.to(device),y.to(device)
            pred=model(x,max_thought_steps=ticks)['logits'].argmax(-1);mask=y.ne(-100)
            hits+=int(((pred==y)&mask).sum())
    return 100*hits/EXAMPLES


def train(cell,seed,device):
    """The CMM-task grid's training loop (scripts/cmm_tasks.train), saving the final checkpoint."""
    import torch
    from ctm_transformer.depth_sampling import lognormal_poisson
    from ctm_transformer.pretrain import lr_at,set_lr,step_depth
    from scripts.cmm_tasks import batch
    out=OUT/f'{cell}_seed{seed}.json'
    if out.exists():return
    base,suffix=split(cell);mean,maximum=SAMPLERS[suffix]
    torch.manual_seed(seed);model,recurrent=build(base);model=model.to(device);assert recurrent
    sampler=lognormal_poisson(mean,0.5,maximum)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    rng=random.Random(f'train-parity-{seed}');start=time.time();steps=CURRICULUM_STEPS if suffix=='_cur' else STEPS
    for step in range(steps):
        set_lr(opt,lr_at(step,1e-3,200,steps,0.1))
        x,y=band_batch(1,curriculum_max(step,steps),BATCH,rng) if suffix=='_cur' else batch('parity','train',BATCH,rng);x,y=x.to(device),y.to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,targets=y,max_thought_steps=step_depth(sampler,seed,step))['loss']
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad(set_to_none=True)
        if hasattr(model,'bio_step'):model.bio_step(step,steps)
    CKPT.mkdir(parents=True,exist_ok=True);torch.save({'model':model.state_dict(),'cell':cell,'seed':seed},CKPT/f'{cell}_seed{seed}.pt')
    model.eval();grid={f'{lo}-{hi}':{str(t):accuracy(model,lo,hi,t,device) for t in TICKS} for lo,hi in BANDS}
    result={'cell':cell,'seed':seed,'seconds':time.time()-start,'final_loss':float(loss),'accuracy':grid}
    OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=1)+'\n')
    print(cell,seed,f"{result['seconds']:.0f}s",'41-60 / 61-80 at 16/64 ticks:',[round(grid[b][t],1) for b in ('41-60','61-80') for t in ('16','64')],flush=True)


def extrapolates(r,ticks='64'):
    g=r['accuracy'];band=lambda t:(g['41-60'][t]+g['61-80'][t])/2  # lengths 41-80: equal-size halves of the band pair
    return band(ticks)>=80 and band(ticks)-band('16')>=20,band('16'),band(ticks)


def summarize():
    res={(r['cell'],r['seed']):r for r in (json.loads(f.read_text()) for f in OUT.glob('*_seed*.json') if not f.name.startswith('analysis_'))}
    L=['# Parity follow-up: tick-scaling length extrapolation','','Protocol and rules: `scripts/parity_ticks.py` (docstring).','',
       '| Model | Seeds | Extrapolating seeds | Accuracy on 41-80 at 16 / 64 ticks, per seed |','|---|---:|---:|---|']
    counts={}
    for c in CELLS:
        rs=[res[(c,s)] for s in SEEDS if (c,s) in res];ex=[extrapolates(r) for r in rs];counts[c]=sum(e[0] for e in ex)
        L.append(f'| {c} | {len(rs)} | {counts[c]} | '+', '.join(f'{a:.0f}/{b:.0f}{"*" if e else ""}' for e,a,b in ex)+' |')
    L+=['',f'Reliable (at least 3 of 10): '+', '.join(f'{c} {"yes" if counts[c]>=3 else "no"}' for c in CELLS)]
    L.append('Secondary (128 ticks in place of 64): '+', '.join(f'{c} {sum(extrapolates(res[(c,s)],"128")[0] for s in SEEDS if (c,s) in res)}' for c in CELLS))
    for a in CELLS:
        for b in CELLS:
            if a!=b and counts[a]-counts[b]>=3:L.append(f'- {a} has the property more than {b} ({counts[a]} vs {counts[b]})')
    L+=['','## Extrapolating runs: smallest tick count reaching 90% per length band','']
    for (c,s),r in sorted(res.items()):
        if extrapolates(r)[0]:
            need=[next((t for t in map(str,TICKS) if r['accuracy'][f'{lo}-{hi}'][t]>=90),'-') for lo,hi in BANDS]
            L.append(f'- {c} seed {s}: '+', '.join(f'{lo}-{hi}: {n}' for (lo,hi),n in zip(BANDS,need)))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'PARITY_TICKS.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--runs',nargs='*',default=[])
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch
    for r in a.runs:c,s=r.split(':');train(c,int(s),torch.device('cuda'))


if __name__=='__main__':main()
